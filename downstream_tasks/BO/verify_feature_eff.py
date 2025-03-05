from collections import Counter
from copy import deepcopy
import random
import sys
from typing import Callable, Tuple
import uuid
import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset

from sklearn import preprocessing
from sklearn.model_selection import KFold, cross_val_score
from sklearn.base import RegressorMixin, clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.base import TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel, RBF, WhiteKernel, Matern, ExpSineSquared, RationalQuadratic
)

from joblib import Parallel, delayed

from model_env import N_ELEM, N_ELEM_FEAT, N_ELEM_FEAT_P1, N_PROC
from model_env import CnnDnnModel
from feature_selection_ga import *

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 设置随机数种子
set_seed(0) # default 0

seeds = np.random.randint(0, 9999, (9999, ))

def load_data(prop_id: int):
    # Load the default dataset
    data = pd.read_excel('data/DSC-DATA-0613.xlsx', skiprows=[1,])
    
    # composition labels 
    comp_labels = ['Ti', 'Ni', 'Cu', 'Hf', 'Co', 'Zr', 'Fe', 'Pd', 'Ta', 'Nb', 'V']
    
    # processing condition labels 
    proc_labels = ['冷轧变形量', '退火处理温度', '退火处理时间']

    # property labels, one by one for ['enthalpy（Heating）', 'Ms', 'Mp', 'Mf', 'As', 'Ap', 'Af']
    prop_labels = ['Mp', 'Ap', 'enthalpy（Heating）']
    prop_labels = [prop_labels[prop_id]]
    print(f'loading SMA data ...')

    ''' delete proc condition all 0 items '''
    _mask = (data[proc_labels[1:]] == 0).all(axis = 1)
    data = data[~ _mask]
    print(f'deleted {sum(_mask)} items')

    ''' delete non usual proc conditions '''
    _mask = (data[proc_labels[0]] == 0.) & (data[proc_labels[1]] == 1273.) & (data[proc_labels[2]] == 1)
    data = data[_mask]
    print(f'deleted {sum(~ _mask)} items')
    print(f'remains {sum(_mask)} items')

    comp_data = data[comp_labels].to_numpy()
    proc_data = data[proc_labels].to_numpy()
    prop_data = data[prop_labels].to_numpy().ravel()

    elem_feature_ep = pd.read_excel('data/sma_element_features.xlsx')
    elem_feature_ep = elem_feature_ep[comp_labels].to_numpy()  # transpose: column for each elemental feature, row for each element

    elem_feature_bt = pd.read_excel('data/ele_vec.xlsx')   # NOTE: bert embedding features
    elem_feature_bt = elem_feature_bt[comp_labels].to_numpy()

    return comp_data, proc_data, prop_data, elem_feature_ep, elem_feature_bt

def cv_mae(x, 
           y, 
           model: RandomForestRegressor, 
           split_n, 
           kf_random_state: int    # TODO: no fixed random state
           ) -> float:
    ''' 
        Custom 10-fold cross validation.
    '''
    assert len(x) == len(y)

    # thread local variables
    x = deepcopy(x)
    y = deepcopy(y)
    model = clone(model)

    scaler = preprocessing.RobustScaler()
    x = scaler.fit_transform(x)
    kf = KFold(split_n, shuffle = True, random_state = kf_random_state)

    y_test_buff, y_pred_buff = [], []
    for train_index, test_index in kf.split(x):
        X_train, X_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        y_test_buff += y_test.tolist()
        y_pred_buff += y_pred.tolist()

    return mean_absolute_error(y_test_buff, y_pred_buff)
    # return r2_score(y_test_buff, y_pred_buff)

def par_cv_mae(x, 
               y,  
               random_seed_list: List[int],
               n_split = 10,) -> float:
    return  Parallel(n_jobs = -1)(delayed(cv_mae)(x, 
                                                  y, 
                                                  get_random_forest(),
                                                  n_split, 
                                                  rs, ) for rs in random_seed_list)

if __name__ == '__main__':
    prop_id = int(sys.argv[1])  # property id
    f_n = int(sys.argv[2])  # number of features to select

    ''' 1. 准备数据 '''
    comp_data, _, prop_data, elem_feature_ep, elem_feature_bt = load_data(prop_id)     # ~ 158 data points
    wavg_feature_ep = np.dot(comp_data, elem_feature_ep.T)
    wavg_feature_bt = np.dot(comp_data, elem_feature_bt.T)

    for traj_id in range(1):
        ''' 2. 随机选取100个成分-性能数据 '''
        selected_index = np.random.choice(len(comp_data), 100, replace = False)
        sel_comp_data = comp_data[selected_index]
        sel_wavg_feature_ep = wavg_feature_ep[selected_index]
        sel_wavg_feature_bt = wavg_feature_bt[selected_index]
        sel_prop_data = prop_data[selected_index]

        ''' 3. 使用GA选取最合适的4个weighted avg feature '''
        env = Env(sel_wavg_feature_ep, sel_prop_data, f_n = f_n)   # NOTE
        ga = FeatureSelectionGA(env, verbose = 1)
        ga.generate(n_pop = 192, cxpb = 0.8, mutxpb = 0.1, ngen = 100)
        sel_f_idx_list_ep = ga.get_fitest_ind().f_idx_list

        ''' 4. 使用empirical feature作为输入x评估模型的mae, 比如64次 '''
        random_seeds_list = [np.random.randint(0, 999) for _ in range(64)]
    
        wavg_to_eval_ep = (sel_wavg_feature_ep.T[sel_f_idx_list_ep]).T
        res_buff_ep = par_cv_mae(wavg_to_eval_ep, sel_prop_data, random_seeds_list)

        ''' 5. 对于bert embedding feature, 重复3-4步 '''
        env = Env(sel_wavg_feature_bt, sel_prop_data, f_n = f_n)   # NOTE
        ga = FeatureSelectionGA(env, verbose = 1)
        ga.generate(n_pop = 192, cxpb = 0.8, mutxpb = 0.1, ngen = 100)
        sel_f_idx_list_bt = ga.get_fitest_ind().f_idx_list

        wavg_to_eval_bt = (sel_wavg_feature_bt.T[sel_f_idx_list_bt]).T
        res_buff_bt = par_cv_mae(wavg_to_eval_bt, sel_prop_data, random_seeds_list)

        ''' 6. 使用comp作为输入评估10-cv '''
        res_buff_comp = par_cv_mae(sel_comp_data, sel_prop_data, random_seeds_list)

        joblib.dump(
            (res_buff_comp, res_buff_ep, res_buff_bt),
            f'r2-{prop_id}-{f_n}-{traj_id}-{str(uuid.uuid4())[:8]}.pkl'
        )
