import random
import sys
import uuid
from collections import Counter
from copy import deepcopy
from typing import Callable, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from joblib import Parallel, delayed
from sklearn import preprocessing
from sklearn.base import RegressorMixin, TransformerMixin, clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel,
    ExpSineSquared,
    Matern,
    RationalQuadratic,
    WhiteKernel,
)
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, TensorDataset

from feature_selection_ga import *
from model_env import N_ELEM, N_ELEM_FEAT, N_ELEM_FEAT_P1, N_PROC, CnnDnnModel


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
set_seed(0)  # default 0

seeds = np.random.randint(0, 9999, (9999,))


def load_data(prop_id: int):
    # Load the default dataset
    prop_labels = ["YTS", "UTS", "VH"][prop_id]
    # prop_labels = [prop_labels]
    data = pd.read_csv(f"data/{prop_labels}.csv")  # EL, UTS, VH, YTS

    # composition labels    # NOTE
    comp_labels = [
        "Ti",
        "Al",
        "Zr",
        "Mo",
        "V",
        "Ta",
        "Nb",
        "Cr",
        "Mn",
        "Fe",
        "Sn",
    ]
    unacc_elem = ["Bi", "O", "N", "Si"]

    # clean data
    _acc_elem_mask = ~(data[unacc_elem] != 0).any(axis=1)
    data = data[_acc_elem_mask]  # if contains un-wanted elem

    _sum = data[comp_labels].sum(axis=1).round(4)
    _sum_eq1_mask = ((99.9 <= _sum) & (_sum <= 100.1)) | (
        (0.998 < _sum) & (_sum <= 1.02)
    )
    data = data[_sum_eq1_mask]

    # processing condition labels
    proc_labels = [
        "temp.hold",
        "time.hold",
        "deformation.amount",
        "temp.hold1",
        "time.hold1",
        "deformation.amount1",
        "temp.hold2",
        "time.hold2",
    ]

    # property labels
    # prop_labels: EL, UTS, VH, YTS
    print(f"loading {prop_labels} data ...")

    # processing condition labels # NOTE
    # proc_labels = ['Hom_Temp(K)', 'CR(%)', 'Anneal_Temp(K)', 'Anneal_Time(h)']

    # property labels, one by one for
    # prop_labels = ['YS(Mpa)']   # NOTE
    # print(f'loading SMA data ...')

    """ delete proc condition all 0 items """
    # _mask = (data[proc_labels[1:]] == 0).all(axis = 1)
    # data = data[~ _mask]
    # print(f'deleted {sum(_mask)} items')

    """ delete non usual proc conditions """
    # _mask = (data[proc_labels[0]] == 0.) & (data[proc_labels[1]] == 1273.) & (data[proc_labels[2]] == 1)
    # data = data[_mask]
    # print(f'deleted {sum(~ _mask)} items')
    # print(f'remains {sum(_mask)} items')

    comp_data = data[comp_labels].to_numpy()
    proc_data = data[proc_labels].to_numpy().round(8)
    proc_data_key = list("*".join(map(str, item)) for item in proc_data)
    most_common_key = Counter(proc_data_key).most_common()[0][0]
    sel_idx = list(map(lambda x: most_common_key == x, proc_data_key))
    print(f"selected {sum(sel_idx)} rows with identical process conditions.")

    prop_data = data[prop_labels].to_numpy().ravel()

    elem_feature_ep = pd.read_excel("data/elemental_features.xlsx")
    elem_feature_ep = elem_feature_ep[comp_labels].to_numpy()

    elem_feature_bt = pd.read_excel(
        "data/ele_vec.xlsx"
    )  # NOTE: bert embedding features
    elem_feature_bt = elem_feature_bt[comp_labels].to_numpy()

    comp_data /= comp_data.sum(axis=1).reshape(-1, 1)
    comp_data = comp_data.round(8)

    # (num_samples, num_elements), (num_samples, num_proc), (num_samples, num_prop), (num_elements, num_elem_features,)
    return (
        comp_data[sel_idx],
        proc_data[sel_idx],
        prop_data[sel_idx],
        elem_feature_ep,
        elem_feature_bt,
    )


def cv_mae(
    x,
    y,
    model: RandomForestRegressor,
    split_n,
    kf_random_state: int,  # TODO: no fixed random state
) -> float:
    """
    Custom 10-fold cross validation.
    """
    assert len(x) == len(y)

    # thread local variables
    x = deepcopy(x)
    y = deepcopy(y)
    model = clone(model)

    scaler = preprocessing.RobustScaler()
    x = scaler.fit_transform(x)
    kf = KFold(split_n, shuffle=True, random_state=kf_random_state)

    y_test_buff, y_pred_buff = [], []
    for train_index, test_index in kf.split(x):
        X_train, X_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        y_test_buff += y_test.tolist()
        y_pred_buff += y_pred.tolist()

    return mean_absolute_error(y_test_buff, y_pred_buff)


def rf_par_cv_mae(
    x,
    y,
    random_seed_list: List[int],
    n_split=10,
) -> float:
    return Parallel(n_jobs=-1)(
        delayed(cv_mae)(
            x,
            y,
            get_random_forest(),
            n_split,
            rs,
        )
        for rs in random_seed_list
    )


def gp_par_cv_mae(
    x,
    y,
    random_seed_list: List[int],
    n_split=10,
) -> float:
    return Parallel(n_jobs=-1)(
        delayed(cv_mae)(
            x,
            y,
            get_gpr(),
            n_split,
            rs,
        )
        for rs in random_seed_list
    )


def mlp_par_cv_mae(
    x,
    y,
    random_seed_list: List[int],
    n_split=10,
) -> float:
    return Parallel(n_jobs=-1)(
        delayed(cv_mae)(
            x,
            y,
            get_mlp(),
            n_split,
            rs,
        )
        for rs in random_seed_list
    )


def svr_par_cv_mae(
    x,
    y,
    random_seed_list: List[int],
    n_split=10,
) -> float:
    return Parallel(n_jobs=-1)(
        delayed(cv_mae)(
            x,
            y,
            get_svr(),
            n_split,
            rs,
        )
        for rs in random_seed_list
    )


def xgb_par_cv_mae(
    x,
    y,
    random_seed_list: List[int],
    n_split=10,
) -> float:
    return Parallel(n_jobs=-1)(
        delayed(cv_mae)(
            x,
            y,
            get_xgb(),
            n_split,
            rs,
        )
        for rs in random_seed_list
    )


if __name__ == "__main__":
    prop_id = int(sys.argv[1])  # property id
    f_n = int(sys.argv[2])  # number of features to select

    """ 1. 准备数据 """
    comp_data, _, prop_data, elem_feature_ep, elem_feature_bt = load_data(
        prop_id
    )  # ~ 158 data points
    wavg_feature_ep = np.dot(comp_data, elem_feature_ep.T)
    wavg_feature_bt = np.dot(comp_data, elem_feature_bt.T)

    for traj_id in range(8):
        """ 2. 随机选取100个成分-性能数据 """
        selected_index = np.random.choice(len(comp_data), 68, replace=False)
        sel_comp_data = comp_data[selected_index]
        sel_wavg_feature_ep = wavg_feature_ep[selected_index]
        sel_wavg_feature_bt = wavg_feature_bt[selected_index]
        sel_prop_data = prop_data[selected_index]

        """ 3. 使用GA选取最合适的4个weighted avg feature """
        env = Env(sel_wavg_feature_ep, sel_prop_data, f_n=f_n)  # NOTE
        ga = FeatureSelectionGA(env, verbose=1)
        ga.generate(n_pop=192, cxpb=0.8, mutxpb=0.1, ngen=100)
        sel_f_idx_list_ep = ga.get_fitest_ind().f_idx_list

        """ 4. 使用empirical feature评估不同模型的mae """
        random_seeds_list = [np.random.randint(0, 999) for _ in range(64)]

        wavg_to_eval_ep = (sel_wavg_feature_ep.T[sel_f_idx_list_ep]).T
        res_buff_ep_rf = rf_par_cv_mae(
            wavg_to_eval_ep, sel_prop_data, random_seeds_list
        )
        res_buff_ep_gp = gp_par_cv_mae(
            wavg_to_eval_ep, sel_prop_data, random_seeds_list
        )
        res_buff_ep_mlp = mlp_par_cv_mae(
            wavg_to_eval_ep, sel_prop_data, random_seeds_list
        )
        res_buff_ep_svr = svr_par_cv_mae(
            wavg_to_eval_ep, sel_prop_data, random_seeds_list
        )
        res_buff_ep_xgb = xgb_par_cv_mae(
            wavg_to_eval_ep, sel_prop_data, random_seeds_list
        )

        """ 5. 对于bert embedding feature评估不同模型的mae """
        env = Env(sel_wavg_feature_bt, sel_prop_data, f_n=f_n)
        ga = FeatureSelectionGA(env, verbose=1)
        ga.generate(n_pop=192, cxpb=0.8, mutxpb=0.1, ngen=100)
        sel_f_idx_list_bt = ga.get_fitest_ind().f_idx_list

        wavg_to_eval_bt = (sel_wavg_feature_bt.T[sel_f_idx_list_bt]).T
        res_buff_bt_rf = rf_par_cv_mae(
            wavg_to_eval_bt, sel_prop_data, random_seeds_list
        )
        res_buff_bt_gp = gp_par_cv_mae(
            wavg_to_eval_bt, sel_prop_data, random_seeds_list
        )
        res_buff_bt_mlp = mlp_par_cv_mae(
            wavg_to_eval_bt, sel_prop_data, random_seeds_list
        )
        res_buff_bt_svr = svr_par_cv_mae(
            wavg_to_eval_bt, sel_prop_data, random_seeds_list
        )
        res_buff_bt_xgb = xgb_par_cv_mae(
            wavg_to_eval_bt, sel_prop_data, random_seeds_list
        )

        # 保存所有模型的结果
        joblib.dump(
            {
                "empirical": {
                    "rf": res_buff_ep_rf,
                    "gp": res_buff_ep_gp,
                    "mlp": res_buff_ep_mlp,
                    "svr": res_buff_ep_svr,
                    "xgb": res_buff_ep_xgb,
                },
                "bert": {
                    "rf": res_buff_bt_rf,
                    "gp": res_buff_bt_gp,
                    "mlp": res_buff_bt_mlp,
                    "svr": res_buff_bt_svr,
                    "xgb": res_buff_bt_xgb,
                },
            },
            f"{prop_id}-{f_n}-{traj_id}-{str(uuid.uuid4())[:8]}.pkl",
        )
