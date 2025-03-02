import os
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
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel,
    ExpSineSquared,
    Matern,
    RationalQuadratic,
    WhiteKernel,
)
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader, Dataset, TensorDataset
from xgboost import XGBClassifier

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


def load_data():
    # Load the default dataset
    data = pd.read_excel(
        "data/HEA-class-1.xlsx",
        skiprows=[
            1,
        ],
    )

    # composition labels - 我们要保留的元素
    comp_labels = [
        "C",
        "Al",
        "Si",
        "Ti",
        "Ni",
        "Cu",
        "Hf",
        "Co",
        "Zr",
        "Mo",
        "Fe",
        "Pd",
        "Ta",
        "Nb",
        "V",
        "Cr",
        "Mn",
        "Ta",
        "W",
    ]
    print(len(comp_labels))

    # 1. 获取表格中所有的列名
    all_cols = data.columns.tolist()
    print(f"所有列名：{all_cols}")

    # 2. 除去标签列y
    cols_without_y = [col for col in all_cols if col != "y"]
    print(f"除去y后的列名：{cols_without_y}")

    # 3. 找到不在comp_labels中的列
    other_cols = [col for col in cols_without_y if col not in comp_labels]
    print(f"需要检查的其他列：{other_cols}")
    print("\n其他列中非零值的统计：")
    for col in other_cols:
        non_zero_count = (data[col] != 0).sum()
        if non_zero_count > 0:
            print(f"{col}: {non_zero_count}行含有非零值")

    # 4. 检查这些列中数据不为0的行并删除
    rows_with_other_elements = (data[other_cols] != 0).any(axis=1)
    data = data[~rows_with_other_elements]
    print(f"\n删除含有其他元素的数据行数：{sum(rows_with_other_elements)}行")
    print(f"保留的数据行数：{sum(~rows_with_other_elements)}行")

    # 分类标签列
    class_label = "y"

    # 将分类标签转换为数值
    label_encoder = preprocessing.LabelEncoder()
    class_data = label_encoder.fit_transform(data[class_label])
    print(
        f"类别映射: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}"
    )

    comp_data = data[comp_labels].to_numpy()

    # 读取元素特征数据
    elem_feature_ep = pd.read_excel("data/ELEMENTS-19.xlsx")
    elem_feature_ep = elem_feature_ep[comp_labels].to_numpy()

    elem_feature_bt = pd.read_excel(
        "data/ele_vec.xlsx"
    )  # NOTE: bert embedding features
    elem_feature_bt = elem_feature_bt[comp_labels].to_numpy()

    return comp_data, class_data, elem_feature_ep, elem_feature_bt


def cv_accuracy(
    x,
    y,
    model,
    split_n,
    kf_random_state: int,
) -> float:
    """
    Custom n-fold cross validation for classification.
    使用F1分数来评估分类性能。
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

    return f1_score(y_test_buff, y_pred_buff, average="weighted")  # 使用加权F1分数


def rf_par_cv_accuracy(
    x,
    y,
    random_seed_list: List[int],
    n_split=10,
) -> float:
    return Parallel(n_jobs=-1)(
        delayed(cv_accuracy)(
            x,
            y,
            get_random_forest(),  # 需要改用分类器
            n_split,
            rs,
        )
        for rs in random_seed_list
    )


def gp_par_cv_accuracy(
    x,
    y,
    random_seed_list: List[int],
    n_split=10,
) -> float:
    return Parallel(n_jobs=-1)(
        delayed(cv_accuracy)(
            x,
            y,
            get_gpc(),
            n_split,
            rs,
        )
        for rs in random_seed_list
    )


if __name__ == "__main__":
    prop_id = "classification_1"  # property id
    f_n = int(sys.argv[1])  # number of features to select

    """ 1. 准备数据 """
    comp_data, class_data, elem_feature_ep, elem_feature_bt = load_data()
    wavg_feature_ep = np.dot(comp_data, elem_feature_ep.T)
    wavg_feature_bt = np.dot(comp_data, elem_feature_bt.T)

    for traj_id in range(48):
        # 每次循环都会重新进行特征选择和评估
        # 生成唯一实验ID
        exp_id = str(uuid.uuid4())[:8]

        """ 2. 随机选取100个成分-性能数据 """
        selected_index = np.random.choice(len(comp_data), 100, replace=False)
        sel_comp_data = comp_data[selected_index]
        sel_wavg_feature_ep = wavg_feature_ep[selected_index]
        sel_wavg_feature_bt = wavg_feature_bt[selected_index]
        sel_class_data = class_data[selected_index]

        """ 3. 使用GA选取最合适的特征 """
        env = Env(sel_wavg_feature_ep, sel_class_data, f_n=f_n)
        ga = FeatureSelectionGA(env, verbose=1)
        ga.generate(n_pop=192, cxpb=0.8, mutxpb=0.1, ngen=100)
        sel_f_idx_list_ep = ga.get_fitest_ind().f_idx_list

        """ 4. 使用empirical feature评估不同模型的准确率 """
        random_seeds_list = [np.random.randint(0, 999) for _ in range(48)]

        wavg_to_eval_ep = (sel_wavg_feature_ep.T[sel_f_idx_list_ep]).T

        res_buff_ep_rf = rf_par_cv_accuracy(
            wavg_to_eval_ep, sel_class_data, random_seeds_list
        )

        # res_buff_ep_gp = gp_par_cv_accuracy(
        #     wavg_to_eval_ep, sel_class_data, random_seeds_list
        # )

        """ 5. 对于bert embedding feature评估不同模型的准确率 """
        env = Env(sel_wavg_feature_bt, sel_class_data, f_n=f_n)
        ga = FeatureSelectionGA(env, verbose=1)
        ga.generate(n_pop=192, cxpb=0.8, mutxpb=0.1, ngen=100)
        sel_f_idx_list_bt = ga.get_fitest_ind().f_idx_list

        wavg_to_eval_bt = (sel_wavg_feature_bt.T[sel_f_idx_list_bt]).T

        res_buff_bt_rf = rf_par_cv_accuracy(
            wavg_to_eval_bt, sel_class_data, random_seeds_list
        )

        # res_buff_bt_gp = gp_par_cv_accuracy(
        #     wavg_to_eval_bt, sel_class_data, random_seeds_list
        # )

        # 保存所有模型的结果
        joblib.dump(
            {
                "empirical": {
                    "rf": res_buff_ep_rf,
                    # "gp": res_buff_ep_gp,
                },
                "bert": {
                    "rf": res_buff_bt_rf,
                    # "gp": res_buff_bt_gp,
                },
            },
            f"{prop_id}-{f_n}-{traj_id}-{str(uuid.uuid4())[:8]}.pkl",
        )
