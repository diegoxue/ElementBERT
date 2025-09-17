import os
import json
import random
import sys
import glob
import numpy as np
import pandas as pd
import joblib
from typing import List, Tuple, Callable
from copy import deepcopy
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge, HuberRegressor, QuantileRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C
from sklearn.neural_network import MLPRegressor
from sklearn.cross_decomposition import PLSRegression
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn import preprocessing
from sklearn.base import clone
from sklearn.model_selection import KFold, GroupKFold, train_test_split, RandomizedSearchCV, RepeatedKFold
from sklearn.metrics import mean_absolute_error, r2_score
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from feature_selection_ga import *

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
set_seed(0)  # default 0

seeds = np.random.randint(0, 9999, (9999,))

def load_data(prop_id: int, feature_kind: str):
    """
    loading datasets and features from files
    
    Args:
        prop_id: property ID, 0=YTS, 1=UTS, 2=VH
        
    Returns:
        comp_data: composition data
        proc_data: processing conditions data
        prop_data: property data
        elem_feature: element features data
    """
    # loading datasets
    prop_labels = ["YTS", "UTS"][prop_id]
    data = pd.read_csv(f"data/{prop_labels}.csv")

    # composition labels
    comp_labels = [
        "Ti", "Al", "Zr", "Mo", "V", "Nb", "Cr", "Mn", "Fe", "Sn",'Si',
        # 'Cu'
    ]
    unacc_elem = ["Bi", "O", "N"]

    # clean data
    # backfill missing composition columns with 0 to avoid KeyError (e.g., 'Cu' not present in CSV)
    # missing_cols = [c for c in comp_labels if c not in data.columns]
    # if missing_cols:
    #     for c in missing_cols:
    #         data[c] = 0.0
    #     try:
    #         print(f"[Warn] 缺失的元素列已用0填充: {missing_cols}")
    #     except Exception:
    #         pass

    _acc_elem_mask = ~(data[unacc_elem] != 0).any(axis=1)
    data = data[_acc_elem_mask]  # filter out data with unwanted elements

    _sum = data[comp_labels].sum(axis=1).round(4)
    _sum_eq1_mask = ((99.9 <= _sum) & (_sum <= 100.1)) | ((0.998 < _sum) & (_sum <= 1.02))
    data = data[_sum_eq1_mask]  # ensure composition sum is close to 100% or 1

    # processing conditions labels
    proc_labels = [
        "temp.hold", "time.hold", "deformation.amount",
        "temp.hold1", "time.hold1", "deformation.amount1",
        "temp.hold2", "time.hold2",
    ]

    print(f"loading {prop_labels} data...")

    # extract data
    comp_data = data[comp_labels].to_numpy()
    proc_data = data[proc_labels].to_numpy().round(8)
    proc_data_key = list("*".join(map(str, item)) for item in proc_data)
    
    # select the most common processing conditions
    from collections import Counter
    most_common_key = Counter(proc_data_key).most_common()[0][0]
    sel_idx = list(map(lambda x: most_common_key == x, proc_data_key))
    print(f"selected {sum(sel_idx)} rows with the same processing conditions.")

    prop_data = data[prop_labels].to_numpy().ravel()
    if feature_kind == "extend_ep":
        # loading extend empirical features, keep consistent with feature selection
        elem_feature = pd.read_excel("data/extend_ep_19.xlsx")
        elem_feature = elem_feature[comp_labels].to_numpy()
    elif feature_kind == "bt":
        # loading BERT-based features
        elem_feature = pd.read_excel("data/ElementBERT_ele_vec.xlsx")
        elem_feature = elem_feature[comp_labels].to_numpy()
    elif feature_kind == "tradition_ep":
        # loading tradition empirical features
        elem_feature = pd.read_excel("data/Element_features-19.xlsx")
        elem_feature = elem_feature[comp_labels].to_numpy()
    else:
        raise ValueError("feature_kind 必须是 'bt' / 'extend_ep' / 'tradition_ep'")

    # normalize composition data
    comp_data /= comp_data.sum(axis=1).reshape(-1, 1)
    comp_data = comp_data.round(8)

    return (
        comp_data[sel_idx],
        proc_data[sel_idx],
        prop_data[sel_idx],
        elem_feature,
    )

def _get_best_indices_from_evolution(method: str, prop_id: int, f_n: int = 6) -> List[int]:
    """
    从对应 evolution 目录中获取给定方法与属性的最佳特征索引（按最高 fitness 的轨迹）。
    method: 'bt' / 'extend_ep' / 'tradition_ep'
    返回 indices 列表。
    """
    if method == "bt":
        evo_root = "Ti_bt_evolution_history"
        tag = "bt"
    elif method == "extend_ep":
        evo_root = "Ti_extend_ep_evolution_history"
        tag = "ep"
    elif method == "tradition_ep":
        evo_root = "Ti_tradition_ep_evolution_history"
        tag = "ep"
    else:
        raise ValueError("method 必须是 'bt' / 'extend_ep' / 'tradition_ep'")

    pattern = os.path.join(evo_root, f"evolution-{tag}-{prop_id}-{f_n}-*-*.pkl")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"未找到进化历史: {pattern}")

    # 加载，寻找最大 fitness 的 Individual
    best_indices = None
    best_fit = -np.inf
    for fp in files:
        buff = joblib.load(fp)
        for _, ind in buff.items():
            fit = getattr(ind, "fitness", None)
            if fit is None:
                continue
            if fit > best_fit:
                best_fit = fit
                best_indices = list(map(int, ind.f_idx_list.tolist()))
    if best_indices is None:
        raise RuntimeError("未能从进化历史中解析最佳个体")
    return best_indices

# def load_feature_id(feat_path: str, prop_id: int):
#     feat_df = pd.read_csv(feat_path)
#     feat_list = feat_df["feature"].tolist()
#     feat_id = feat_list[prop_id]
#     print(f"加载 {feat_id} 特征...")
#     return 

def get_random_forest() -> RandomForestRegressor:
    """
    获取RandomForestRegressor对象用于回归

    Returns:
        RandomForestRegressor对象
    """
    return RandomForestRegressor(
        n_estimators=100,
        random_state=43,
    )

def get_extra_trees() -> ExtraTreesRegressor:
    """
    获取ExtraTreesRegressor对象
    """
    return ExtraTreesRegressor(
        n_estimators=300,
        random_state=43,
        n_jobs=-1,
    )

def get_gradient_boosting() -> GradientBoostingRegressor:
    """
    获取GradientBoostingRegressor对象
    """
    return GradientBoostingRegressor(
        random_state=43,
    )

def get_hist_gradient_boosting() -> HistGradientBoostingRegressor:
    """
    获取HistGradientBoostingRegressor对象
    """
    return HistGradientBoostingRegressor(
        random_state=43,
    )

def get_adaboost() -> AdaBoostRegressor:
    """
    获取AdaBoostRegressor对象
    """
    return AdaBoostRegressor(
        random_state=43,
    )

def get_bagging_tree() -> BaggingRegressor:
    """
    获取以决策树为基学习器的BaggingRegressor对象
    """
    return BaggingRegressor(
        estimator=DecisionTreeRegressor(random_state=43),  # Updated from base_estimator
        n_estimators=200,
        random_state=43,
        n_jobs=1,  # Changed from -1 to avoid parallel processing issues
    )

def get_decision_tree() -> DecisionTreeRegressor:
    """
    获取DecisionTreeRegressor对象
    """
    return DecisionTreeRegressor(
        random_state=43,
    )

def get_knn() -> KNeighborsRegressor:
    """
    获取KNeighborsRegressor对象
    """
    return KNeighborsRegressor(
        n_neighbors=5,
        weights="distance",
        n_jobs=-1,
    )

def get_svr_rbf() -> SVR:
    """
    获取RBF核的SVR对象（需标准化/稳健缩放）
    """
    return SVR(
        kernel="rbf",
        C=10.0,
        gamma="scale",
        epsilon=0.1,
    )

def get_linear_svr() -> LinearSVR:
    """
    获取LinearSVR对象（需标准化/稳健缩放）
    """
    return LinearSVR(
        C=10.0,
        epsilon=0.1,
        random_state=43,
        max_iter=10000,
    )

def get_kernel_ridge_rbf() -> KernelRidge:
    """
    获取RBF核的KernelRidge对象（需标准化/稳健缩放）
    """
    return KernelRidge(
        kernel="rbf",
        alpha=1.0,
        gamma=None,
    )

def get_ridge() -> Ridge:
    """
    获取Ridge回归对象（需标准化/稳健缩放）
    """
    return Ridge(
        alpha=1.0,
        random_state=43,
    )

def get_lasso() -> Lasso:
    """
    获取Lasso回归对象（需标准化/稳健缩放）
    """
    return Lasso(
        alpha=0.001,
        random_state=43,
        max_iter=10000,
    )

def get_elastic_net() -> ElasticNet:
    """
    获取ElasticNet回归对象（需标准化/稳健缩放）
    """
    return ElasticNet(
        alpha=0.001,
        l1_ratio=0.5,
        random_state=43,
        max_iter=10000,
    )

def get_bayesian_ridge() -> BayesianRidge:
    """
    获取BayesianRidge回归对象（需标准化/稳健缩放）
    """
    return BayesianRidge()

def get_huber() -> HuberRegressor:
    """
    获取HuberRegressor鲁棒回归对象（需标准化/稳健缩放）
    """
    return HuberRegressor(
        epsilon=1.35,
        max_iter=1000,
    )

# def get_quantile_regressor() -> QuantileRegressor:
#     """
#     获取QuantileRegressor对象（需标准化/稳健缩放）
#     默认中位数回归(quantile=0.5)
#     """
#     return QuantileRegressor(
#         quantile=0.5,
#         alpha=0.01,
#         solver="interior-point",  # More stable solver
#         max_iter=500,     # Reduced iterations for stability
#     )

def get_gaussian_process() -> GaussianProcessRegressor:
    """
    获取GaussianProcessRegressor对象（适合小样本；需标准化/稳健缩放）
    """
    kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0)
    return GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-6,
        normalize_y=True,
        random_state=43,
    )

def get_mlp() -> MLPRegressor:
    """
    获取MLPRegressor对象（需标准化/稳健缩放）
    """
    return MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        learning_rate_init=1e-3,
        max_iter=500,
        early_stopping=True,
        random_state=43,
    )

def get_pls() -> PLSRegression:
    """
    获取PLSRegression对象（需标准化/稳健缩放）
    """
    return PLSRegression(
        n_components=2,
    )

def get_xgb_regressor() -> XGBRegressor:
    """
    获取XGBRegressor对象（需要安装 xgboost）。
    若未安装将抛出 ImportError。
    """
    return XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=43,
        n_jobs=-1,
        tree_method="hist",
    )

def get_lgbm_regressor() -> LGBMRegressor:
    """
    获取LGBMRegressor对象（需要安装 lightgbm）。
    若未安装将抛出 ImportError。
    """
    return LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=-1,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=43,
        n_jobs=-1,
    )

def get_catboost_regressor() -> CatBoostRegressor:
    """
    获取CatBoostRegressor对象（需要安装 catboost）。
    若未安装将抛出 ImportError。
    """
    return CatBoostRegressor(
        iterations=2000,
        learning_rate=0.03,
        depth=6,
        loss_function="RMSE",
        random_seed=43,
        verbose=False,
    )

# def tune_random_forest(X_train_scaled, y_train):
#     """
#     随机搜索优化 RandomForest 超参数，返回最优模型与参数。
#     使用 RepeatedKFold 与 MAE 作为评分。
#     """
#     base_rf = RandomForestRegressor(random_state=43, n_jobs=-1)
#     param_dist = {
#         # 使用离散候选集合，避免引入 scipy 依赖
#         "n_estimators": list(range(300, 1001, 50)),
#         "max_depth": [None] + list(range(8, 41, 4)),
#         "max_features": ["sqrt", "log2", 0.4, 0.6, 0.8, 1.0],
#         "min_samples_split": [2, 5, 10],
#         "min_samples_leaf": [1, 2, 4],
#         "bootstrap": [True, False],
#     }
#     cv = RepeatedKFold(n_splits=5, n_repeats=2, random_state=42)
#     search = RandomizedSearchCV(
#         base_rf,
#         param_distributions=param_dist,
#         n_iter=50,
#         scoring={"mae": "neg_mean_absolute_error", "r2": "r2"},
#         refit="r2",
#         cv=cv,
#         n_jobs=-1,
#         verbose=1,
#         random_state=42,
#     )
#     search.fit(X_train_scaled, y_train)
#     best_model = search.best_estimator_
#     best_params = search.best_params_
#     best_index = search.best_index_
#     # 将 MAE 取反为正数
#     best_cv_mae = -search.cv_results_["mean_test_mae"][best_index]
#     best_cv_r2 = search.cv_results_["mean_test_r2"][best_index]
#     return best_model, best_params, {"cv_mae": best_cv_mae, "cv_r2": best_cv_r2}

def tune_random_forest_with_cv(X_train_scaled, y_train, inner_cv, inner_groups=None):
    """
    使用给定的内层 CV(可为 GroupKFold/KFold) 对随机森林进行随机搜索调参。
    支持传入 groups（当 inner_cv 需要分组时）。
    返回最优模型、最优参数，以及内层交叉验证上的评估（R²/MAE）。
    """
    base_rf = RandomForestRegressor(random_state=43, n_jobs=-1)
    param_dist = {
        "n_estimators": list(range(50, 500, 50)),  # 包含默认值100
        "max_depth": [None] + list(range(8, 41, 4)),
        "max_features": ["sqrt", "log2", 0.4, 0.6, 0.8, 1.0],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "bootstrap": [True, False],
    }
    search = RandomizedSearchCV(
        base_rf,
        param_distributions=param_dist,
        n_iter=50,
        scoring={"mae": "neg_mean_absolute_error", "r2": "r2"},
        refit="r2",
        cv=inner_cv,
        n_jobs=-1,
        verbose=0,
        random_state=42,
    )
    if inner_groups is not None:
        search.fit(X_train_scaled, y_train, groups=inner_groups)
    else:
        search.fit(X_train_scaled, y_train)

    best_model = search.best_estimator_
    best_params = search.best_params_
    best_index = search.best_index_
    best_cv_mae = -search.cv_results_["mean_test_mae"][best_index]
    best_cv_r2 = search.cv_results_["mean_test_r2"][best_index]
    return best_model, best_params, {"cv_mae": best_cv_mae, "cv_r2": best_cv_r2}

def _build_composition_groups(comp_data: np.ndarray, rounding_decimals: int = 8) -> np.ndarray:
    """
    基于成分向量构造分组标签。相同成分（按指定小数位四舍五入）将被分到同一组。
    返回长度为样本数的一维整数数组作为 groups。
    """
    if comp_data is None or len(comp_data) == 0:
        return None
    # 生成签名字符串
    rounded = np.round(comp_data.astype(float), decimals=rounding_decimals)
    sig_strings = ["|".join(map(lambda v: f"{v:.{rounding_decimals}f}", row)) for row in rounded]
    # 映射到连续整数标签
    sig_to_gid = {}
    groups = np.zeros(len(sig_strings), dtype=int)
    next_gid = 0
    for i, sig in enumerate(sig_strings):
        if sig not in sig_to_gid:
            sig_to_gid[sig] = next_gid
            next_gid += 1
        groups[i] = sig_to_gid[sig]
    return groups

def _print_duplicate_composition_stats(comp_data: np.ndarray, rounding_decimals: int = 8) -> None:
    """
    打印相同成分（按 rounding_decimals 四舍五入）出现次数的统计。
    """
    if comp_data is None or len(comp_data) == 0:
        print("[Composition] 无可用成分数据用于统计")
        return
    rounded = np.round(comp_data.astype(float), decimals=rounding_decimals)
    sig_strings = ["|".join(map(lambda v: f"{v:.{rounding_decimals}f}", row)) for row in rounded]
    from collections import Counter
    cnt = Counter(sig_strings)
    dup_counts = [c for c in cnt.values() if c > 1]
    if len(dup_counts) == 0:
        print("[Composition] 未检测到重复成分（在指定精度下）")
        return
    print(f"[Composition] 检测到重复成分组数: {len(dup_counts)}，重复样本总数: {sum(dup_counts) - len(dup_counts)}")
    topk = sorted(cnt.items(), key=lambda x: x[1], reverse=True)[:5]
    print("[Composition] 示例（前5个重复组计数）：", [c for _, c in topk])

def _deduplicate_by_composition_mean(X: np.ndarray,
                                     y: np.ndarray,
                                     comp_data: np.ndarray,
                                     rounding_decimals: int = 8) -> tuple:
    """
    对完全相同成分（按指定精度）进行分组平均，返回去重后的 X, y, comp_data。
    """
    rounded = np.round(comp_data.astype(float), decimals=rounding_decimals)
    sig_strings = ["|".join(map(lambda v: f"{v:.{rounding_decimals}f}", row)) for row in rounded]
    sig_to_indices = {}
    for idx, sig in enumerate(sig_strings):
        sig_to_indices.setdefault(sig, []).append(idx)
    X_new, y_new, comp_new = [], [], []
    for _, indices in sig_to_indices.items():
        X_new.append(np.mean(X[indices], axis=0))
        y_new.append(np.mean(y[indices]))
        comp_new.append(np.mean(comp_data[indices], axis=0))
    return np.array(X_new), np.array(y_new), np.array(comp_new)

def nested_cv_evaluate(X: np.ndarray,
                       y: np.ndarray,
                       comp_data_for_group: np.ndarray = None,
                       outer_splits: int = 5,
                       inner_splits: int = 3,
                       use_grouping: bool = True,
                       group_rounding_decimals: int = 8,
                       deduplicate: bool = False,
                       verbose: bool = True,
                       random_state: int = 42) -> dict:
    """
    嵌套交叉验证评估：外层评估、内层调参。
    若 use_grouping=True 且存在重复成分，则使用 GroupKFold 保证相同成分不跨训练/测试。
    返回包含每折与总体统计的结果字典。
    """
    assert len(X) == len(y)

    # 可选：先对完全相同成分做均值去重
    original_len = len(y)
    if deduplicate and comp_data_for_group is not None:
        X, y, comp_data_for_group = _deduplicate_by_composition_mean(
            X, y, comp_data_for_group, rounding_decimals=group_rounding_decimals
        )
        if verbose:
            print(f"[Deduplicate] 按成分聚合均值后样本数: {len(y)} (原始 {original_len})")

    # 计算分组标签（若需要）
    groups_all = None
    grouped = False
    if use_grouping and comp_data_for_group is not None:
        groups_all = _build_composition_groups(comp_data_for_group, rounding_decimals=group_rounding_decimals)
        unique_group_count = len(np.unique(groups_all))
        if unique_group_count < len(y):
            grouped = True
        # 决定外层折数（不超过唯一组数）
        if grouped:
            outer_splits = max(2, min(outer_splits, unique_group_count))

    # 外层 CV
    if grouped:
        outer_cv = GroupKFold(n_splits=outer_splits)
        outer_split_iter = outer_cv.split(X, y, groups_all)
    else:
        outer_cv = KFold(n_splits=outer_splits, shuffle=True, random_state=random_state)
        outer_split_iter = outer_cv.split(X, y)

    outer_mae_list, outer_r2_list = [], []
    outer_params_list = []
    fold_id = 0
    for split in outer_split_iter:
        train_index, test_index = split
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # 标准化仅在训练集上拟合
        scaler = preprocessing.RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 内层 CV（与外层一致的分组策略）
        inner_groups = None
        if grouped:
            groups_train = groups_all[train_index]
            inner_unique = len(np.unique(groups_train))
            inner_k = max(2, min(inner_splits, inner_unique))
            inner_cv = GroupKFold(n_splits=inner_k)
            inner_groups = groups_train
        else:
            inner_cv = KFold(n_splits=inner_splits, shuffle=True, random_state=random_state)

        # # 调参并获取最优模型
        # best_model, best_params, inner_cv_best = tune_random_forest_with_cv(
        #     X_train_scaled, y_train, inner_cv=inner_cv, inner_groups=inner_groups
        # )
        # outer_params_list.append(best_params)
        
        # 使用固定超参数的随机森林
        best_model = get_random_forest()
        best_model.fit(X_train_scaled, y_train)
        outer_params_list.append(best_model.get_params())

        # 外层评估
        y_pred = best_model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        outer_mae_list.append(mae)
        outer_r2_list.append(r2)

        if verbose:
            fold_id += 1
            # print(f"[Outer Fold {fold_id}] R²={r2:.4f}, MAE={mae:.4f}; InnerCV best R²={inner_cv_best['cv_r2']:.
            # 4f}, MAE={inner_cv_best['cv_mae']:.4f}")
            print(f"[Outer Fold {fold_id}] R²={r2:.4f}, MAE={mae:.4f}")

    summary = {
        "grouped": grouped,
        "outer_splits": outer_splits,
        "inner_splits": inner_splits,
        "deduplicated": bool(deduplicate),
        "group_rounding_decimals": int(group_rounding_decimals),
        "r2_mean": float(np.mean(outer_r2_list)) if outer_r2_list else None,
        "r2_std": float(np.std(outer_r2_list)) if outer_r2_list else None,
        "mae_mean": float(np.mean(outer_mae_list)) if outer_mae_list else None,
        "mae_std": float(np.std(outer_mae_list)) if outer_mae_list else None,
        "per_fold_r2": outer_r2_list,
        "per_fold_mae": outer_mae_list,
        "per_fold_params": outer_params_list,
    }

    if verbose:
        print("")
        print("外层评估（Nested CV）:")
        print(f"R²: {summary['r2_mean']:.4f} ± {summary['r2_std']:.4f}")
        print(f"MAE: {summary['mae_mean']:.4f} ± {summary['mae_std']:.4f}")
        print(f"分组评估: {'是' if grouped else '否'} (若是，则按成分签名进行 GroupKFold)")
        if deduplicate:
            print("已对完全相同成分的样本进行分组均值去重")

    return summary

def nested_cv_evaluate_with_model(X: np.ndarray,
                                  y: np.ndarray,
                                  model,
                                  comp_data_for_group: np.ndarray = None,
                                  outer_splits: int = 5,
                                  inner_splits: int = 3,
                                  use_grouping: bool = True,
                                  group_rounding_decimals: int = 8,
                                  deduplicate: bool = False,
                                  verbose: bool = False,
                                  random_state: int = 42) -> dict:
    """
    针对给定模型执行嵌套交叉验证（外层评估、固定超参）。
    与 nested_cv_evaluate 基本一致，但使用传入的模型实例。
    """
    assert len(X) == len(y)

    original_len = len(y)
    if deduplicate and comp_data_for_group is not None:
        X, y, comp_data_for_group = _deduplicate_by_composition_mean(
            X, y, comp_data_for_group, rounding_decimals=group_rounding_decimals
        )
        if verbose:
            print(f"[Deduplicate] 按成分聚合均值后样本数: {len(y)} (原始 {original_len})")

    groups_all = None
    grouped = False
    if use_grouping and comp_data_for_group is not None:
        groups_all = _build_composition_groups(comp_data_for_group, rounding_decimals=group_rounding_decimals)
        unique_group_count = len(np.unique(groups_all))
        if unique_group_count < len(y):
            grouped = True
        if grouped:
            outer_splits = max(2, min(outer_splits, unique_group_count))

    if grouped:
        outer_cv = GroupKFold(n_splits=outer_splits)
        outer_split_iter = outer_cv.split(X, y, groups_all)
    else:
        outer_cv = KFold(n_splits=outer_splits, shuffle=True, random_state=random_state)
        outer_split_iter = outer_cv.split(X, y)

    outer_mae_list, outer_r2_list = [], []
    outer_params_list = []
    fold_id = 0
    for split in outer_split_iter:
        train_index, test_index = split
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        scaler = preprocessing.RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        m = clone(model)
        m.fit(X_train_scaled, y_train)
        outer_params_list.append(getattr(m, 'get_params', lambda: {})())

        y_pred = m.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        outer_mae_list.append(mae)
        outer_r2_list.append(r2)

        if verbose:
            fold_id += 1
            print(f"[Outer Fold {fold_id}] R²={r2:.4f}, MAE={mae:.4f}")

    summary = {
        "grouped": grouped,
        "outer_splits": outer_splits,
        "inner_splits": inner_splits,
        "deduplicated": bool(deduplicate),
        "group_rounding_decimals": int(group_rounding_decimals),
        "r2_mean": float(np.mean(outer_r2_list)) if outer_r2_list else None,
        "r2_std": float(np.std(outer_r2_list)) if outer_r2_list else None,
        "mae_mean": float(np.mean(outer_mae_list)) if outer_mae_list else None,
        "mae_std": float(np.std(outer_mae_list)) if outer_mae_list else None,
        "per_fold_r2": outer_r2_list,
        "per_fold_mae": outer_mae_list,
        "per_fold_params": outer_params_list,
    }
    return summary

def get_model_factories() -> List[Tuple[str, Callable[[], object]]]:
    """
    返回模型工厂列表：(模型名, 构造函数)
    """
    factories: List[Tuple[str, Callable[[], object]]] = [
        ("RandomForest", get_random_forest),
        ("ExtraTrees", get_extra_trees),
        ("GradientBoosting", get_gradient_boosting),
        ("HistGradientBoosting", get_hist_gradient_boosting),
        ("AdaBoost", get_adaboost),
        ("BaggingTree", get_bagging_tree),
        ("DecisionTree", get_decision_tree),
        ("KNN", get_knn),
        ("SVR_RBF", get_svr_rbf),
        ("LinearSVR", get_linear_svr),
        ("KernelRidge_RBF", get_kernel_ridge_rbf),
        ("Ridge", get_ridge),
        ("Lasso", get_lasso),
        ("ElasticNet", get_elastic_net),
        ("BayesianRidge", get_bayesian_ridge),
        ("Huber", get_huber),
        # ("QuantileRegressor", get_quantile_regressor),  # Removed due to instability
        ("GaussianProcess", get_gaussian_process),
        ("MLP", get_mlp),
        ("PLS", get_pls),
        ("XGBoost", get_xgb_regressor),
        ("LightGBM", get_lgbm_regressor),
        ("CatBoost", get_catboost_regressor),
    ]
    return factories

def compare_models(prop_id: int,
                   feature_kind: str,
                   selected_indices: List[int],
                   outer_splits: int = 5,
                   inner_splits: int = 3,
                   use_grouping: bool = True,
                   group_rounding_decimals: int = 8,
                   deduplicate: bool = False,
                   verbose: bool = False,
                   random_state: int = 42) -> pd.DataFrame:
    """
    对多个模型进行嵌套CV比较，并将结果保存为CSV。
    返回包含汇总结果的数据框。
    """
    comp_data, proc_data, prop_data, elem_feature = load_data(prop_id, feature_kind)
    wavg_feature = np.dot(comp_data, elem_feature.T)
    if not selected_indices:
        raise ValueError("selected_indices 不能为空")
    wavg_feature = wavg_feature[:, selected_indices]

    try:
        _print_duplicate_composition_stats(comp_data, rounding_decimals=group_rounding_decimals)
    except Exception:
        pass

    results = []
    per_fold_max = 0
    for model_name, factory in get_model_factories():
        try:
            model_instance = factory()
        except Exception as e:
            print(f"[Skip] 构造 {model_name} 失败: {e}")
            continue
        try:
            summary = nested_cv_evaluate_with_model(
                X=wavg_feature,
                y=prop_data,
                model=model_instance,
                comp_data_for_group=comp_data,
                outer_splits=outer_splits,
                inner_splits=inner_splits,
                use_grouping=use_grouping,
                group_rounding_decimals=group_rounding_decimals,
                deduplicate=deduplicate,
                verbose=verbose,
                random_state=random_state,
            )
            per_fold_max = max(per_fold_max, len(summary.get("per_fold_r2", [])))
            results.append((model_name, summary))
            print(f"[Done] {model_name}: R²={summary['r2_mean']:.4f}±{summary['r2_std']:.4f}, MAE={summary['mae_mean']:.4f}±{summary['mae_std']:.4f}")
        except Exception as e:
            print(f"[Error] 评估 {model_name} 失败: {e}")
            continue

    # 构建DataFrame
    rows = []
    for model_name, summary in results:
        row = {
            "model": model_name,
            "r2_mean": summary.get("r2_mean"),
            "r2_std": summary.get("r2_std"),
            "mae_mean": summary.get("mae_mean"),
            "mae_std": summary.get("mae_std"),
            "grouped": summary.get("grouped"),
            "outer_splits": summary.get("outer_splits"),
            "inner_splits": summary.get("inner_splits"),
            "deduplicated": summary.get("deduplicated"),
        }
        r2_list = summary.get("per_fold_r2", [])
        mae_list = summary.get("per_fold_mae", [])
        for i in range(per_fold_max):
            row[f"r2_fold_{i+1}"] = r2_list[i] if i < len(r2_list) else np.nan
            row[f"mae_fold_{i+1}"] = mae_list[i] if i < len(mae_list) else np.nan
        rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs("models", exist_ok=True)
    out_csv = f"models/model_comparison_prop_{prop_id}_{feature_kind}.csv"
    try:
        df.to_csv(out_csv, index=False)
        print(f"模型比较结果已保存: {out_csv}")
    except Exception as e:
        print(f"保存模型比较CSV失败: {e}")
    return df

# def cv_mae(
#     x,
#     y,
#     model: RandomForestRegressor,
#     split_n,
#     kf_random_state: int,
# ) -> float:
#     """
#     自定义交叉验证计算MAE
    
#     Args:
#         x: 特征数据
#         y: 目标数据
#         model: 模型
#         split_n: 折数
#         kf_random_state: 随机种子
        
#     Returns:
#         平均绝对误差
#     """
#     assert len(x) == len(y)

#     # 局部变量
#     x = deepcopy(x)
#     y = deepcopy(y)
#     model = clone(model)

#     # 标准化
#     scaler = preprocessing.RobustScaler()
#     x = scaler.fit_transform(x)
#     kf = KFold(split_n, shuffle=True, random_state=kf_random_state)

#     y_test_buff, y_pred_buff = [], []
#     for train_index, test_index in kf.split(x):
#         X_train, X_test = x[train_index], x[test_index]
#         y_train, y_test = y[train_index], y[test_index]

#         model.fit(X_train, y_train)
#         y_pred = model.predict(X_test)

#         y_test_buff += y_test.tolist()
#         y_pred_buff += y_pred.tolist()

#     return mean_absolute_error(y_test_buff, y_pred_buff)


    
def train_test_model(prop_id: int, feature_kind: str, selected_indices: List[int]):
    """
    训练和测试模型
    
    Args:
        prop_id: 属性ID
        feature_count: 使用的特征数量
    """
    # 1. 加载数据
    comp_data, proc_data, prop_data, elem_feature = load_data(prop_id, feature_kind)
    
    # 2. 计算加权平均特征
    wavg_feature = np.dot(comp_data, elem_feature.T)
    # 仅保留所选特征列
    if selected_indices is None or len(selected_indices) == 0:
        raise ValueError("selected_indices 不能为空")
    wavg_feature = wavg_feature[:, selected_indices]
    
    # 3. 嵌套交叉验证评估（按用户需求，更稳健的评估；保持既有特征不变）
    # 输出重复成分统计
    try:
        _print_duplicate_composition_stats(comp_data, rounding_decimals=8)
    except Exception:
        pass

    summary = nested_cv_evaluate(
        X=wavg_feature,
        y=prop_data,
        comp_data_for_group=comp_data,  # 基于成分签名的 GroupKFold（若存在重复成分）
        outer_splits=5,
        inner_splits=3,
        use_grouping=True,
        group_rounding_decimals=8,
        deduplicate=False,
        verbose=True,
        random_state=42,
    )

    # 保存嵌套CV汇总与每折结果
    os.makedirs("models", exist_ok=True)
    summary_path_json = f"models/nestedcv_summary_prop_{prop_id}_{feature_kind}.json"
    summary_path_csv = f"models/nestedcv_folds_prop_{prop_id}_{feature_kind}.csv"
    try:
        with open(summary_path_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"保存 NestedCV JSON 失败: {e}")
    try:
        import csv as _csv
        with open(summary_path_csv, "w", newline="", encoding="utf-8") as f:
            writer = _csv.writer(f)
            writer.writerow(["fold", "r2", "mae", "params"])
            for i, (r2, mae, params) in enumerate(zip(summary.get("per_fold_r2", []), summary.get("per_fold_mae", []), summary.get("per_fold_params", [])), start=1):
                writer.writerow([i, r2, mae, params])
    except Exception as e:
        print(f"保存 NestedCV CSV 失败: {e}")

    # # 4. 训练一个最终模型（可选，用于导出特征重要性与保存）：在全量数据上调参后拟合
    # scaler = preprocessing.RobustScaler()
    # X_scaled_full = scaler.fit_transform(wavg_feature)
    # # 内层 CV 使用分组策略（若适用）
    # if summary.get("grouped", False):
    #     groups_all = _build_composition_groups(comp_data)
    #     inner_cv_full = GroupKFold(n_splits=max(2, min(3, len(np.unique(groups_all)))))
    #     final_model, final_params, _ = tune_random_forest_with_cv(
    #         X_scaled_full, prop_data, inner_cv=inner_cv_full, inner_groups=groups_all
    #     )
    # else:
    #     inner_cv_full = KFold(n_splits=3, shuffle=True, random_state=42)
    #     final_model, final_params, _ = tune_random_forest_with_cv(
    #         X_scaled_full, prop_data, inner_cv=inner_cv_full, inner_groups=None
    #     )

    # print("Best RF params (full-data tuning):", final_params)



    # 4. 训练一个最终模型（用于导出特征重要性与保存）：使用固定超参数的模型在全量数据上拟合
    scaler = preprocessing.RobustScaler()
    X_scaled_full = scaler.fit_transform(wavg_feature)
    final_model = get_random_forest()
    final_model.fit(X_scaled_full, prop_data)
    final_params = final_model.get_params()

    print("RF params (fixed):", final_params)

    # 5. 特征重要性（基于最终模型）
    feature_importances = final_model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': [f'Feature_{i}' for i in range(wavg_feature.shape[1])],
        'Importance': feature_importances
    }).sort_values('Importance', ascending=False)

    print("\n特征重要性:")
    print(importance_df.head(10))

    return final_model, scaler, importance_df

if __name__ == "__main__":
    # 对三个方法分别在 prop 0/1 上使用GA选出的最佳特征训练并保存模型
    methods = ["bt", "extend_ep", "tradition_ep"]
    labels_map = ["YTS", "UTS"]
    os.makedirs("models", exist_ok=True)

    for prop_id in [0]:
        for method in methods:
            print(f"\n=== 训练与比较: {method}, 属性: {prop_id} ({labels_map[prop_id]}) ===")
            try:
                sel_idx = _get_best_indices_from_evolution(method, prop_id, f_n=6)
            except Exception as e:
                print(f"跳过 {method}-prop{prop_id}: 未能获取特征索引 -> {e}")
                continue

            # 进行模型比较并保存汇总CSV（便于 Origin 作图）
            _ = compare_models(
                prop_id=prop_id,
                feature_kind=method,
                selected_indices=sel_idx,
                outer_splits=5,
                inner_splits=3,
                use_grouping=True,
                group_rounding_decimals=8,
                deduplicate=False,
                verbose=False,
                random_state=42,
            )

            # 同时保留旧的RF训练与特征重要性导出
            model, scaler, importance_df = train_test_model(prop_id, method, selected_indices=sel_idx)

            # 保存模型和结果
            joblib.dump(model, f"models/rf_model_prop_{prop_id}_{method}.pkl")
            joblib.dump(scaler, f"models/rf_scaler_prop_{prop_id}_{method}.pkl")
            importance_df.to_csv(
                f"models/feature_importance_prop_{prop_id}_{method}.csv",
                index=False,
            )
            # 保存所选特征索引
            with open(f"models/selected_features_prop_{prop_id}_{method}.json", "w", encoding="utf-8") as f:
                json.dump({"prop_id": prop_id, "method": method, "selected_indices": sel_idx}, f, ensure_ascii=False, indent=2)

    print("\n全部任务完成，模型与结果已保存到 'models/' 目录")
