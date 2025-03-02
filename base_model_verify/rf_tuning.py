import joblib
import numpy as np
from bayes_opt import BayesianOptimization
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler

from verify_feature_eff import load_data


def cv_score(
    x: np.ndarray,
    y: np.ndarray,
    model: RandomForestRegressor,
    n_splits: int = 10,
    random_state: int = 42,
) -> float:
    """计算交叉验证的R²分数"""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scaler = RobustScaler()
    x = scaler.fit_transform(x)

    def single_fold(train_idx, test_idx):
        X_train, X_test = x[train_idx], x[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model_copy = RandomForestRegressor(**model.get_params())
        model_copy.fit(X_train, y_train)
        y_pred = model_copy.predict(X_test)

        return y_test, y_pred

    results = Parallel(n_jobs=-1)(
        delayed(single_fold)(train_idx, test_idx) for train_idx, test_idx in kf.split(x)
    )

    y_test_all, y_pred_all = [], []
    for y_test, y_pred in results:
        y_test_all.extend(y_test)
        y_pred_all.extend(y_pred)

    return r2_score(y_test_all, y_pred_all)


def rf_optimize(x: np.ndarray, y: np.ndarray, random_state: int = 42):
    """使用贝叶斯优化寻找最佳超参数"""

    def objective(
        n_estimators,
        max_depth,
        min_samples_split,
        min_samples_leaf,
        max_features,
        max_samples,
    ):
        model = RandomForestRegressor(
            n_estimators=int(n_estimators),  
            max_depth=int(max_depth),  
            min_samples_split=int(min_samples_split),  
            min_samples_leaf=int(min_samples_leaf),  
            max_features=max_features,  
            max_samples=max_samples,  
            n_jobs=-1,  
            random_state=random_state,
        )
        return cv_score(x, y, model)

    # 参数搜索范围
    pbounds = {
        "n_estimators": (50, 300),  
        "max_depth": (3, 20),  
        "min_samples_split": (2, 10),  
        "min_samples_leaf": (1, 5),  
        "max_features": (0.3, 1.0),  
        "max_samples": (0.5, 1.0),  
    }

    optimizer = BayesianOptimization(
        f=objective, pbounds=pbounds, random_state=random_state, verbose=2
    )

    optimizer.maximize(init_points=20, n_iter=100)
    return optimizer.max


if __name__ == "__main__":
    # 加载数据
    prop_id = 0
    comp_data, proc_data, prop_data, elem_feature_ep, elem_feature_bt = load_data(
        prop_id
    )

    wavg_feature_ep = np.dot(comp_data, elem_feature_ep.T)

    np.random.seed(42)
    selected_index = np.random.choice(len(comp_data), 68, replace=False)
    sel_wavg_feature_ep = wavg_feature_ep[selected_index]
    sel_prop_data = prop_data[selected_index]

    print("开始RandomForest超参数贝叶斯优化...")
    best_result = rf_optimize(sel_wavg_feature_ep, sel_prop_data)

    print("\n最佳参数:")
    print(f"树的数量: {int(best_result['params']['n_estimators'])}")
    print(f"最大深度: {int(best_result['params']['max_depth'])}")
    print(f"最小分裂样本数: {int(best_result['params']['min_samples_split'])}")
    print(f"最小叶节点样本数: {int(best_result['params']['min_samples_leaf'])}")
    print(f"特征采样比例: {best_result['params']['max_features']:.3f}")
    print(f"样本采样比例: {best_result['params']['max_samples']:.3f}")
    print(f"最佳R²分数: {best_result['target']:.4f}")

    # 保存结果
    results = {
        "params": {
            k: int(v)
            if k
            in ["n_estimators", "max_depth", "min_samples_split", "min_samples_leaf"]
            else float(v)
            for k, v in best_result["params"].items()
        },
        "score": best_result["target"],
    }
    joblib.dump(results, f"rf_tuning_results_prop{prop_id}.pkl")
