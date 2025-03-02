import joblib
import numpy as np
from bayes_opt import BayesianOptimization
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVR

from verify_feature_eff import load_data


def cv_score(
    x: np.ndarray, y: np.ndarray, model: SVR, n_splits: int = 10, random_state: int = 42
) -> float:
    """计算交叉验证的R²分数"""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scaler = RobustScaler()
    x = scaler.fit_transform(x)

    y_test_all, y_pred_all = [], []
    for train_idx, test_idx in kf.split(x):
        X_train, X_test = x[train_idx], x[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        y_test_all.extend(y_test)
        y_pred_all.extend(y_pred)

    return r2_score(y_test_all, y_pred_all)


def svr_optimize(x: np.ndarray, y: np.ndarray, random_state: int = 42):
    """使用贝叶斯优化寻找最佳超参数"""

    def objective(C, epsilon, gamma):
        model = SVR(
            kernel="rbf",  
            C=C,  
            epsilon=epsilon,  
            gamma=gamma,  
            cache_size=1000,  
        )
        return cv_score(x, y, model)

    # 设置参数搜索范围
    pbounds = {
        "C": (0.1, 1000.0),  
        "epsilon": (0.001, 1.0),  
        "gamma": (0.001, 10.0),  
    }

    optimizer = BayesianOptimization(
        f=objective,
        pbounds=pbounds,
        random_state=random_state,
        verbose=2,  
    )

    # 优化设置
    optimizer.maximize(
        init_points=15,  
        n_iter=100,  
    )
    return optimizer.max


if __name__ == "__main__":
    # 加载数据
    prop_id = 0  
    comp_data, proc_data, prop_data, elem_feature_ep, elem_feature_bt = load_data(
        prop_id
    )

    # 计算加权平均特征
    wavg_feature_ep = np.dot(comp_data, elem_feature_ep.T)

    # 随机选择部分数据进行优化
    np.random.seed(42)
    selected_index = np.random.choice(len(comp_data), 68, replace=False)
    sel_wavg_feature_ep = wavg_feature_ep[selected_index]
    sel_prop_data = prop_data[selected_index]

    # 运行贝叶斯优化
    print("开始SVR超参数贝叶斯优化...")
    best_result = svr_optimize(sel_wavg_feature_ep, sel_prop_data)
    print("\n最佳参数:")
    print(best_result["params"])
    print(f"最佳R²分数: {best_result['target']:.4f}")

    # 保存结果
    joblib.dump(best_result, f"svr_tuning_results_prop{prop_id}.pkl")
