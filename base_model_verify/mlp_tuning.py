import joblib
import numpy as np
from bayes_opt import BayesianOptimization
from joblib import Parallel, delayed
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import RobustScaler

from verify_feature_eff import load_data


def cv_score(
    x: np.ndarray,
    y: np.ndarray,
    model: MLPRegressor,
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

        model_copy = MLPRegressor(**model.get_params())
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


def mlp_optimize(x: np.ndarray, y: np.ndarray, random_state: int = 42):
    """使用贝叶斯优化寻找最佳超参数"""

    def objective(
        layer1, layer2, layer3, layer4, layer5, learning_rate_init, alpha, batch_size
    ):
        # 移除2048选项
        nodes_options = np.array([1024, 512, 256, 128, 64])

        layer_sizes = []
        for layer_val in [layer1, layer2, layer3, layer4, layer5]:
            idx = int(layer_val * (len(nodes_options) - 1))
            layer_sizes.append(nodes_options[idx])

        model = MLPRegressor(
            hidden_layer_sizes=tuple(layer_sizes),
            activation="relu",
            solver="adam",
            learning_rate_init=learning_rate_init,
            alpha=alpha,
            batch_size=int(batch_size),
            max_iter=3000,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=50,
            random_state=random_state,
            learning_rate="adaptive",
        )
        return cv_score(x, y, model)

    pbounds = {
        "layer1": (0, 1),
        "layer2": (0, 1),
        "layer3": (0, 1),
        "layer4": (0, 1),
        "layer5": (0, 1),
        "learning_rate_init": (0.0001, 0.01),
        "alpha": (0.00001, 0.001),
        "batch_size": (8, 32),
    }

    optimizer = BayesianOptimization(
        f=objective, pbounds=pbounds, random_state=random_state, verbose=2
    )

    optimizer.maximize(init_points=20, n_iter=150)
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

    print("开始MLP超参数贝叶斯优化...")
    print("可选的隐藏层节点数:", [1024, 512, 256, 128, 64])
    best_result = mlp_optimize(sel_wavg_feature_ep, sel_prop_data)

    # 将最佳参数转换为实际的节点数
    nodes_options = np.array([1024, 512, 256, 128, 64])
    layer_params = [best_result["params"][f"layer{i}"] for i in range(1, 6)]
    actual_layers = [
        nodes_options[int(val * (len(nodes_options) - 1))] for val in layer_params
    ]

    print("\n最佳参数:")
    print(f"隐藏层节点数: {actual_layers}")
    print(f"学习率: {best_result['params']['learning_rate_init']}")
    print(f"正则化参数: {best_result['params']['alpha']}")
    print(f"批量大小: {int(best_result['params']['batch_size'])}")
    print(f"最佳R²分数: {best_result['target']:.4f}")

    results = {
        "params": best_result["params"],
        "actual_layers": actual_layers,
        "score": best_result["target"],
    }
    joblib.dump(results, f"mlp_tuning_results_prop{prop_id}.pkl")
