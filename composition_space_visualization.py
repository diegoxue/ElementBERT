import os
import argparse
from typing import Tuple, List, Union, Optional
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# 与 ML_future_analysis.py 中保持一致
COMP_LABELS: List[str] = [
    "Ti", "Al", "Zr", "Mo", "V", "Ta", "Nb", "Cr", "Mn", "Fe", "Sn",
]


def _load_elem_feature(feature_kind: str) -> np.ndarray:
    """
    读取元素特征矩阵（行=特征, 列=元素），与训练脚本保持一致：
    - bt: data/ElementBERT_ele_vec.xlsx
    - extend_ep: data/extend_ep_19.xlsx
    - tradition_ep: data/Element_features-19.xlsx
    """
    if feature_kind == "bt":
        path = "data/ElementBERT_ele_vec.xlsx"
    elif feature_kind == "extend_ep":
        path = "data/extend_ep_19.xlsx"
    elif feature_kind == "tradition_ep":
        path = "data/Element_features-19.xlsx"
    else:
        raise ValueError("feature_kind 必须为 'bt' / 'extend_ep' / 'tradition_ep'")
    elem_feature = pd.read_excel(path)
    elem_feature = elem_feature[COMP_LABELS].to_numpy()
    return elem_feature


def _load_model_assets(prop_id: int, feature_kind: str):
    model_path = f"models_fixed_rf/rf_model_prop_{prop_id}_{feature_kind}.pkl"
    scaler_path = f"models_fixed_rf/rf_scaler_prop_{prop_id}_{feature_kind}.pkl"
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError("未找到已训练模型或scaler，请先运行 ML_future_analysis.py 训练并保存模型")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


def _load_selected_indices(prop_id: int, feature_kind: str) -> List[int]:
    """
    读取与模型对应的特征索引（GA选出的最佳组合）。
    文件由 ML_future_analysis.py 保存为 models_fixed_rf/selected_features_prop_{prop_id}_{feature_kind}.json
    """
    path = f"models_fixed_rf/selected_features_prop_{prop_id}_{feature_kind}.json"
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到所选特征索引: {path}")
    import json
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return list(map(int, data.get("selected_indices", [])))


# def _baseline_composition(prop_id: int) -> np.ndarray:
#     """
#     从训练数据中估计一个基线成分（均值）。
#     为避免依赖 ML_future_analysis 的数据过滤逻辑，这里直接读取 CSV 并取 COMP_LABELS 均值。
#     """
#     labels = ["YTS", "UTS"]
#     csv_path = f"data/{labels[prop_id]}.csv"
#     df = pd.read_csv(csv_path)
#     comp = df[COMP_LABELS].to_numpy(dtype=float)
#     comp = comp / comp.sum(axis=1, keepdims=True)
#     return comp.mean(axis=0)


def _wavg_features(comp: np.ndarray, elem_feature: np.ndarray) -> np.ndarray:
    comp = np.asarray(comp, dtype=float)
    comp = comp / comp.sum(axis=-1, keepdims=True)
    return np.dot(comp, elem_feature.T)


def _normalize_elem_name(name: str) -> str:
    name = str(name).strip().replace('%', '')
    if not name:
        return name
    # 标准元素符号格式：首字母大写，其余小写
    return name[0].upper() + name[1:].lower()


def _load_external_compositions(filepath: str, labels: List[str]) -> np.ndarray:
    """
    读取外部成分表（Excel/CSV），提取与 labels 对齐的列，不存在的元素填 0；
    将提取后的部分归一化为和为 1 的原子分数。
    支持 .xlsx/.xls/.csv。
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"未找到外部成分文件: {filepath}")
    if filepath.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(filepath)
    elif filepath.lower().endswith(".csv"):
        df = pd.read_csv(filepath)
    else:
        raise ValueError("仅支持 .xlsx/.xls/.csv 文件")

    # 规范化外部文件的列名（去空格/百分号，元素名大小写统一）
    norm_cols = [_normalize_elem_name(c) for c in df.columns]
    df.columns = norm_cols

    # 仅保留 labels 中的列，不存在的补 0
    data = pd.DataFrame({col: (df[col] if col in df.columns else 0.0) for col in labels})
    arr = data.to_numpy(dtype=float)
    # 归一化到和为 1（若总和为 0 则跳过该行）
    row_sums = arr.sum(axis=1, keepdims=True)
    mask = row_sums.squeeze() > 0
    if not np.all(mask):
        # 对于全 0 的行，保持为 0（后续可能会被过滤）
        pass
    arr[mask] = arr[mask] / row_sums[mask]
    return arr[mask]


# def validate_external_against_discrete_space(ext_file: str,
#                                              constraints: dict,
#                                              step: float,
#                                              labels: List[str],
#                                              report_csv: Optional[str] = None) -> Optional[pd.DataFrame]:
#     """
#     校验外部成分是否位于离散约束空间内：
#     - 每元素在[min,max]内
#     - 每元素为step的整数倍，且单位和为1/step
#     - 每行总和≈1
#     打印汇总，并（可选）保存CSV报告。
#     返回DataFrame（若外部数据为空则返回None）。
#     """
#     try:
#         arr = _load_external_compositions(ext_file, labels)
#     except Exception as e:
#         print(f"[Validate] 外部数据读取失败: {e}")
#         return None
#     if arr is None or len(arr) == 0:
#         print("[Validate] 无有效外部成分可供校验（列未匹配或行为0和）")
#         return None

#     mins, maxs = _build_bounds_from_constraints(constraints, labels)
#     tol = 1e-8
#     units = int(round(1.0 / step))

#     records = []
#     for i, v in enumerate(arr):
#         sum_ok = bool(np.isclose(v.sum(), 1.0, atol=1e-6))
#         within = (v >= mins - tol) & (v <= maxs + tol)
#         within_ok = bool(np.all(within))

#         # 网格性：每个分量是 step 的整数倍，且单位和=units
#         ratios = v / step
#         ratios_rounded = np.round(ratios)
#         on_grid_each = np.isclose(ratios, ratios_rounded, atol=1e-6)
#         on_grid_ok = bool(np.all(on_grid_each) and int(ratios_rounded.sum()) == units)

#         offgrid_elems = [labels[j] for j, ok in enumerate(on_grid_each) if not ok]
#         viol_elems = [labels[j] for j, ok in enumerate(within) if not ok]

#         records.append({
#             "row": i,
#             "sum_ok": sum_ok,
#             "within_bounds": within_ok,
#             "on_grid": on_grid_ok,
#             "sum": float(v.sum()),
#             "sum_units": int(ratios_rounded.sum()),
#             "units_expected": units,
#             "violations": ",".join(viol_elems) if viol_elems else "",
#             "offgrid_elems": ",".join(offgrid_elems) if offgrid_elems else "",
#         })

#     df = pd.DataFrame.from_records(records)
#     total = len(df)
#     print(f"[Validate] 外部样本数: {total}")
#     print(f"[Validate] 在边界内: {df['within_bounds'].sum()}/{total}")
#     print(f"[Validate] 满足离散网格(step={step})且单位和=1/step: {df['on_grid'].sum()}/{total}")
#     print(f"[Validate] 总和≈1: {df['sum_ok'].sum()}/{total}")
#     if report_csv:
#         try:
#             os.makedirs(os.path.dirname(report_csv) or ".", exist_ok=True)
#             df.to_csv(report_csv, index=False)
#             print(f"[Validate] 详细报告已保存: {report_csv}")
#         except Exception as e:
#             print(f"[Validate] 保存报告失败: {e}")
#     return df


def _build_bounds_from_constraints(constraints: dict, labels: List[str], default_min: float = 0.0, default_max: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    将 {element: (min, max)} 约束映射到与 labels 对齐的 (mins, maxs)。
    未提供约束的元素使用默认 [0,1]。
    会对 min/max 做裁剪到 [0,1] 且确保 min<=max。
    """
    mins, maxs = [], []
    for el in labels:
        mn, mx = constraints.get(el, (default_min, default_max))
        mn = float(np.clip(mn, 0.0, 1.0))
        mx = float(np.clip(mx, 0.0, 1.0))
        if mx < mn:
            mn, mx = mx, mn
        mins.append(mn)
        maxs.append(mx)
    return np.array(mins, dtype=float), np.array(maxs, dtype=float)


# def _sample_bounded_discrete_compositions(n_samples: int,
#                                           step: float,
#                                           mins: np.ndarray,
#                                           maxs: np.ndarray,
#                                           random_state: int = 42) -> np.ndarray:
#     """
#     在给定步长 step (如 0.01) 的离散网格上，随机采样满足分量边界与和为1的成分向量。
#     使用逐维有界随机分配算法，保证可行性。
#     返回形状 (n_samples, n_dim) 的数组。
#     """
#     assert mins.shape == maxs.shape
#     n_dim = len(mins)
#     units = int(round(1.0 / step))  # 0.01 -> 100 单位
#     rng = np.random.default_rng(random_state)

#     min_units = np.ceil(mins * units).astype(int)
#     max_units = np.floor(maxs * units).astype(int)

#     # 若不可行，做最小修正
#     total_min = int(min_units.sum())
#     total_max = int(max_units.sum())
#     if total_min > units:
#         # 归一地缩小部分最小值
#         overflow = total_min - units
#         order = np.argsort(-(min_units - 0))  # 优先减少大的
#         for i in order:
#             reducible = min(min_units[i] - 0, overflow)
#             min_units[i] -= reducible
#             overflow -= reducible
#             if overflow <= 0:
#                 break
#     elif total_max < units:
#         # 扩大部分最大值
#         short = units - total_max
#         order = np.argsort(-(units - max_units))
#         for i in order:
#             inc = min(units - max_units[i], short)
#             max_units[i] += inc
#             short -= inc
#             if short <= 0:
#                 break

#     samples = []
#     tries = 0
#     while len(samples) < n_samples and tries < n_samples * 50:
#         tries += 1
#         remaining = units
#         x = np.zeros(n_dim, dtype=int)
#         # 动态区间
#         cur_min = min_units.copy()
#         cur_max = max_units.copy()
#         feasible = True
#         for d in range(n_dim - 1):
#             min_rest = cur_min[d:].copy()
#             max_rest = cur_max[d:].copy()
#             sum_min_rest_next = int(min_rest[1:].sum())
#             sum_max_rest_next = int(max_rest[1:].sum())
#             lo = max(cur_min[d], remaining - sum_max_rest_next)
#             hi = min(cur_max[d], remaining - sum_min_rest_next)
#             if lo > hi:
#                 feasible = False
#                 break
#             val = rng.integers(lo, hi + 1)
#             x[d] = val
#             remaining -= val
#         if not feasible:
#             continue
#         x[-1] = remaining
#         if x[-1] < cur_min[-1] or x[-1] > cur_max[-1]:
#             continue
#         samples.append(x / units)

#     if len(samples) == 0:
#         raise RuntimeError("未能在给定约束与步长下采样到可行成分，请检查边界是否过紧或不一致")
#     return np.vstack(samples)


def _project_to_capped_simplex(x: np.ndarray,
                               mins: np.ndarray,
                               maxs: np.ndarray,
                               tol: float = 1e-12,
                               max_iter: int = 100) -> np.ndarray:
    """
    将向量 x 投影到带上下界的单位和单纯形：
    找 lambda 使得 sum(clip(x - lambda, mins, maxs)) = 1。
    使用二分搜索，范围在 [min(x - maxs), max(x - mins)]。
    """
    assert x.shape == mins.shape == maxs.shape
    low = float(np.min(x - maxs))
    high = float(np.max(x - mins))
    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        clipped = np.clip(x - mid, mins, maxs)
        s = clipped.sum()
        if abs(s - 1.0) <= tol:
            return clipped
        if s > 1.0:
            low = mid
        else:
            high = mid
    return np.clip(x - 0.5 * (low + high), mins, maxs)


def _local_grid_walk_samples_around(ext_comps: np.ndarray,
                                    mins: np.ndarray,
                                    maxs: np.ndarray,
                                    step: float,
                                    per_point: int,
                                    radius_units: int,
                                    random_state: int = 42) -> np.ndarray:
    """
    在每个外部成分的附近进行离散网格随机走动采样：
    - 将起点转换为单位计数（按 step）
    - 随机选择 donor/receiver 元素转移单位，步数<=radius_units
    - 保证每步不违反 min/max 单位界限
    返回合并后的样本（去重）。
    """
    rng = np.random.default_rng(random_state)
    units = int(round(1.0 / step))
    min_units = np.ceil(mins * units).astype(int)
    max_units = np.floor(maxs * units).astype(int)

    all_samples = []
    for v in ext_comps:
        start_units = np.round(v * units).astype(int)
        # 项目到可行单位区间与和=units
        start_units = np.clip(start_units, min_units, max_units)
        diff = units - int(start_units.sum())
        if diff != 0:
            # 调整到和=units
            if diff > 0:
                # 增加 diff 个单位
                for _ in range(diff):
                    candidates = np.where(start_units < max_units)[0]
                    if len(candidates) == 0:
                        break
                    j = int(rng.choice(candidates))
                    start_units[j] += 1
            else:
                for _ in range(-diff):
                    candidates = np.where(start_units > min_units)[0]
                    if len(candidates) == 0:
                        break
                    j = int(rng.choice(candidates))
                    start_units[j] -= 1

        for _ in range(per_point):
            x = start_units.copy()
            steps = int(rng.integers(1, radius_units + 1))
            for _s in range(steps):
                donors = np.where(x > min_units)[0]
                receivers = np.where(x < max_units)[0]
                if len(donors) == 0 or len(receivers) == 0:
                    break
                d = int(rng.choice(donors))
                r = int(rng.choice(receivers))
                if d == r:
                    continue
                x[d] -= 1
                x[r] += 1
            all_samples.append(x / units)

    if not all_samples:
        return np.empty((0, mins.shape[0]))
    arr = np.vstack(all_samples)
    # 去重
    arr_unique = np.unique((np.round(arr / step).astype(int)), axis=0) * step
    # 投影防止数值误差
    proj = np.vstack([_project_to_capped_simplex(a, mins, maxs) for a in arr_unique])
    return proj


def local_discrete_scatter(prop_id: int,
                           feature_kind: str,
                           constraints: dict,
                           step: float = 0.01,
                           per_point: int = 1000,
                           radius_units: int = 3,
                           embed: str = "pca",
                           out_png: Optional[str] = None,
                           random_state: int = 42,
                           ext_file: Optional[str] = "./data/Ti-newdata_atomic.xlsx",
                           ext_marker_size: float = 64.0,
                           ext_color: str = 'red',
                           ext_edgecolor: str = 'black',
                           ext_label: str = 'New data',
                           min_pred: Optional[float] = None) -> None:
    """
    以外部候选成分为中心，在其周围进行离散网格随机走动采样，并与外部点一起可视化（PCA/t-SNE）。
    """
    mins, maxs = _build_bounds_from_constraints(constraints, COMP_LABELS)
    if ext_file is None:
        print("[Local] 未提供外部文件 ext_file")
        return
    try:
        ext_comps = _load_external_compositions(ext_file, COMP_LABELS)
    except Exception as e:
        print(f"[Local] 读取外部成分失败: {e}")
        return
    if ext_comps is None or len(ext_comps) == 0:
        print("[Local] 无有效外部成分用于局部采样")
        return

    # 生成局部样本
    local_samples = _local_grid_walk_samples_around(
        ext_comps=ext_comps, mins=mins, maxs=maxs, step=step,
        per_point=per_point, radius_units=radius_units, random_state=random_state,
    )
    if local_samples.shape[0] == 0:
        print("[Local] 未生成局部样本，请检查约束/步长/半径")
        return

    # 预测与嵌入
    model, scaler = _load_model_assets(prop_id, feature_kind)
    elem_feature = _load_elem_feature(feature_kind)
    sel_idx = _load_selected_indices(prop_id, feature_kind)
    feats = _wavg_features(local_samples, elem_feature)
    feats = feats[:, sel_idx]
    feats_scaled = scaler.transform(feats)
    preds = model.predict(feats_scaled).astype(float)
    # 外部点预测用于过滤
    try:
        ext_feats = _wavg_features(ext_comps, elem_feature)
        ext_feats = ext_feats[:, sel_idx]
        ext_preds = model.predict(scaler.transform(ext_feats)).astype(float)
    except Exception:
        ext_preds = None

    if embed.lower() == "pca":
        pca = PCA(n_components=2, random_state=random_state)
        emb = pca.fit_transform(local_samples)
        ext_emb = pca.transform(ext_comps)
        xlabel, ylabel, title_embed = "PCA-1", "PCA-2", "PCA"
    else:
        joint = np.vstack([local_samples, ext_comps])
        tsne = TSNE(n_components=2, init='pca', perplexity=15.0,
                    n_iter=2000, learning_rate='auto', random_state=random_state, verbose=0)
        joint_emb = tsne.fit_transform(joint)
        emb = joint_emb[: local_samples.shape[0]]
        ext_emb = joint_emb[local_samples.shape[0]:]
        xlabel, ylabel, title_embed = "t-SNE-1", "t-SNE-2", "t-SNE"

    plt.figure(figsize=(7, 6))
    mask = np.ones_like(preds, dtype=bool)
    if min_pred is not None:
        mask = preds >= float(min_pred)

    # 确保输出目录：默认保存到 models_fixed_rf 下
    if out_png is None:
        out_png = f"models_fixed_rf/composition_local_{title_embed.lower()}_prop_{prop_id}_{feature_kind}.png"
    out_dir = os.path.dirname(out_png) or "."
    os.makedirs(out_dir, exist_ok=True)

    # 绘图
    sc = plt.scatter(emb[mask, 0], emb[mask, 1], c=preds[mask], s=10, cmap="viridis")
    plt.colorbar(sc, label="Predicted Property")
    # 始终显示外部点（不受 min_pred 过滤）
    if ext_emb is not None:
        plt.scatter(ext_emb[:, 0], ext_emb[:, 1], c=ext_color, s=ext_marker_size,
                    edgecolors=ext_edgecolor, marker='^', linewidths=0.8, label=ext_label, zorder=5)
    plt.legend(loc='best')
    plt.xlabel(f"{xlabel} (composition)")
    plt.ylabel(f"{ylabel} (composition)")
    plt.title(f"Local {title_embed} scatter around external | prop_id={prop_id}, kind={feature_kind}, step={step}")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

    # 保存用于复现的CSV数据点（仅保存实际绘制的点）
    csv_path = os.path.splitext(out_png)[0] + ".csv"
    rows = []
    # 本地采样点（受 min_pred 过滤）
    for x, y, p in zip(emb[mask, 0], emb[mask, 1], preds[mask]):
        rows.append({"set": "local", "x": float(x), "y": float(y), "pred": float(p)})
    # 外部点（不受 min_pred 过滤）
    if ext_emb is not None:
        if ext_preds is None:
            ext_preds_arr = [np.nan] * ext_emb.shape[0]
        else:
            ext_preds_arr = ext_preds
        for x, y, p in zip(ext_emb[:, 0], ext_emb[:, 1], ext_preds_arr):
            p_val = float(p) if p is not None and not (isinstance(p, float) and np.isnan(p)) else np.nan
            rows.append({"set": "external", "x": float(x), "y": float(y), "pred": p_val})
    try:
        pd.DataFrame(rows, columns=["set", "x", "y", "pred"]).to_csv(csv_path, index=False)
    except Exception as e:
        print(f"保存CSV失败: {e}")

    print(f"已保存局部散点图: {out_png}")
    print(f"已保存数据点CSV: {csv_path}")




if __name__ == "__main__":
    # 仅保留局部离散随机走动采样的可视化（使用已训练模型与所选特征）
    constraints = {
        "Ti": (0.70, 0.95), "Al": (0.00, 0.15), "Zr": (0.00, 0.15), "Mo": (0.00, 0.10),
        "V": (0.00, 0.15), "Ta": (0.00, 0.05), "Nb": (0.00, 0.15), "Cr": (0.00, 0.10),
        "Mn": (0.00, 0.05), "Fe": (0.00, 0.05), "Sn": (0.00, 0.15),
    }

    local_discrete_scatter(
        prop_id=0,               # 0=YTS
        feature_kind="tradition_ep",       # 与训练一致
        constraints=constraints, # 上下界
        step=0.01,               # 网格步长
        per_point=1000,          # 每个外部点邻域采样数
        radius_units=3,          # 最多走动 3 个单位(=0.03)
        embed="tsne",            # 或 "pca"
        out_png=None,            # 默认文件名保存
        ext_file="data/Ti-newdata_atomic.xlsx",
        ext_marker_size=64.0,
        ext_color="red",
        ext_edgecolor="black",
        ext_label="New data",
        min_pred=1100,
    )

    local_discrete_scatter(
        prop_id=0,               # 0=YTS
        feature_kind="bt",       # 与训练一致
        constraints=constraints, # 上下界
        step=0.01,               # 网格步长
        per_point=1000,          # 每个外部点邻域采样数
        radius_units=3,          # 最多走动 3 个单位(=0.03)
        embed="tsne",            # 或 "pca"
        out_png=None,            # 默认文件名保存
        ext_file="data/Ti-newdata_atomic.xlsx",
        ext_marker_size=64.0,   
        ext_color="red",
        ext_edgecolor="black",
        ext_label="New data",
        min_pred=1100,
    )

    local_discrete_scatter(
        prop_id=0,               # 0=YTS
        feature_kind="extend_ep",       # 与训练一致
        constraints=constraints, # 上下界
        step=0.01,               # 网格步长
        per_point=1000,          # 每个外部点邻域采样数
        radius_units=3,          # 最多走动 3 个单位(=0.03)
        embed="tsne",            # 或 "pca"
        out_png=None,            # 默认文件名保存
        ext_file="data/Ti-newdata_atomic.xlsx",
        ext_marker_size=64.0,
        ext_color="red",
        ext_edgecolor="black",
        ext_label="New data",
        min_pred=1100,
    ) 







