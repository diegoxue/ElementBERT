import glob
import os
from collections import Counter

import joblib
import numpy as np


def load_final_feature_selection(prop_id, f_n, traj_id=None):
    """
    读取特征选择信息
    Args:
        prop_id: 属性ID
        f_n: 特征数量
        traj_id: 轨迹ID（如果指定）
    Returns:
        如果指定traj_id，返回单个结果
        如果不指定traj_id，返回所有轨迹的结果列表
    """
    # 构建搜索模式
    if traj_id is not None:
        pattern = f"feature_selection/fs-{prop_id}-{f_n}-{traj_id}-*.pkl"
    else:
        pattern = f"feature_selection/fs-{prop_id}-{f_n}-*.pkl"

    # 获取所有匹配的文件
    files = glob.glob(pattern)

    results = []
    for file in files:
        data = joblib.load(file)
        results.append(data)

    # 如果指定了traj_id，只返回第一个结果
    if traj_id is not None and results:
        return results[0]

    return results


def load_best_features(prop_id, f_n, traj_id=None):
    """
    读取最佳特征选择结果
    Args:
        prop_id: 属性ID
        f_n: 特征数量
        traj_id: 轨迹ID（如果指定）
    Returns:
        最佳特征的索引列表
    """
    # 构建文件路径
    if traj_id is not None:
        pattern = f"feature_selection/fs-{prop_id}-{f_n}-{traj_id}-*.pkl"
    else:
        pattern = f"feature_selection/fs-{prop_id}-{f_n}-*.pkl"

    files = glob.glob(pattern)

    if not files:
        raise FileNotFoundError(f"没有找到匹配的文件: {pattern}")

    # 读取文件
    results = []
    for file in files:
        data = joblib.load(file)
        # 打印文件内容的键，以便查看具体结构
        print(f"文件 {file} 包含的键: {data.keys()}")
        results.append(data)

    return results


def get_best_features(prop_id, f_n, traj_id):
    """
    读取指定条件下的最佳特征
    Args:
        prop_id: 属性ID
        f_n: 特征数量
        traj_id: 轨迹ID
    Returns:
        feature_indices: 最佳特征的索引
        fitness: 对应的适应度值
    """
    file_pattern = f"feature_selection/fs-{prop_id}-{f_n}-{traj_id}-*.pkl"
    files = glob.glob(file_pattern)

    if not files:
        raise FileNotFoundError(f"没有找到匹配的文件: {file_pattern}")

    data = joblib.load(files[0])
    best_features = data["best_features"]

    return best_features["indices"], best_features["fitness"]


def get_all_trajectories_best_features(prop_id, f_n):
    """
    读取所有轨迹的最佳特征索引
    Args:
        prop_id: 属性ID
        f_n: 特征数量
    Returns:
        dict: 包含每个轨迹的最佳特征索引和适应度值
    """
    # 获取所有匹配的文件
    file_pattern = f"feature_selection/fs-{prop_id}-{f_n}-*-*.pkl"
    files = glob.glob(file_pattern)

    results = {}
    for file in files:
        # 从文件名中提取轨迹ID
        traj_id = int(file.split("-")[3])

        # 读取数据
        data = joblib.load(file)
        best_features = data["best_features"]

        results[traj_id] = {
            "indices": best_features["indices"],
            "fitness": best_features["fitness"],
        }

    return results


def get_best_trajectory_features(prop_id, f_n):
    """
    获取所有轨迹中适应度最高的特征组合
    Args:
        prop_id: 属性ID
        f_n: 特征数量
    Returns:
        best_indices: 最佳特征索引
        best_fitness: 最佳适应度值
        best_traj: 对应的轨迹ID
    """
    file_pattern = f"feature_selection/fs-{prop_id}-{f_n}-*-*.pkl"
    files = glob.glob(file_pattern)

    best_fitness = -np.inf
    best_indices = None
    best_traj = None

    for file in files:
        traj_id = int(file.split("-")[3])
        data = joblib.load(file)
        current_fitness = data["best_features"]["fitness"]

        if current_fitness > best_fitness:
            best_fitness = current_fitness
            best_indices = data["best_features"]["indices"]
            best_traj = traj_id

    return best_indices, best_fitness, best_traj


def get_all_best_features(prop_id, f_n):
    """
    获取所有轨迹中最佳特征索引的集合
    Args:
        prop_id: 属性ID
        f_n: 特征数量
    Returns:
        unique_best_features: 所有轨迹最佳特征的集合（去重后的）
    """
    file_pattern = f"feature_selection/fs-{prop_id}-{f_n}-*-*.pkl"
    files = glob.glob(file_pattern)

    # 收集所有轨迹的最佳特征
    best_features = []
    for file in files:
        data = joblib.load(file)
        best_features.extend(data["best_features"]["indices"])

    # 去重并排序
    unique_best_features = sorted(set(best_features))

    return unique_best_features


def analyze_feature_frequency(prop_id, f_n, output_file="feature_frequency.csv"):
    """
    分析特征出现频率并保存到CSV文件
    Args:
        prop_id: 属性ID
        f_n: 特征数量
        output_file: 输出CSV文件名
    """
    import pandas as pd
    from collections import Counter

    # 获取所有轨迹的特征选择结果
    results = get_all_trajectories_best_features(prop_id, f_n)

    # 收集所有特征索引
    all_features = []
    for traj_id in results:
        all_features.extend(results[traj_id]["indices"])

    # 统计频率
    feature_counts = Counter(all_features)

    # 转换为DataFrame并排序
    df = pd.DataFrame.from_dict(feature_counts, orient="index", columns=["出现次数"])
    df.index.name = "特征索引"
    df = df.sort_values("出现次数", ascending=False)

    # 保存到CSV
    df.to_csv(output_file)
    print(f"特征频率统计已保存到: {output_file}")
    return df


# 使用示例：
df = analyze_feature_frequency(prop_id=0, f_n=4)
print("\n前10个最常出现的特征：")
print(df.head(10))











# # 使用示例：
# best_features = get_all_best_features(prop_id=0, f_n=4)
# print("\n所有轨迹最佳特征索引的集合:")
# print(f"特征索引: {best_features}")
# print(f"总共有 {len(best_features)} 个独特特征")

# # 显示对应的Excel行号
# excel_rows = [idx + 1 for idx in best_features]
# print(f"\n对应的Excel行号:")
# print(excel_rows)


# # 使用示例：
# best_indices, best_fitness, best_traj = get_best_trajectory_features(prop_id=0, f_n=4)
# print(f"\n最佳轨迹 {best_traj}:")
# print(f"特征索引: {best_indices}")
# print(f"适应度值: {best_fitness}")

# # 如果需要查看对应的Excel行号
# excel_rows = [idx + 1 for idx in best_indices]
# print(f"对应Excel行号: {excel_rows}")


# # 使用示例：
# results = get_all_trajectories_best_features(prop_id=0, f_n=4)

# # 按轨迹ID排序输出
# for traj_id in sorted(results.keys()):
#     print(f"轨迹 {traj_id}:")
#     print(f"  特征索引: {results[traj_id]['indices']}")
#     print(f"  适应度值: {results[traj_id]['fitness']}")

# # 统计分析（可选）
# all_indices = [results[traj_id]["indices"] for traj_id in results]
# all_fitness = [results[traj_id]["fitness"] for traj_id in results]

# print("\n统计信息:")
# print(f"平均适应度: {np.mean(all_fitness):.4f} ± {np.std(all_fitness):.4f}")

# all_features = [f for traj in all_indices for f in traj]
# feature_counts = Counter(all_features)

# print("\n特征出现频率（top 10）:")
# for feature, count in feature_counts.most_common(20):
#     print(f"特征 {feature}: 出现 {count} 次")

# # 获取所有独特的特征索引
# unique_features = sorted(set(all_features))
# print("\n所有被选中的特征索引:")
# print(unique_features)
# print(f"总共有 {len(unique_features)} 个独特特征被选中")


# # 使用示例：
# indices, fitness = get_best_features(prop_id=0, f_n=4, traj_id=0)
# print(f"最佳特征索引: {indices}")
# print(f"适应度值: {fitness}")


# # 使用示例
# try:
#     # 先读取一个文件看看结构
#     result = load_best_features(prop_id=0, f_n=4, traj_id=0)
#     print("\n文件内容示例：")
#     print(result)
# except Exception as e:
#     print(f"读取出错: {str(e)}")


# # 使用示例：
# # 读取特定轨迹的结果
# result = load_final_feature_selection(prop_id=0, f_n=4, traj_id=0)
# print(f"实验ID: {result['exp_id']}")
# print(f"选择的特征索引: {result['best_features']['indices']}")
# print(f"适应度值: {result['best_features']['fitness']}")

# # 或者读取所有轨迹的结果
# all_results = load_final_feature_selection(prop_id=0, f_n=4)


# # 获取所有轨迹的最佳特征索引
# all_results = load_final_feature_selection(prop_id=0, f_n=4)
# best_features = [result["best_features"]["indices"] for result in all_results]
# print("所有轨迹的最佳特征索引：")
# for i, features in enumerate(best_features):
#     print(f"轨迹 {i}: {features}")
