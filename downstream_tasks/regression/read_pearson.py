import os

import pandas as pd


def extract_top_features(
    freq_file="feature_frequency.csv",
    pearson_file="data/pearson.xlsx",
    output_file="top_30_features.xlsx",
    top_n=30,
):
    """
    提取出现频率最高的特征，并从pearson文件中选取对应列
    Args:
        freq_file: 特征频率统计文件路径
        pearson_file: pearson相关系数文件路径
        output_file: 输出文件路径
        top_n: 选取前N个特征
    """
    # 读取特征频率文件
    freq_df = pd.read_csv(freq_file)

    # 获取前30个特征的索引并+1（因为pearson文件中特征从1开始计数）
    top_indices = [str(i + 1) for i in freq_df["特征索引"].head(top_n)]

    # 读取pearson.xlsx文件
    pearson_df = pd.read_excel(pearson_file)

    # 获取第一列名
    first_column = pearson_df.columns[0]

    # 获取列名（第一行）
    column_names = pearson_df.columns.astype(str)

    # 找到对应的列名索引
    selected_columns = [first_column]  # 先添加第一列
    for idx in top_indices:
        if idx in column_names:
            selected_columns.append(idx)

    # 选取对应的列
    result_df = pearson_df[selected_columns]

    # 保存结果
    result_df.to_excel(output_file, index=False)

    print(f"已选取前{top_n}个特征并保存到: {output_file}")
    print(f"选取的特征索引: {top_indices}")

    return result_df


# 使用示例
if __name__ == "__main__":
    try:
        result_df = extract_top_features()
        print(f"数据形状: {result_df.shape}")
    except FileNotFoundError as e:
        print(f"错误：找不到文件 - {e}")
    except Exception as e:
        print(f"处理过程中出现错误: {e}")
