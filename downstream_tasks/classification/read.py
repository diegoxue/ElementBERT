import glob
from collections import defaultdict

import joblib
import numpy as np

# 创建数据结构来存储不同特征数量的结果
results = defaultdict(lambda: {"rf": {"empirical": [], "bert": []}})

# 读取所有pkl文件并按特征数量分组
for file in glob.glob("results/*.pkl"):
    # 从文件名中提取特征数量
    f_n = int(file.split("-")[1])
    data = joblib.load(file)

    # 将结果添加到对应的特征数量组
    results[f_n]["rf"]["empirical"].extend(data["empirical"]["rf"])

    results[f_n]["rf"]["bert"].extend(data["bert"]["rf"])


feature_nums = sorted(results.keys(), key=int)  # 确保按数值大小排序
rf_emp_median = [np.median(results[f]["rf"]["empirical"]) for f in feature_nums]
rf_emp_std = [np.std(results[f]["rf"]["empirical"]) for f in feature_nums]
rf_bert_median = [np.median(results[f]["rf"]["bert"]) for f in feature_nums]
rf_bert_std = [np.std(results[f]["rf"]["bert"]) for f in feature_nums]

# 包含特征数量的版本
with open("results/ep_res.txt", "w") as f:
    f.write("Features\tMedian\tStd\n")  # 添加表头
    for feat_num, median, std in zip(feature_nums, rf_emp_median, rf_emp_std):
        f.write(f"{feat_num}\t{median:.4f}\t{std:.4f}\n")

with open("results/bt_res.txt", "w") as f:
    f.write("Features\tMedian\tStd\n")  # 添加表头
    for feat_num, median, std in zip(feature_nums, rf_bert_median, rf_bert_std):
        f.write(f"{feat_num}\t{median:.4f}\t{std:.4f}\n")
