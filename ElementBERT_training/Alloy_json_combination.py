import random

from tqdm import tqdm

random.seed(123)

with open(r"/hy-tmp/TiBert_data/All_abstract_data", "r", encoding="utf-8") as f1:
    All = f1.readlines()

with open(r"/hy-tmp/TiBert_data/Ti_abstract_data", "r", encoding="utf-8") as f2:
    Ti = f2.readlines()


# 去除每行末尾的换行符
All = [line.strip() for line in All]

# 计算要分配给验证集的语料数量，假设为10%
num_eval = round(len(All) * 0.10)

# 随机选择验证集的索引
eval_idx = set(random.sample(range(len(All)), num_eval))

# 初始化训练集和验证集
train_set = []
eval_set = []


# 分配语料到训练集或验证集
for idx in tqdm(range(len(All)), desc="Allocating lines"):
    line = All[idx]
    if idx in eval_idx:
        eval_set.append(line)
    else:
        train_set.append(line)


# 输出结果统计信息
print(f"Total: {len(All)}, Eval: {len(eval_set)}, Train: {len(train_set)}")

# 将训练集和验证集数据写入到新的文件中
with open("train_corpus.txt", "w", encoding="utf-8") as train_file:
    for item in train_set:
        train_file.write(f"{item}\n")

with open("eval_corpus.txt", "w", encoding="utf-8") as eval_file:
    for item in eval_set:
        eval_file.write(f"{item}\n")

print("Script run COMPLETE!!!")

