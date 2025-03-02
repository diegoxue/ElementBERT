import glob
import os

base_dir = "results"

for prop_id in range(0, 3):
    for f_n in range(1, 9):
        dir_name = f"{base_dir}/prop_{prop_id}_fn_{f_n}"
        os.makedirs(dir_name, exist_ok=True)  # 如果目录不存在，则创建，如果存在，则不创建
        
        # 将prop_id_fn_n.txt文件移动到prop_id/fn_n目录下
        file_path = f"{base_dir}/{prop_id}-{f_n}-*.pkl"
        for file in glob.glob(file_path):
            os.rename(file, f"{dir_name}/{os.path.basename(file)}")

