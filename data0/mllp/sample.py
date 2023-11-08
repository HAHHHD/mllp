import os
import random
import shutil
# 设置源文件夹路径
src_dir_path = "processed/miplibCs"
 
# 设置目标文件夹路径
dst_dir_path1 = "processed/miplibCs_train"
dst_dir_path2 = "processed/miplibCs_eval"

os.makedirs(dst_dir_path1, exist_ok = True)
os.makedirs(dst_dir_path2, exist_ok = True)

# 获取源文件夹中所有文件的列表
files = [f for f in os.listdir(src_dir_path)]
 
# 随机选择100个文件，如果你想更改文件数量，只需要更改这个100即可
random_files = random.sample(files, int(len(files) * 0.7))
 
# 将选中的文件复制到目标文件夹中
for f in files:
    file_path = os.path.join(src_dir_path, f)
    if f in random_files:
        shutil.copy(file_path, dst_dir_path1)
    else:
        shutil.copy(file_path, dst_dir_path2)