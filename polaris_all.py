import subprocess
import time
import csv
import os  # 用于检查文件是否存在

# 定义要运行的命令
commands = [
    'CUDA_VISIBLE_DEVICES=0 python train_gru.py --epochs 200 --batch_size 16',
    'CUDA_VISIBLE_DEVICES=0 python train_gru.py --epochs 180 --batch_size 16',
    'CUDA_VISIBLE_DEVICES=0 python train_gru.py --epochs 150 --batch_size 16',
    'CUDA_VISIBLE_DEVICES=0 python train_gru.py --epochs 155 --batch_size 32',
    'CUDA_VISIBLE_DEVICES=0 python train_gru.py --epochs 195 --batch_size 32',
    'CUDA_VISIBLE_DEVICES=0 python train_gru.py --epochs 165 --batch_size 32'
]

# 定义CSV文件名
filename = 'gru_times.csv'
file_exists = os.path.isfile(filename)  # 检查文件是否存在

# 使用 'a' 模式打开文件以追加内容
with open(filename, mode='a', newline='') as file:
    writer = csv.writer(file)
    if not file_exists:  # 如果文件不存在，写入标题行
        writer.writerow(['Command', 'Time Elapsed (seconds)'])
        file.flush()  # 确保标题行被写入文件

    # 执行每个命令并记录其运行时间
    for cmd in commands:
        print(f"Running command: {cmd}")
        start_time = time.time()  # 记录开始时间
        subprocess.run(cmd, shell=True)  # 运行命令
        end_time = time.time()  # 记录结束时间
        elapsed_time = end_time - start_time  # 计算运行时间
        print(f"Command finished in {elapsed_time:.2f} seconds")  # 打印时间

        writer.writerow([cmd, elapsed_time])  # 写入命令和运行时间到CSV文件
        file.flush()  # 强制写入磁盘
