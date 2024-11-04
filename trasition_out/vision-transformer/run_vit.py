import subprocess
import time
import csv
import os

# 定义要运行的命令列表
commands = [
    # 'python train_ViT.py --epochs 10',
    # 'python train_ViT.py --epochs 20',
    # 'python train_ViT.py --epochs 25',
    # 'python train_ViT.py --epochs 30',
    # 'python train_ViT.py --epochs 35',
    # 'python train_ViT.py --epochs 40',
    # 'python train_ViT.py --epochs 45',
    # 'python train_ViT.py --epochs 50'
    # 'python train_ViT.py --epochs 80',
    # 'python train_ViT.py --epochs 85',
    # 'python train_ViT.py --epochs 55',
    'python train_ViT.py --epochs 30',
    'python train_ViT.py --epochs 35',
    'python train_ViT.py --epochs 25',
    'python train_ViT.py --epochs 15',
    'python train_ViT.py --epochs 40',
    'python train_ViT.py --epochs 20'
]

times = []
csv_file_path = 'run_ViT_times.csv'
if not os.path.exists(csv_file_path):
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Command', 'Elapsed Time (Seconds)'])

# 执行每个命令并记录时间
for cmd in commands:
    start_time = time.time()
    subprocess.run(cmd, shell=True)
    elapsed_time = time.time() - start_time
    times.append(elapsed_time)
    
    # 将命令和运行时间写入CSV文件
    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([cmd, elapsed_time])

# 打印每个命令的运行时间
for i, cmd in enumerate(commands):
    print(f"Command: {cmd}\nTime elapsed: {times[i]} seconds\n")
