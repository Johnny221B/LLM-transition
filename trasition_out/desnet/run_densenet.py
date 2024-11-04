import subprocess
import time
import csv
import os

# 定义要运行的命令列表
commands = [
    # 'CUDA_VISIBLE_DEVICES=1 python train_desnet0.py --num_epochs 10',
    # 'CUDA_VISIBLE_DEVICES=1 python train_desnet0.py --num_epochs 20',
    # 'CUDA_VISIBLE_DEVICES=1 python train_desnet0.py --num_epochs 25',
    # 'CUDA_VISIBLE_DEVICES=1 python train_desnet0.py --num_epochs 30',
    # 'CUDA_VISIBLE_DEVICES=1 python train_desnet0.py --num_epochs 35',
    # 'CUDA_VISIBLE_DEVICES=1 python train_desnet0.py --num_epochs 40',
    # 'CUDA_VISIBLE_DEVICES=1 python train_desnet0.py --num_epochs 45',
    # 'CUDA_VISIBLE_DEVICES=1 python train_desnet0.py --num_epochs 50'
    # 'CUDA_VISIBLE_DEVICES=1 python train_desnet0.py --num_epochs 80',
    # 'CUDA_VISIBLE_DEVICES=1 python train_desnet0.py --num_epochs 85',
    # 'CUDA_VISIBLE_DEVICES=1 python train_desnet0.py --num_epochs 55',
    'CUDA_VISIBLE_DEVICES=1 python train_desnet0.py --num_epochs 50',
    'CUDA_VISIBLE_DEVICES=1 python train_desnet0.py --num_epochs 60',
    'CUDA_VISIBLE_DEVICES=1 python train_desnet0.py --num_epochs 45',
    'CUDA_VISIBLE_DEVICES=1 python train_desnet0.py --num_epochs 40',
    'CUDA_VISIBLE_DEVICES=1 python train_desnet0.py --num_epochs 65',
    'CUDA_VISIBLE_DEVICES=1 python train_desnet0.py --num_epochs 140'
]

times = []
csv_file_path = 'run_cyclegan_times.csv'
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
