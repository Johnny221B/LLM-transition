import subprocess
import time
import csv
import os

# 定义要运行的命令列表
commands = [
    'python train_VGG0.py --epochs 10 --batch_size 32',
    'python train_VGG0.py --epochs 20 --batch_size 16',
    'python train_VGG0.py --epochs 25 --batch_size 32',
    'python train_VGG0.py --epochs 30 --batch_size 32',
    'python train_VGG0.py --epochs 35 --batch_size 32',
    'python train_VGG0.py --epochs 40 --batch_size 16',
    'python train_VGG0.py --epochs 45 --batch_size 32',
    'python train_VGG0.py --epochs 50 --batch_size 32',
    'python train_VGG0.py --epochs 80 --batch_size 16',
    'python train_VGG0.py --epochs 85 --batch_size 32',
    'python train_VGG0.py --epochs 55 --batch_size 16',
    'python train_VGG0.py --epochs 65 --batch_size 16',
    'python train_VGG0.py --epochs 100 --batch_size 32',
    'python train_VGG0.py --epochs 90 --batch_size 32',
    'python train_VGG0.py --epochs 75 --batch_size 16',
    'python train_VGG0.py --epochs 95 --batch_size 16'
    # 'python train_VGG8.py',
    # 'python train_VGG9.py',
    # 'python train_VGG10.py',
    # 'python train_VGG11.py',
    # 'python train_VGG12.py',
    # 'python train_VGG13.py',
    # 'python train_VGG14.py',
    # 'python train_VGG15.py'
]

times = []
csv_file_path = 'run_vgg_times.csv'
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
