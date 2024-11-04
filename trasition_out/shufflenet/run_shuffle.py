import subprocess
import time

# 定义要运行的命令列表
commands = [
    'CUDA_VISIBLE_DEVICES=0 python train.py --num_epochs 50',
    'CUDA_VISIBLE_DEVICES=0 python train.py --num_epochs 100',
    'CUDA_VISIBLE_DEVICES=0 python train.py --num_epochs 40',
    'CUDA_VISIBLE_DEVICES=0 python train.py --num_epochs 65',
    'CUDA_VISIBLE_DEVICES=0 python train.py --num_epochs 90',
    'CUDA_VISIBLE_DEVICES=0 python train.py --num_epochs 150',
    'CUDA_VISIBLE_DEVICES=0 python train.py --num_epochs 120',
    'CUDA_VISIBLE_DEVICES=0 python train.py --num_epochs 130',
    'CUDA_VISIBLE_DEVICES=0 python train.py --num_epochs 115',
    'CUDA_VISIBLE_DEVICES=0 python train.py --num_epochs 85',
    'CUDA_VISIBLE_DEVICES=0 python train.py --num_epochs 150',
    'CUDA_VISIBLE_DEVICES=0 python train.py --num_epochs 135',
    'CUDA_VISIBLE_DEVICES=0 python train.py --num_epochs 95',
    'CUDA_VISIBLE_DEVICES=0 python train.py --num_epochs 160',
    'CUDA_VISIBLE_DEVICES=0 python train.py --num_epochs 155',
    'CUDA_VISIBLE_DEVICES=0 python train.py --num_epochs 125'
]

# 存储每个命令的运行时间
times = []

# 执行每个命令并记录时间
for cmd in commands:
    start_time = time.time()  # 开始时间
    subprocess.run(cmd, shell=True)  # 运行命令
    end_time = time.time()  # 结束时间
    elapsed_time = end_time - start_time  # 计算运行时间
    times.append(elapsed_time)  # 将时间添加到列表中

# 打印每个命令的运行时间
for i, cmd in enumerate(commands):
    print(f"Command: {cmd}\nTime elapsed: {times[i]} seconds\n")
