import subprocess
import time

# 定义要运行的命令列表
commands = [
    'CUDA_VISIBLE_DEVICES=6 python train.py --num_epochs 100 --batch_size 8',
    'CUDA_VISIBLE_DEVICES=6 python train.py --num_epochs 80 --batch_size 8',
    'CUDA_VISIBLE_DEVICES=6 python train.py --num_epochs 120 --batch_size 8',
    'CUDA_VISIBLE_DEVICES=6 python train.py --num_epochs 110 --batch_size 8',
    'CUDA_VISIBLE_DEVICES=6 python train.py --num_epochs 85 --batch_size 8',
    'CUDA_VISIBLE_DEVICES=6 python train.py --num_epochs 70 --batch_size 8',
    'CUDA_VISIBLE_DEVICES=6 python train.py --num_epochs 95 --batch_size 8',
    'CUDA_VISIBLE_DEVICES=6 python train.py --num_epochs 130 --batch_size 8',
    'CUDA_VISIBLE_DEVICES=6 python train.py --num_epochs 115 --batch_size 8',
    'CUDA_VISIBLE_DEVICES=6 python train.py --num_epochs 105 --batch_size 8',
    'CUDA_VISIBLE_DEVICES=6 python train.py --num_epochs 200 --batch_size 8',
    'CUDA_VISIBLE_DEVICES=6 python train.py --num_epochs 180 --batch_size 8',
    'CUDA_VISIBLE_DEVICES=6 python train.py --num_epochs 220 --batch_size 8',
    'CUDA_VISIBLE_DEVICES=6 python train.py --num_epochs 165 --batch_size 8',
    'CUDA_VISIBLE_DEVICES=6 python train.py --num_epochs 145 --batch_size 8',
    'CUDA_VISIBLE_DEVICES=6 python train.py --num_epochs 150 --batch_size 8'
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
