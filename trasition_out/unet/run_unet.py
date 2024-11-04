import subprocess
import time

# 定义要运行的命令列表
commands = [
    'CUDA_VISIBLE_DEVICES=7 python train_unet0.py --epochs 550',
    'CUDA_VISIBLE_DEVICES=7 python train_unet0.py --epochs 600',
    'CUDA_VISIBLE_DEVICES=7 python train_unet0.py --epochs 580',
    'CUDA_VISIBLE_DEVICES=7 python train_unet0.py --epochs 700',
    'CUDA_VISIBLE_DEVICES=7 python train_unet0.py --epochs 660',
    'CUDA_VISIBLE_DEVICES=7 python train_unet0.py --epochs 470',
    'CUDA_VISIBLE_DEVICES=7 python train_unet0.py --epochs 410',
    'CUDA_VISIBLE_DEVICES=7 python train_unet0.py --epochs 610',
    'CUDA_VISIBLE_DEVICES=7 python train_unet0.py --epochs 800',
    'CUDA_VISIBLE_DEVICES=7 python train_unet0.py --epochs 420',
    'CUDA_VISIBLE_DEVICES=7 python train_unet0.py --epochs 720',
    'CUDA_VISIBLE_DEVICES=7 python train_unet0.py --epochs 850',
    'CUDA_VISIBLE_DEVICES=7 python train_unet0.py --epochs 780',
    'CUDA_VISIBLE_DEVICES=7 python train_unet0.py --epochs 750'
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
