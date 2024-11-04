import subprocess
import time

# 定义要运行的命令列表
commands = [
    'CUDA_VISIBLE_DEVICES=6 python train_retinanet.py --epochs 2',
    'CUDA_VISIBLE_DEVICES=6 python train_retinanet.py --epochs 4',
    'CUDA_VISIBLE_DEVICES=6 python train_retinanet.py --epochs 50',
    'CUDA_VISIBLE_DEVICES=6 python train_retinanet.py --epochs 45',
    'CUDA_VISIBLE_DEVICES=6 python train_retinanet.py --epochs 55',
    'CUDA_VISIBLE_DEVICES=6 python train_retinanet.py --epochs 60',
    'CUDA_VISIBLE_DEVICES=6 python train_retinanet.py --epochs 65',
    'CUDA_VISIBLE_DEVICES=6 python train_retinanet.py --epochs 70',
    'CUDA_VISIBLE_DEVICES=6 python train_retinanet.py --epochs 30',
    'CUDA_VISIBLE_DEVICES=6 python train_retinanet.py --epochs 25',
    'CUDA_VISIBLE_DEVICES=6 python train_retinanet.py --epochs 40',
    'CUDA_VISIBLE_DEVICES=6 python train_retinanet.py --epochs 90',
    'CUDA_VISIBLE_DEVICES=6 python train_retinanet.py --epochs 80',
    'CUDA_VISIBLE_DEVICES=6 python train_retinanet.py --epochs 35',
    'CUDA_VISIBLE_DEVICES=6 python train_retinanet.py --epochs 100',
    'CUDA_VISIBLE_DEVICES=6 python train_retinanet.py --epochs 85',
    'CUDA_VISIBLE_DEVICES=6 python train_retinanet.py --epochs 80',
    'CUDA_VISIBLE_DEVICES=7 python train_retinanet.py --epochs 75'
    # 可以继续添加更多命令
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
    print(elapsed_time)

# 打印每个命令的运行时间
for i, cmd in enumerate(commands):
    print(f"Command: {cmd}\nTime elapsed: {times[i]} seconds\n")
