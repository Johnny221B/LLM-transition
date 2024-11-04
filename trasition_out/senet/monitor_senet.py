import subprocess
import time
import os
import csv
import psutil  # 导入psutil库

task_runtime = 800  # 任务运行时间限制

tasks = [
    'CUDA_VISIBLE_DEVICES=3 EPOCH=200 python train.py',
    'CUDA_VISIBLE_DEVICES=3 EPOCH=160 python train.py',
    'CUDA_VISIBLE_DEVICES=3 EPOCH=110 python train.py',
    'CUDA_VISIBLE_DEVICES=3 EPOCH=220 python train.py',
    'CUDA_VISIBLE_DEVICES=3 EPOCH=170 python train.py',
    'CUDA_VISIBLE_DEVICES=3 EPOCH=140 python train.py',
    'CUDA_VISIBLE_DEVICES=3 EPOCH=190 python train.py',
    'CUDA_VISIBLE_DEVICES=3 EPOCH=260 python train.py',
    'CUDA_VISIBLE_DEVICES=3 EPOCH=230 python train.py',
    'CUDA_VISIBLE_DEVICES=3 EPOCH=210 python train.py',
    'CUDA_VISIBLE_DEVICES=3 EPOCH=300 python train.py',
    'CUDA_VISIBLE_DEVICES=3 EPOCH=320 python train.py',
    'CUDA_VISIBLE_DEVICES=3 EPOCH=380 python train.py',
    'CUDA_VISIBLE_DEVICES=3 EPOCH=400 python train.py',
    'CUDA_VISIBLE_DEVICES=3 EPOCH=410 python train.py',
    'CUDA_VISIBLE_DEVICES=3 EPOCH=350 python train.py',
]

# tasks = [
#     "python train_gan_style1.py",
#     "python train_gan_style2.py"
# ]

def get_power_usage():
    result = subprocess.run(['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'],
                            stdout=subprocess.PIPE,
                            text=True)
    power_usage = result.stdout.strip()
    power_usages = power_usage.split('\n')
    first_gpu_power = power_usages[0]
    return float(first_gpu_power)

def kill_proc_tree(pid, including_parent=True):    
    parent = psutil.Process(pid)
    children = parent.children(recursive=True)
    for child in children:
        child.terminate()
    gone, still_alive = psutil.wait_procs(children, timeout=5)
    if including_parent:
        parent.terminate()
        parent.wait(5)

with open('senet_power_usage.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Task Name', 'Total Power Usage (W)'])

    for task in tasks:
        print(f"正在运行任务: {task}")
        process = subprocess.Popen(task, shell=True)
        ps_process = psutil.Process(process.pid)  # 创建psutil的Process对象

        start_time = time.time()
        total_power = 0.0
        last_print_time = 0

        while True:
            current_time = time.time()
            elapsed_time = current_time - start_time

            if elapsed_time > task_runtime:
                kill_proc_tree(process.pid)  # 使用psutil终止进程树
                print(f"任务 {task} 超过运行时间，已终止。")
                break

            power = get_power_usage()
            total_power += power

            if current_time - last_print_time >= 100 or elapsed_time > task_runtime:
                print(f"当前能耗: {power} W,累计能耗: {total_power} W")
                last_print_time = current_time

            time.sleep(1)

        print(f"任务 {task} 完成，累计能耗: {total_power} W\n")
        writer.writerow([task, total_power])
