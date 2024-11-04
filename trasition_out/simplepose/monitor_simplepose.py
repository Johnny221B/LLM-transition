import subprocess
import time
import os
import csv
import psutil  # 导入psutil库

task_runtime = 1000  # 任务运行时间限制

tasks = [
    'CUDA_VISIBLE_DEVICES=7 python main.py --config_path configs/dp_fast_pose_0.yaml',
    'CUDA_VISIBLE_DEVICES=7 python main.py --config_path configs/dp_fast_pose_1.yaml',
    'CUDA_VISIBLE_DEVICES=7 python main.py --config_path configs/dp_fast_pose_2.yaml',
    'CUDA_VISIBLE_DEVICES=7 python main.py --config_path configs/dp_fast_pose_3.yaml'
    'CUDA_VISIBLE_DEVICES=7 python main.py --config_path configs/dp_fast_pose_4.yaml',
    'CUDA_VISIBLE_DEVICES=7 python main.py --config_path configs/dp_fast_pose_5.yaml',
    'CUDA_VISIBLE_DEVICES=7 python main.py --config_path configs/dp_fast_pose_6.yaml',
    'CUDA_VISIBLE_DEVICES=7 python main.py --config_path configs/dp_fast_pose_7.yaml'
    'CUDA_VISIBLE_DEVICES=7 python main.py --config_path configs/dp_fast_pose_8.yaml',
    'CUDA_VISIBLE_DEVICES=7 python main.py --config_path configs/dp_fast_pose_9.yaml',
    'CUDA_VISIBLE_DEVICES=7 python main.py --config_path configs/dp_fast_pose_10.yaml',
    'CUDA_VISIBLE_DEVICES=7 python main.py --config_path configs/dp_fast_pose_11.yaml',
    'CUDA_VISIBLE_DEVICES=7 python main.py --config_path configs/dp_fast_pose_12.yaml',
    'CUDA_VISIBLE_DEVICES=7 python main.py --config_path configs/dp_fast_pose_13.yaml',
    'CUDA_VISIBLE_DEVICES=7 python main.py --config_path configs/dp_fast_pose_14.yaml',
    'CUDA_VISIBLE_DEVICES=7 python main.py --config_path configs/dp_fast_pose_15.yaml'
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

with open('simplepose_power_usage.csv', 'w', newline='') as file:
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
