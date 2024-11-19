import subprocess
import time
import os
import csv
import psutil  # 用于管理进程和子进程

# 任务运行时间限制（秒）
task_runtime = 2000
# 能耗采样间隔（秒）
sampling_interval = 1

# 定义任务列表
tasks = [
    "python train_gan.py",
    "python train_gan1.py",
    "python train_gan2.py",
    "python train_gan3.py",
    "python train_gan4.py",
    "python train_gan5.py",
    "python train_gan6.py",
    "python train_gan7.py",
    "python train_gan8.py",
    "python train_gan9.py"
]

# 获取 GPU 当前功耗
def get_power_usage():
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'],
            stdout=subprocess.PIPE,
            text=True
        )
        power_usage = result.stdout.strip()
        power_usages = power_usage.split('\n')
        first_gpu_power = power_usages[0]
        return float(first_gpu_power)
    except Exception as e:
        print(f"获取功耗失败: {e}")
        return 0.0

# 强制终止进程树
def kill_proc_tree(pid, including_parent=True):
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        for child in children:
            child.terminate()
        gone, still_alive = psutil.wait_procs(children, timeout=5)
        if including_parent:
            parent.terminate()
            parent.wait(5)
    except Exception as e:
        print(f"终止任务失败: {e}")

# 打开 CSV 文件并记录任务结果
with open('gan_power_usage.csv', 'a', newline='') as file:
    writer = csv.writer(file)
    # 如果文件为空，则写入标题行
    if os.stat('gan_power_usage.csv').st_size == 0:
        writer.writerow(['Task Name', 'Elapsed Time (s)', 'Total Power Usage (W)'])
        file.flush()  # 确保标题行立即写入

    # 遍历任务列表
    for task in tasks:
        print(f"正在运行任务: {task}")
        process = subprocess.Popen(task, shell=True)
        ps_process = psutil.Process(process.pid)  # 创建 psutil 的 Process 对象

        start_time = time.time()
        total_power = 0.0

        while True:
            current_time = time.time()
            elapsed_time = current_time - start_time

            # 检查任务是否超时
            if elapsed_time > task_runtime:
                kill_proc_tree(process.pid)  # 超时强制终止任务
                print(f"任务 {task} 超过运行时间，已终止。")
                break

            # 检查任务是否已经结束
            if process.poll() is not None:
                print(f"任务 {task} 已完成。")
                break

            # 记录能耗
            power = get_power_usage()
            total_power += power * sampling_interval  # 能耗采样累积
            print(f"当前能耗: {power:.2f} W, 累计能耗: {total_power:.2f} W")
            time.sleep(sampling_interval)

        elapsed_time = time.time() - start_time  # 计算总运行时间
        print(f"任务 {task} 完成，运行时间: {elapsed_time:.2f} 秒，累计能耗: {total_power:.2f} W\n")
        
        # 写入任务结果到 CSV 文件
        writer.writerow([task, elapsed_time, total_power])
        file.flush()  # 确保每次写入都立即保存到文件
