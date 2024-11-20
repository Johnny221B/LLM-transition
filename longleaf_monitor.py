#!/bin/bash
#SBATCH -N 1                
#SBATCH -n 1                
#SBATCH -c 4                
#SBATCH -p a100-gpu       
#SBATCH -t 12:00:00
#SBATCH --mem=5g            
#SBATCH --qos=gpu_access   
#SBATCH --gres=gpu:1       

# 清除已有模块加载，防止冲突
module purge

# 加载需要的模块
module load python/3.11.9
module load cuda/12.2

# 激活虚拟环境
source /work/users/l/i/linyuliu/jingxuan/jingxuan/bin/activate

# 设置 Python 环境变量
export PYTHONPATH=/work/users/l/i/linyuliu/jingxuan/jingxuan/lib/python3.11/site-packages:$PYTHONPATH

# 定义任务列表（train_gan0.py, train_gan1.py, train_gan2.py 等）
tasks=("train_gan0.py" "train_gan1.py" "train_gan2.py" "train_gan3.py" "train_gan4.py" "train_gan5.py" "train_gan6.py" "train_gan7.py" "train_gan8.py" "train_gan9.py")

# 任务运行时间限制（以秒为单位，3600 = 1 小时）
task_runtime=3600

# Python 代码：监控 GPU 能耗并保存到 CSV 文件的函数
monitor_power_script=$(cat << 'END'
import subprocess
import time
import csv
import signal
import sys

terminate = False

def get_power_usage():
    result = subprocess.run(['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'],
                            stdout=subprocess.PIPE,
                            text=True)
    power_usage = result.stdout.strip()
    power_usages = power_usage.split('\n')
    first_gpu_power = power_usages[0]
    return float(first_gpu_power)

def handle_sigterm(signum, frame):
    global terminate
    terminate = True

def monitor_power(task_name, runtime):
    total_power = 0.0
    start_time = time.time()
    with open('gan_power_usage.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([task_name, 'Total Power Usage (W)'])
        
        signal.signal(signal.SIGTERM, handle_sigterm)

        while True:
            current_time = time.time()
            elapsed_time = current_time - start_time

            if terminate or elapsed_time > runtime:
                break

            power = get_power_usage()
            total_power += power
            time.sleep(1)

        print(f"任务 {task_name} 完成，累计能耗: {total_power} W\n")
        writer.writerow([task_name, total_power])

if __name__ == "__main__":
    task_name = sys.argv[1]
    runtime = int(sys.argv[2])
    monitor_power(task_name, runtime)
END
)

# 将 Python 代码写入一个文件 (monitor_power.py)
echo "$monitor_power_script" > monitor_power.py

# 捕捉任务被终止（timeout）的信号，确保能耗记录完成
trap 'echo "任务被中断，确保能耗记录完毕..."; pkill -P $$' SIGTERM

# 循环执行任务
for task in "${tasks[@]}"
do
    echo "正在运行任务: $task"

    # 启动监控脚本，监控任务 $task 的 GPU 能耗
    python monitor_power.py $task $task_runtime &

    # 使用 timeout 限制任务运行时间并运行实际的任务脚本
    timeout $task_runtime python $task

    # 终止监控脚本，如果它还在运行
    pkill -P $$ -f monitor_power.py

    # 等待监控脚本完成
    wait

    echo "任务 $task 已完成。"
done
