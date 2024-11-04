import subprocess
import time
import csv
import os

# 配置文件列表
config_files = [
    'configs/dp_fast_pose_0.yaml',
    'configs/dp_fast_pose_1.yaml',
    # 'configs/dp_fast_pose_2.yaml',
    # 'configs/dp_fast_pose_3.yaml'
    # 'configs/dp_fast_pose_4.yaml',
    # 'configs/dp_fast_pose_5.yaml',
    # 'configs/dp_fast_pose_6.yaml',
    # 'configs/dp_fast_pose_7.yaml'
    # 'configs/dp_fast_pose_8.yaml',
    # 'configs/dp_fast_pose_9.yaml',
    # 'configs/dp_fast_pose_10.yaml',
    # 'configs/dp_fast_pose_11.yaml',
    # 'configs/dp_fast_pose_12.yaml',
    # 'configs/dp_fast_pose_13.yaml',
    'configs/dp_fast_pose_14.yaml',
    'configs/dp_fast_pose_15.yaml'
    # 添加更多配置文件路径
]

times = []
csv_file_path = 'run_times.csv'
if not os.path.exists(csv_file_path):
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Command', 'Elapsed Time (Seconds)'])

for config in config_files:
    cmd = f'CUDA_VISIBLE_DEVICES=7 python main.py --config_path {config}'
    start_time = time.time()
    subprocess.run(cmd, shell=True)
    elapsed_time = time.time() - start_time
    times.append(elapsed_time)
    
    # 将命令和运行时间写入CSV文件
    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([cmd, elapsed_time])

# 输出运行结果
for i, cmd in enumerate(commands):
    print(f"Command: {cmd}\nTime elapsed: {times[i]} seconds")
