import os
import subprocess
import time
import csv

def run_commands_and_log():
    original_ld_library_path = os.environ.get('LD_LIBRARY_PATH', '')
    new_path = '/home/linyuliu/.conda/envs/jingxuan/lib/python3.11/site-packages/nvidia/cusparse/lib'
    os.environ['LD_LIBRARY_PATH'] = new_path + ':' + original_ld_library_path

    commands = [
        'CUDA_VISIBLE_DEVICES=0 python train_gru.py --epochs 200 --batch_size 16',
        'CUDA_VISIBLE_DEVICES=0 python train_gru.py --epochs 180 --batch_size 16',
        'CUDA_VISIBLE_DEVICES=0 python train_gru.py --epochs 150 --batch_size 16',
        'CUDA_VISIBLE_DEVICES=0 python train_gru.py --epochs 155 --batch_size 32',
        'CUDA_VISIBLE_DEVICES=0 python train_gru.py --epochs 195 --batch_size 32',
        'CUDA_VISIBLE_DEVICES=0 python train_gru.py --epochs 165 --batch_size 32'
    ]

    filename = 'gru_times.csv'
    file_exists = os.path.isfile(filename)

    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Command', 'Time Elapsed (seconds)'])

        for cmd in commands:
            start_time = time.time()
            subprocess.run(cmd, shell=True)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(elapsed_time)
            writer.writerow([cmd, elapsed_time])

    # 恢复原始的环境变量
    os.environ['LD_LIBRARY_PATH'] = original_ld_library_path

# 调用函数
run_commands_and_log()
