#!/bin/bash

# SBATCH 配置
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -p a100-gpu
#SBATCH --mem=5g
#SBATCH --qos=gpu_access
#SBATCH -t 3:00:00
#SBATCH --gres=gpu:1

# 加载模块和环境
module purge
module load python/3.11.9
module load cuda/12.2
source /work/users/l/i/linyuliu/jingxuan/jingxuan/bin/activate
export PYTHONPATH=/work/users/l/i/linyuliu/jingxuan/lib/python3.11/site-packages:$PYTHONPATH

# 定义输出文件
output_csv="epoch_times.csv"
if [ ! -f "$output_csv" ]; then
  echo "epoch,start_time,end_time,duration" > $output_csv
fi

# 定义epoch数组
epochs=(10 20 30 40 50 60 70 80 90 100)  # 你可以根据需要修改这个数组

for epoch in "${epochs[@]}"
do
    start_time=$(date +%s)  # 记录开始时间
    srun python train.py --epochs $epoch
    end_time=$(date +%s)  # 记录结束时间
    duration=$((end_time - start_time))  # 计算持续时间
    echo "$epoch,$(date -d @$start_time +'%Y-%m-%d %H:%M:%S'),$(date -d @$end_time +'%Y-%m-%d %H:%M:%S'),$duration" >> $output_csv
done
