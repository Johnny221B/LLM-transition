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
output_csv="specified_train_times.csv"
if [ ! -f "$output_csv" ]; then
  echo "epoch,batch_size,start_time,end_time,duration" > $output_csv
fi

# 定义特定的组合数组，格式为: "epoch,batch_size"
combinations=(
    "10,16"
    "20,32"
    "30,64"
    "40,128"
)

# 循环执行指定的组合
for combo in "${combinations[@]}"
do
    IFS=',' read epoch batch_size <<< "$combo"
    start_time=$(date +%s)  # 记录开始时间
    echo "Running epoch: $epoch, batch size: $batch_size"
    srun python train.py --epochs $epoch --batch_size $batch_size
    end_time=$(date +%s)  # 记录结束时间
    duration=$((end_time - start_time))  # 计算持续时间
    echo "$epoch,$batch_size,$(date -d @$start_time +'%Y-%m-%d %H:%M:%S'),$(date -d @$end_time +'%Y-%m-%d %H:%M:%S'),$duration" >> $output_csv
done
