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
module load cuda/12.2

# 激活 NLP 虚拟环境
conda activate /work/users/l/i/linyuliu/jingxuan/miniconda3/envs/NLP
export PYTHONPATH=/work/users/l/i/linyuliu/jingxuan/miniconda3/envs/NLP/lib/python3.10/site-packages:$PYTHONPATH

# 定义输出文件
output_csv="epoch_times.csv"

# 检查文件是否已存在。如果不存在，则添加表头
if [ ! -f "$output_csv" ]; then
    echo "epoch,start_time,end_time,duration" > $output_csv
fi

# 定义epoch数组
epochs=(10 20 30 40 50 60 70 80 90 100)  # 你可以根据需要修改这个数组

for epoch in "${epochs[@]}"
do
    start_time=$(date +%s)  # 记录开始时间
    srun python train_overnet.py --epochs $epoch
    end_time=$(date +%s)  # 记录结束时间
    duration=$((end_time - start_time))  # 计算持续时间
    echo "$epoch,$(date -d @$start_time +'%Y-%m-%d %H:%M:%S'),$(date -d @$end_time +'%Y-%m-%d %H:%M:%S'),$duration" >> $output_csv
done
