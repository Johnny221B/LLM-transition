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
