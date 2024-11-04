import os
import json
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModel
import re

# matplotlib.rcParams['font.family'] = 'Times New Roman'
# 设置设备
device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')

# 加载模型和分词器
model_path = '/home/jingxuan/.cache/modelscope/hub/AI-ModelScope/starcoder2-7b'  # 请替换为你的模型路径
tokenizer = AutoTokenizer.from_pretrained(model_path)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model = AutoModel.from_pretrained(model_path).to(device)
model.eval()

# 读取并处理JSON文件
def read_json(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
    return data

# 获取embeddings
def get_embeddings(codes):
    embeddings = []
    for code in codes:
        inputs = tokenizer(code, return_tensors='pt', padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            # 获取最后一个隐藏层的输出
            embedding = outputs.last_hidden_state[:, -1, :]
            embeddings.append(embedding.squeeze(0))
    return torch.stack(embeddings)

# 计算欧氏距离矩阵
def euclidean_distance_matrix(embeddings):
    dist_matrix = torch.cdist(embeddings, embeddings, p=2).cpu().numpy()
    return dist_matrix

# 绘制热力图
def plot_heatmap(matrix, title, filename, row_labels, col_labels):
    plt.rcParams.update(matplotlib.rcParamsDefault)  # Reset to default
    plt.rcParams.update({'font.size': 25})  # Set global font size larger for axes labels
    plt.figure(figsize=(15, 15))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True,
                annot_kws={'size': 20}) # xticklabels=col_labels,yticklabels=row_labels
    plt.title(title)
    # plt.xlabel('Sample Index')
    # plt.ylabel('Sample Index')
    plt.xticks(rotation=90)  
    plt.yticks(rotation=0)
    plt.savefig(filename,bbox_inches='tight')
    plt.show()


def read_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def clean_code(code):
    # return re.sub(r'\n+', '\n', code)
    return re.sub(r'^\s*\n', '', code, flags=re.MULTILINE)

def format_code_files_to_json(file_paths, times, output_file):
    if len(file_paths) != len(times):
        raise ValueError("File paths and times lists must have the same length.")
    
    formatted_data = []
    for file_path, time in zip(file_paths, times):
        code = read_file(file_path)
        # code = code.strip()
        code = clean_code(code)
        formatted_data.append({"code": code.strip(), "time": time})
    
    with open(output_file, 'w') as f:
        json.dump(formatted_data, f, indent=4)

# 主函数
def main(json_file, heatmap_filename):
    data = read_json(json_file)
    codes = [item['code'] for item in data]  # 提取所有代码段
    embeddings = get_embeddings(codes)      # 生成embeddings
    dist_matrix = euclidean_distance_matrix(embeddings)  # 计算距离矩阵
    row_labels = ['code1', 'code2', 'code3', 'code4', 'style1', 'translation1', 'style2', 'translation2']
    col_labels = ['code1', 'code2', 'code3', 'code4', 'style1', 'translation1', 'style2', 'translation2']
    plot_heatmap(dist_matrix, 'Euclidean Distance Heatmap', heatmap_filename, row_labels, col_labels)
    

# file_paths = [
#     "/home/jingxuan/base_task/basic_task1/train_gan.py",
#     "/home/jingxuan/base_task/basic_task1/train_gan1.py",
#     "/home/jingxuan/base_task/basic_task1/train_gan2.py",
#     "/home/jingxuan/base_task/basic_task1/train_gan3.py",
#     "/home/jingxuan/base_task/basic_task1/train_gan4.py",
#     "/home/jingxuan/base_task/basic_task1/train_gan5.py",
#     "/home/jingxuan/base_task/basic_task1/train_gan6.py",
#     "/home/jingxuan/base_task/basic_task1/train_gan7.py",
#     "/home/jingxuan/base_task/basic_task1/train_gan8.py", 
#     "/home/jingxuan/base_task/basic_task1/train_gan9.py" 
# ]
# file_paths = [
#     "/home/jingxuan/base_task/basic_task1/train_resnet50.py",
#     "/home/jingxuan/base_task/basic_task1/train_resnet501.py",
#     "/home/jingxuan/base_task/basic_task1/train_resnet502.py",
#     "/home/jingxuan/base_task/basic_task1/train_resnet503.py",
#     "/home/jingxuan/base_task/basic_task1/train_resnet504.py",
#     "/home/jingxuan/base_task/basic_task1/train_resnet505.py",
#     "/home/jingxuan/base_task/basic_task1/train_resnet506.py",
#     "/home/jingxuan/base_task/basic_task1/train_resnet507.py",
#     "/home/jingxuan/base_task/basic_task1/train_resnet508.py",
#     "/home/jingxuan/base_task/basic_task1/train_resnet509.py"
# ]
# file_paths = [
#     "/home/jingxuan/base_task/basic_task1/train_ViT.py",
#     "/home/jingxuan/base_task/basic_task1/train_ViT1.py",
#     "/home/jingxuan/base_task/basic_task1/train_ViT2.py",
#     "/home/jingxuan/base_task/basic_task1/train_ViT3.py",
#     "/home/jingxuan/base_task/basic_task1/train_ViT4.py",
#     "/home/jingxuan/base_task/basic_task1/train_ViT5.py",
#     "/home/jingxuan/base_task/basic_task1/train_ViT6.py",
#     "/home/jingxuan/base_task/basic_task1/train_ViT7.py",
#     "/home/jingxuan/base_task/basic_task1/train_ViT8.py",
#     "/home/jingxuan/base_task/basic_task1/train_ViT9.py"
# ]
# file_paths = [
#     "/home/jingxuan/base_task/basic_task2/run_bert.py",
#     "/home/jingxuan/base_task/basic_task2/run_bert1.py",
#     "/home/jingxuan/base_task/basic_task2/run_bert2.py",
#     "/home/jingxuan/base_task/basic_task2/run_bert3.py",
#     "/home/jingxuan/base_task/basic_task2/run_bert4.py",
#     "/home/jingxuan/base_task/basic_task2/run_bert5.py",
#     "/home/jingxuan/base_task/basic_task2/run_bert6.py",
#     "/home/jingxuan/base_task/basic_task2/run_bert7.py",
#     "/home/jingxuan/base_task/basic_task2/run_bert8.py",
#     "/home/jingxuan/base_task/basic_task2/run_bert9.py"
# ]

# file_paths = [
#     "/home/jingxuan/base_task/basic_task2/run_bert1.py",
#     "/home/jingxuan/base_task/basic_task2/run_bert2.py",
#     "/home/jingxuan/base_task/basic_task1/train_resnet501.py",
#     "/home/jingxuan/base_task/basic_task1/train_resnet502.py",
#     "/home/jingxuan/base_task/basic_task1/train_gan2.py",
#     "/home/jingxuan/base_task/basic_task1/train_gan1.py",
#     "/home/jingxuan/base_task/basic_task1/train_ViT7.py",
#     "/home/jingxuan/base_task/basic_task1/train_ViT8.py"
# ]

# file_paths = [
#     "/home/jingxuan/base_task/basic_task2/run_bert1.py",
#     "/home/jingxuan/base_task/basic_task2/run_bert2.py",
#     "/home/jingxuan/base_task/basic_task2/run_bert3.py",
#     "/home/jingxuan/base_task/basic_task2/run_bert4.py",
#     "/home/jingxuan/base_task/basic_task1/train_ViT5.py",
#     "/home/jingxuan/base_task/basic_task1/train_ViT6.py",
#     "/home/jingxuan/base_task/basic_task1/train_ViT7.py",
#     "/home/jingxuan/base_task/basic_task1/train_ViT8.py"
# ]

# file_paths = [
#     "/home/jingxuan/base_task/basic_task2/run_bert.py",
#     "/home/jingxuan/base_task/basic_task2/run_bert1.py",
#     "/home/jingxuan/base_task/basic_task2/run_bert2.py",
#     "/home/jingxuan/base_task/basic_task2/run_bert3.py",
#     "/home/jingxuan/base_task/basic_task2/run_bert_style1.py",
#     "/home/jingxuan/base_task/basic_task2/run_bert_style1_modify.py",
#     "/home/jingxuan/base_task/basic_task2/run_bert_style2.py",
#     "/home/jingxuan/base_task/basic_task2/run_bert_style2_modify.py"
# ]
# file_paths = [
#     "/home/jingxuan/base_task/basic_task1/train_gan.py",
#     "/home/jingxuan/base_task/basic_task1/train_gan1.py",
#     "/home/jingxuan/base_task/basic_task1/train_gan2.py",
#     "/home/jingxuan/base_task/basic_task1/train_gan3.py",
#     "/home/jingxuan/base_task/basic_task1/train_gan_style1.py",
#     "/home/jingxuan/base_task/basic_task1/train_gan_style1_modify.py",
#     "/home/jingxuan/base_task/basic_task1/train_gan_style2.py",
#     "/home/jingxuan/base_task/basic_task1/train_gan_style2_modify.py"
# ]
# file_paths = [
#     "/home/jingxuan/base_task/basic_task1/train_resnet50.py",
#     "/home/jingxuan/base_task/basic_task1/train_resnet501.py",
#     "/home/jingxuan/base_task/basic_task1/train_resnet502.py",
#     "/home/jingxuan/base_task/basic_task1/train_resnet503.py",
#     "/home/jingxuan/base_task/basic_task1/train_resnet_style1.py",
#     "/home/jingxuan/base_task/basic_task1/train_resnet_style1_modify.py",
#     "/home/jingxuan/base_task/basic_task1/train_resnet_style2.py",
#     "/home/jingxuan/base_task/basic_task1/train_resnet_style2_modify.py"
# ]

file_paths = [
    "/home/jingxuan/base_task/basic_task1/train_ViT.py",
    "/home/jingxuan/base_task/basic_task1/train_ViT1.py",
    "/home/jingxuan/base_task/basic_task1/train_ViT2.py",
    "/home/jingxuan/base_task/basic_task1/train_ViT3.py",
    "/home/jingxuan/base_task/basic_task1/train_ViT_style1.py",
    "/home/jingxuan/base_task/basic_task1/train_ViT_style1_modify.py",
    "/home/jingxuan/base_task/basic_task1/train_ViT_style2.py",
    "/home/jingxuan/base_task/basic_task1/train_ViT_style2_modify.py"
]

file_name = 'train.json'
fold_path = '/home/jingxuan/linear_probing'
times = [0] * len(file_paths)
format_code_files_to_json(file_paths, times, file_name)

# 运行主函数
if __name__ == '__main__':
    json_file = os.path.join(fold_path, file_name)
    heatmap_filename = 'heatmap.png'
    main(json_file, heatmap_filename)
