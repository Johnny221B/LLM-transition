import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

class EmbeddingDataset(Dataset):
    def __init__(self, embedding_dir):
        self.embedding_files = [os.path.join(embedding_dir, f) for f in sorted(os.listdir(embedding_dir)) if 'embedding' in f]
        self.scaler = torch.load(os.path.join(embedding_dir, 'scaler.pt'))

    def __len__(self):
        return len(self.embedding_files)

    def __getitem__(self, idx):
        embedding, time = torch.load(self.embedding_files[idx])
        return embedding.squeeze(), torch.tensor(time, dtype=torch.float).squeeze()

# 定义计算欧式距离的函数
def euclidean_distance_matrix(embeddings):
    dist_matrix = torch.cdist(embeddings, embeddings, p=2).cpu().numpy()
    return dist_matrix

def cosine_distance_matrix(embeddings):
    # 计算余弦相似度
    similarity = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)
    # 将余弦相似度转换为余弦距离
    distance_matrix = 1 - similarity
    return distance_matrix.cpu().numpy()

# 提取和组织embeddings
def extract_embeddings(dataset):
    embeddings = []
    for idx in range(len(dataset)):
        embedding, _ = dataset[idx]
        embeddings.append(embedding)
    return torch.stack(embeddings)

# 绘制热力图的函数
def plot_heatmap(matrix, title, filename):
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
    plt.title(title)
    plt.xlabel('Sample Index')
    plt.ylabel('Sample Index')
    plt.savefig(filename)
    plt.show()

# 初始化数据集
dataset_dir = '/home/jingxuan/linear_probing/embeddings_A100_time/train'  # 修改为你的路径
dataset = EmbeddingDataset(dataset_dir)
# train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# 从数据集中提取embeddings
all_embeddings = extract_embeddings(dataset)

# 类别标签和其对应的范围
categories = {
    'GAN': (0, 8),
    'ResNet': (8, 16),
    'ViT': (16, 24),
    'BERT': (24, 32),
    'Mix1':(14,22)
}

# 计算和绘制每个类别的热力图
for label, (start_idx, end_idx) in categories.items():
    category_embeddings = all_embeddings[start_idx:end_idx]
    dist_matrix = euclidean_distance_matrix(category_embeddings)
    # dist_matrix = cosine_distance_matrix(category_embeddings)
    plot_heatmap(dist_matrix, f'Heatmap for {label}', f'heatmap_{label}.png')
