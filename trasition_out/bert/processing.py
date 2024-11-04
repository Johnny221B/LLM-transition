from datasets import load_dataset
from transformers import BertTokenizer
import torch
import os

# 加载IMDB数据集
dataset = load_dataset('imdb')

# 加载预训练的BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 定义tokenizer函数
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

# 对数据集进行tokenize
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 确保每个样本包含 'labels' 键
def rename_label_column(examples):
    examples['labels'] = examples['label']
    return examples

tokenized_datasets = tokenized_datasets.map(rename_label_column, batched=True)

# 设置格式
tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
train_dataset = tokenized_datasets['train']
test_dataset = tokenized_datasets['test']

# 创建保存路径
save_dir = 'data'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 保存处理后的数据集
torch.save(train_dataset, os.path.join(save_dir, 'train_dataset.pt'))
torch.save(test_dataset, os.path.join(save_dir, 'test_dataset.pt'))

print("Datasets saved successfully.")
