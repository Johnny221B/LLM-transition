import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 导入你的GhostNet模型
from ghostnetv3 import ghostnetv3

# 解析命令行参数
parser = argparse.ArgumentParser(description='Train GhostNet on Flowers Dataset')
parser.add_argument('--epochs', type=int, default=2, help='number of epochs to train')
parser.add_argument('--batch-size', type=int, default=8, help='batch size for training')
parser.add_argument('--data-dir', type=str, default='data/flowers', help='directory of the dataset')
args = parser.parse_args()

# 设定基本参数
num_classes = 5  # 分类数目，对应5种花

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图片大小
    transforms.ToTensor(),  # 转为Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 标准化
])

# 数据加载
train_dataset = datasets.ImageFolder(root=f'{args.data_dir}/train', transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 初始化模型
model = ghostnetv3(width=1.0, num_classes=num_classes).to(device)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 训练模型
model.train()
for epoch in range(args.epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 7 == 0:
            print(f'Epoch [{epoch+1}/{args.epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

print("Training complete.")
