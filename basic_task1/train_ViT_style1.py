import torch

import torchvision

import torchvision.transforms as transforms

import torch.optim as optim

import torch.nn as nn

import timm

import wandb

import time

 

# 初始化 wandb

wandb.init(project="gpu-performance-benchmark",name="ViT-style1")

 

# 定义数据预处理

def get_transform():

    return transforms.Compose([

        transforms.Resize(256),

        transforms.CenterCrop(224),

        transforms.ToTensor(),

        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

    ])

 

# 数据集加载函数

def load_datasets(transform):

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    return trainset, testset

 

# 数据加载器函数

def get_dataloaders(trainset, testset, batch_size=32, num_workers=2):

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return trainloader, testloader

 

# 模型定义

class VisionTransformer(nn.Module):

    def __init__(self, model_name='vit_base_patch16_224', num_classes=10, dropout_rate=0.5):

        super(VisionTransformer, self).__init__()

        self.model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)

        self.dropout = nn.Dropout(dropout_rate)

 

    def forward(self, x):

        x = self.model(x)

        x = self.dropout(x)

        return x

 

# 设置设备

def get_device():

    return torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

 

# 训练一个epoch

def train_one_epoch(model, dataloader, criterion, optimizer, device):

    model.train()

    epoch_loss = 0.0

    for inputs, labels in dataloader:

        inputs, labels = inputs.to(device), labels.to(device)

 

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

 

        epoch_loss += loss.item()

 

    return epoch_loss / len(dataloader)

 

# 验证模型

def evaluate_model(model, dataloader, criterion, device):

    model.eval()

    val_loss = 0.0

    correct_predictions = 0

    total_samples = 0

 

    with torch.no_grad():

        for inputs, labels in dataloader:

            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            val_loss += loss.item()

 

            _, predictions = torch.max(outputs.data, 1)

            correct_predictions += (predictions == labels).sum().item()

            total_samples += labels.size(0)

 

    avg_val_loss = val_loss / len(dataloader)

    accuracy = 100 * correct_predictions / total_samples

 

    return avg_val_loss, accuracy

 

# 训练和验证流程

def train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs, device):

    start_time = time.time()

 

    for epoch in range(num_epochs):

        print(f"Epoch {epoch + 1}/{num_epochs}")

 

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        val_loss, val_accuracy = evaluate_model(model, test_loader, criterion, device)

 

        print(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

       

        wandb.log({

            "epoch": epoch + 1,

            "train_loss": train_loss,

            "val_loss": val_loss,

            "val_accuracy": val_accuracy

        })

 

        scheduler.step()

 

    end_time = time.time()

    training_time = end_time - start_time

    print(f'Training Time: {training_time:.2f} seconds')

 

    wandb.log({"training_time": training_time})

    wandb.finish()

 

# 主程序

def main():

    transform = get_transform()

    trainset, testset = load_datasets(transform)

    train_loader, test_loader = get_dataloaders(trainset, testset)

 

    device = get_device()

    model = VisionTransformer().to(device)

 

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

 

    train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs=80, device=device)

 

    model_path = './vit_cifar10.pth'

    torch.save(model.state_dict(), model_path)

 

if __name__ == "__main__":

    main()