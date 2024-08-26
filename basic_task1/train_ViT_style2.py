import torch

import torchvision

import torchvision.transforms as transforms

import torch.optim as optim

import torch.nn as nn

import timm

import wandb

import time

 

# 初始化 wandb

wandb.init(project="gpu-performance-benchmark",name="ViT-style2")

 

# 配置参数

config = {

    "batch_size": 32,

    "num_workers": 2,

    "learning_rate": 0.001,

    "num_epochs": 80,

    "model_name": 'vit_base_patch16_224',

    "num_classes": 10,

    "dropout_rate": 0.5,

    "scheduler_step_size": 10,

    "scheduler_gamma": 0.1,

    "device": torch.device("cuda:6" if torch.cuda.is_available() else "cpu")

}

 

# 数据预处理和加载

def prepare_data(batch_size, num_workers):

    transform = transforms.Compose([

        transforms.Resize(256),

        transforms.CenterCrop(224),

        transforms.ToTensor(),

        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

    ])

   

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

   

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

   

    return trainloader, testloader

 

# 模型定义

def build_model(model_name, num_classes, dropout_rate, device):

    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)

    model.dropout = nn.Dropout(dropout_rate)

    return model.to(device)

 

# 训练函数

def run_training(trainloader, model, criterion, optimizer, scheduler, device, num_epochs):

    train_loss = 0.0

    start_time = time.time()  # 开始时间记录

 

    for epoch in range(num_epochs):

        model.train()

        epoch_start_time = time.time()

 

        for inputs, labels in trainloader:

            inputs, labels = inputs.to(device), labels.to(device)

 

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

 

            train_loss += loss.item()

 

        scheduler.step()

 

        elapsed_time = time.time() - epoch_start_time

        avg_train_loss = train_loss / len(trainloader)

 

        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Time: {elapsed_time:.2f}s')

        wandb.log({"epoch": epoch + 1, "train_loss": avg_train_loss, "epoch_time": elapsed_time})

 

    total_training_time = time.time() - start_time  # 总时间记录

    wandb.log({"total_training_time": total_training_time})

    print(f'Total Training Time: {total_training_time:.2f} seconds')

 

# 验证函数

def run_evaluation(testloader, model, criterion, device):

    model.eval()

    correct = 0

    total = 0

    val_loss = 0.0

   

    with torch.no_grad():

        for inputs, labels in testloader:

            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            val_loss += loss.item()

           

            _, predicted = torch.max(outputs, 1)

            correct += (predicted == labels).sum().item()

            total += labels.size(0)

   

    avg_val_loss = val_loss / len(testloader)

    accuracy = 100 * correct / total

    print(f'Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}%')

    wandb.log({"val_loss": avg_val_loss, "val_accuracy": accuracy})

 

# 主程序

def main(config):

    trainloader, testloader = prepare_data(config["batch_size"], config["num_workers"])

    model = build_model(config["model_name"], config["num_classes"], config["dropout_rate"], config["device"])

   

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config["scheduler_step_size"], gamma=config["scheduler_gamma"])

   

    run_training(trainloader, model, criterion, optimizer, scheduler, config["device"], config["num_epochs"])

    run_evaluation(testloader, model, criterion, config["device"])

 

    torch.save(model.state_dict(), './vit_cifar10_final.pth')

    wandb.finish()

 

if __name__ == "__main__":

    main(config)