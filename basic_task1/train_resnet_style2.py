import torch

import torchvision

import torchvision.transforms as transforms

import torch.optim as optim

import torch.nn as nn

import wandb

import time

 

# 初始化 wandb

wandb.init(project="cifar100-resnet50",name="resnet50-style2")

 

# 数据预处理

train_transform = transforms.Compose([

    transforms.RandomResizedCrop(224),

    transforms.RandomHorizontalFlip(),

    transforms.ToTensor(),

    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),

])

 

test_transform = transforms.Compose([

    transforms.Resize(256),

    transforms.CenterCrop(224),

    transforms.ToTensor(),

    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),

])

 

# 数据加载

def load_data(batch_size, num_workers):

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

   

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

   

    return trainloader, testloader

 

# 模型定义

def create_model(num_classes):

    model = torchvision.models.resnet50(pretrained=True)

    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model

 

# 训练过程

def train(model, trainloader, criterion, optimizer, device):

    model.train()

    total_loss = 0.0

    for inputs, labels in trainloader:

        inputs, labels = inputs.to(device), labels.to(device)

       

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

       

        total_loss += loss.item()

   

    avg_loss = total_loss / len(trainloader)

    return avg_loss

 

# 验证过程

def validate(model, testloader, criterion, device):

    model.eval()

    val_loss = 0.0

    correct = 0

    total = 0

   

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

    return avg_val_loss, accuracy

 

# 训练和验证循环

def train_and_validate(model, trainloader, testloader, criterion, optimizer, scheduler, num_epochs, device):

    for epoch in range(num_epochs):

        start_time = time.time()

       

        train_loss = train(model, trainloader, criterion, optimizer, device)

        val_loss, val_accuracy = validate(model, testloader, criterion, device)

       

        print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

       

        wandb.log({

            "epoch": epoch + 1,

            "train_loss": train_loss,

            "val_loss": val_loss,

            "val_accuracy": val_accuracy

        })

       

        scheduler.step()

       

        elapsed_time = time.time() - start_time

        print(f"Time: {elapsed_time:.2f}s")

 

# 主程序

def main():

    batch_size = 32

    num_workers = 2

    num_epochs = 70

    learning_rate = 0.001

   

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

   

    trainloader, testloader = load_data(batch_size, num_workers)

    model = create_model(num_classes=100).to(device)

   

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

   

    # 记录训练开始时间

    start_time = time.time()

   

    train_and_validate(model, trainloader, testloader, criterion, optimizer, scheduler, num_epochs, device)

   

    # 计算并记录总训练时间

    total_time = time.time() - start_time

    print(f"Total Training Time: {total_time:.2f} seconds")

    wandb.log({"training_time": total_time})

   

    torch.save(model.state_dict(), './resnet50_cifar100.pth')

    wandb.finish()

 

if __name__ == "__main__":

    main()