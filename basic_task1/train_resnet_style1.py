import torch

import torchvision

import torchvision.transforms as transforms

import torch.optim as optim

import torch.nn as nn

import wandb

import time

 

# 初始化 wandb

wandb.init(project="cifar100-resnet50",name="resnet50-style1")

 

# 配置参数

config = {

    "batch_size": 32,

    "num_workers": 2,

    "learning_rate": 0.003,

    "num_epochs": 70,

    "step_size": 10,

    "gamma": 0.1,

    "device": torch.device("cuda:1" if torch.cuda.is_available() else "cpu"),

}

 

# 数据预处理

def get_transforms(train=True):

    if train:

        return transforms.Compose([

            transforms.RandomResizedCrop(224),

            transforms.RandomHorizontalFlip(),

            transforms.ToTensor(),

            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),

        ])

    else:

        return transforms.Compose([

            transforms.Resize(256),

            transforms.CenterCrop(224),

            transforms.ToTensor(),

            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),

        ])

 

# 数据加载

def prepare_dataloaders(config):

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=get_transforms(True))

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=get_transforms(False))

   

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"])

    testloader = torch.utils.data.DataLoader(testset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])

   

    return trainloader, testloader

 

# 模型初始化

def initialize_model(device):

    model = torchvision.models.resnet50(pretrained=True)

    model.fc = nn.Linear(model.fc.in_features, 100)

    return model.to(device)

 

# 训练一个 epoch

def train_one_epoch(model, trainloader, criterion, optimizer, device):

    model.train()

    running_loss = 0.0

    for i, data in enumerate(trainloader):

        inputs, labels = data

        inputs, labels = inputs.to(device), labels.to(device)

 

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

 

        running_loss += loss.item()

        if (i + 1) % 200 == 0:

            avg_loss = running_loss / 200

            print(f'Batch {i + 1}, Loss: {avg_loss:.3f}')

            running_loss = 0.0

            wandb.log({"batch": i + 1, "loss": avg_loss})

 

# 模型验证

def evaluate(model, testloader, criterion, device):

    model.eval()

    val_loss = 0.0

    correct = 0

    total = 0

    with torch.no_grad():

        for data in testloader:

            images, labels = data

            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

 

    val_loss /= len(testloader)

    accuracy = 100 * correct / total

    print(f'Validation Loss: {val_loss:.3f}, Validation Accuracy: {accuracy:.2f}%')

    wandb.log({"val_loss": val_loss, "val_accuracy": accuracy})

 

# 训练和验证

def train_and_evaluate(config):

    trainloader, testloader = prepare_dataloaders(config)

    model = initialize_model(config["device"])

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config["step_size"], gamma=config["gamma"])

 

    start_time = time.time()

 

    for epoch in range(config["num_epochs"]):

        print(f'Epoch {epoch + 1}/{config["num_epochs"]}')

        train_one_epoch(model, trainloader, criterion, optimizer, config["device"])

        evaluate(model, testloader, criterion, config["device"])

        scheduler.step()

 

    training_time = time.time() - start_time

    print(f'Training Time: {training_time:.2f} seconds')

    wandb.log({"training_time": training_time})

 

    torch.save(model.state_dict(), './resnet50_cifar100.pth')

    wandb.finish()

 

# 主程序入口

if __name__ == "__main__":

    train_and_evaluate(config)