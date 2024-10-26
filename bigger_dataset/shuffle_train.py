import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import shufflenet_v2_x0_5  # 可以选择不同大小的预训练模型
import torch.backends.cudnn as cudnn
import os

class ShuffleNetV2Custom(nn.Module):
    def __init__(self, num_classes=1000):
        super(ShuffleNetV2Custom, self).__init__()
        self.base_model = shufflenet_v2_x0_5(pretrained=False)  # 不使用自动下载的预训练模型
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)


def train(model, device, train_loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    corrects = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * data.size(0)
        _, preds = torch.max(outputs, 1)
        corrects += torch.sum(preds == target.data)

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = corrects.double() / len(train_loader.dataset)
    return epoch_loss, epoch_acc


def validate(model, device, val_loader, criterion):
    model.eval()
    val_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    val_loss /= len(val_loader.dataset)
    val_acc = correct / len(val_loader.dataset)
    return val_loss, val_acc


def main():
    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 数据预处理
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
    ])

    # 加载数据集
    train_dataset = datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
    val_dataset = datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    # 加载预训练模型
    model_load_path = '../shufflenetv2_x0.5-f707e7126e.pth'
    if os.path.exists(model_load_path):
        model = ShuffleNetV2Custom(num_classes=10).to(device)
        
        # 加载预训练权重，并忽略不匹配的部分
        pretrained_dict = torch.load(model_load_path)
        model_dict = model.state_dict()
        
        # 仅加载匹配的部分
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print(f"Loaded pretrained model from {model_load_path}")
    else:
        model = ShuffleNetV2Custom(num_classes=10).to(device)
        print("Pretrained model not found.")

    # 设置训练参数
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # 主训练循环
    num_epochs = 1
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, device, train_loader, optimizer, criterion)
        val_loss, val_acc = validate(model, device, val_loader, criterion)
        scheduler.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}]')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # 指定模型保存路径
            model_save_path = './shufflenetv2_cifar10.pth'
            torch.save(model.state_dict(), model_save_path)

    print('Training completed.')


if __name__ == "__main__":
    main()
