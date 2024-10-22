import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import models
from torchvision.datasets import ImageFolder, CIFAR10
from tqdm import tqdm  # 导入 tqdm 库以显示进度条
from torch.utils.data import DataLoader  # 导入 DataLoader

# 可用的模型列表
AVAILABLE_MODELS = {
    "densenet121": models.densenet121,
    "densenet161": models.densenet161,
    "densenet169": models.densenet169,
    "densenet201": models.densenet201
}


def get_model(model_input, num_classes):
    if model_input in AVAILABLE_MODELS:
        model = AVAILABLE_MODELS[model_input](pretrained=True)
        # 修改最后的分类器层
        model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
        return model
    elif os.path.isfile(model_input):
        model_name = model_input.split('/')[-1].split('_')[0]
        if model_name in AVAILABLE_MODELS:
            model = AVAILABLE_MODELS[model_name](pretrained=False)
            # 修改最后的分类器层
            model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
            model.load_state_dict(torch.load(model_input))
            print(f'Model weights loaded from {model_input}')
            return model
        else:
            raise ValueError(f"Unsupported model name derived from path: {model_name}")
    else:
        raise ValueError(f"Unsupported model input: {model_input}")


def load_data(data_path, batch_size):
    if data_path == 'cifar10':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        trainset = CIFAR10(root='./data', train=True, download=True, transform=transform)
        testset = CIFAR10(root='./data', train=False, download=True, transform=transform)

    elif os.path.isdir(data_path):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        trainset = ImageFolder(root=os.path.join(data_path, 'train'), transform=transform)
        testset = ImageFolder(root=os.path.join(data_path, 'test'), transform=transform)

    else:
        raise ValueError("数据目录不存在或不是有效的目录。")

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader, testloader, len(trainset.classes)


def get_loss_function(loss_name):
    if loss_name == 'cross_entropy':
        return nn.CrossEntropyLoss()
    elif loss_name == 'mse':
        return nn.MSELoss()
    else:
        raise ValueError(f"Unsupported loss function: {loss_name}")


def get_optimizer(optimizer_name, model_parameters, learning_rate):
    if optimizer_name == 'adam':
        return optim.Adam(model_parameters, lr=learning_rate)
    elif optimizer_name == 'sgd':
        return optim.SGD(model_parameters, lr=learning_rate, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def main(args):
    # 设置要使用的 GPU ID
    torch.cuda.set_device(args.gpu_id)

    # 记录训练开始的时间
    start_time = time.time()

    # 加载数据
    trainloader, testloader, num_classes = load_data(args.data_path, args.batch_size)

    # 加载模型
    model = get_model(args.model, num_classes)

    # 选择设备：CUDA 或 CPU
    device = torch.device(f"cuda:{args.gpu_id}" if args.use_cuda and torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 定义损失函数和优化器
    criterion = get_loss_function(args.loss_function)
    optimizer = get_optimizer(args.optimizer, model.parameters(), args.learning_rate)

    # 训练模型
    for epoch in range(args.num_epochs):
        model.train()  # 设置为训练模式
        running_loss = 0.0

        # 使用 tqdm 显示进度条
        with tqdm(enumerate(trainloader), total=len(trainloader), desc=f'Epoch {epoch + 1}/{args.num_epochs}') as pbar:
            for i, (inputs, labels) in pbar:
                inputs, labels = inputs.to(device), labels.to(device)

                # 前向传播
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # 反向传播和优化
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                # 更新进度条
                pbar.set_postfix(loss=running_loss / (i + 1))

        print(f'Epoch [{epoch + 1}/{args.num_epochs}], Average Loss: {running_loss / len(trainloader):.4f}')

    # 测试模型
    model.eval()  # 设置为评估模式
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy: {100 * correct / total:.2f}%')

    # 记录训练结束的时间
    end_time = time.time()

    # 计算训练持续的时间
    elapsed_time = end_time - start_time
    print(f'Training completed in: {elapsed_time:.2f} seconds')

    # 创建输出文件夹
    os.makedirs(args.output_dir, exist_ok=True)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='在数据集上训练 DenseNet 模型')
    parser.add_argument('--data_path', type=str, default='./data/flowers',
                        choices=['cifar10'],
                        help='使用的数据名称或路径，例如cifar10 或 ./data/my_data（默认: cifar10）.my_data 应包含 train 和 test 子目录')

    # 列出可用模型并设置默认模型
    parser.add_argument('--model', type=str, default='densenet121',
                        choices=AVAILABLE_MODELS.keys(),
                        help='使用的模型名称或路径，例如 densenet201 或 ./saved_models/model_weights.pth（默认: densenet121）。 .pth 文件应包含模型权重名称，如 densenet121_custom.pth')

    parser.add_argument('--output_dir', type=str, default='./saved_models/',
                        help='输出文件夹路径，保存训练结果')

    parser.add_argument('--use_cuda', action='store_true', default=True, help='如果可用，使用 CUDA（默认: True）')
    parser.add_argument('--loss_function', type=str, choices=['cross_entropy', 'mse'],
                        default='cross_entropy', help='使用的损失函数（默认: cross_entropy）')
    parser.add_argument('--optimizer', type=str, choices=['adam', 'sgd'],
                        default='adam', help='使用的优化器（默认: adam）')
    parser.add_argument('--batch_size', type=int, default=16, help='训练和测试的批处理大小（默认: 2）')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='优化器的学习率（默认: 0.001）')
    parser.add_argument('--num_epochs', type=int, default=1, help='训练的轮数（默认: 1）')

    args = parser.parse_args()
    main(args)
