import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import time
import argparse
import GPUtil

# Set up argument parser
parser = argparse.ArgumentParser(description='Train ResNet on CIFAR10 with GPU monitoring')
parser.add_argument('--epochs', type=int, default=1, help='number of epochs to train for')
args = parser.parse_args()

# Transforms for training and test data
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
])

transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
])

# Data loaders for training and testing
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)

# Model setup
model = torchvision.models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 10)  # CIFAR10 has 10 classes

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Device configuration
device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training parameters
num_epochs = args.epochs
gpu_usages = []
start_time = time.time()

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 200 == 199:
            avg_loss = running_loss / 200
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, avg_loss))
            running_loss = 0.0

            gpus = GPUtil.getGPUs()
            gpu_usage = gpus[0].load * 100
            gpu_usages.append(gpu_usage)
            print(f'GPU Usage at batch {i + 1}: {gpu_usage:.2f}%')

    scheduler.step()

    # Validation phase
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
    val_accuracy = 100 * correct / total
    print(f'Validation Loss: {val_loss:.3f}, Validation Accuracy: {val_accuracy:.2f}%')

end_time = time.time()
training_time = end_time - start_time
average_gpu_usage = sum(gpu_usages) / len(gpu_usages) if gpu_usages else 0
print('Average GPU Usage: {:.2f}%'.format(average_gpu_usage))
print('Training Time: {:.2f} seconds'.format(training_time))

# Save the trained model
PATH = './resnet50_cifar10.pth'
torch.save(model.state_dict(), PATH)
