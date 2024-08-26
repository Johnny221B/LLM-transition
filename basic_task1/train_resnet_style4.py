import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import wandb
import time

wandb.init(project="cifar100-resnet50")

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
])

test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
])

train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transforms)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True, num_workers=2)

test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transforms)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False, num_workers=2)

model = torchvision.models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 100)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

def train_and_validate(model, train_loader, test_loader, criterion, optimizer, scheduler, device, num_epochs):
    start = time.time()
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            if i % 200 == 199:
                print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {train_loss / 200:.3f}')
                wandb.log({"epoch": epoch + 1, "batch": i + 1, "train_loss": train_loss / 200})
                train_loss = 0.0

        scheduler.step()

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        val_accuracy = 100 * correct / total
        print(f'Epoch {epoch + 1}, Validation Loss: {val_loss / len(test_loader):.3f}, Accuracy: {val_accuracy:.2f}%')
        wandb.log({"epoch": epoch + 1, "val_loss": val_loss / len(test_loader), "val_accuracy": val_accuracy})

    end = time.time()
    print(f'Training Time: {end - start:.2f} seconds')
    wandb.log({"training_time": end - start})

train_and_validate(model, train_loader, test_loader, criterion, optimizer, scheduler, device, num_epochs=70)

torch.save(model.state_dict(), './resnet50_cifar100_model_2.pth')
wandb.finish()
