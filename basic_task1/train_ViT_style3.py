import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import timm
import wandb
import time

wandb.init(project="gpu-performance-benchmark")

data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=data_transforms)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=data_transforms)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

class VisionTransformerWithDropout(nn.Module):
    def __init__(self, model_name, num_classes):
        super(VisionTransformerWithDropout, self).__init__()
        self.base_model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.base_model(x)
        return self.dropout(x)

model = VisionTransformerWithDropout('vit_base_patch16_224', 10)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 200 == 199:
            print(f'Batch {i + 1}, Loss: {running_loss / 200:.3f}')
            running_loss = 0.0
            wandb.log({"batch_loss": running_loss / 200})

def validate(model, loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_accuracy = 100 * correct / total
    avg_val_loss = val_loss / len(loader)
    print(f'Validation Loss: {avg_val_loss:.3f}, Accuracy: {val_accuracy:.2f}%')
    wandb.log({"val_loss": avg_val_loss, "val_accuracy": val_accuracy})
    return avg_val_loss, val_accuracy

start_time = time.time()
num_epochs = 70

for epoch in range(num_epochs):
    print(f'Epoch {epoch + 1}/{num_epochs}')
    train_one_epoch(model, train_loader, optimizer, criterion, device)
    validate(model, test_loader, criterion, device)
    scheduler.step()

end_time = time.time()
training_time = end_time - start_time
print(f'Training complete in {training_time:.2f} seconds')

wandb.log({"training_time": training_time})

torch.save(model.state_dict(), './vit_cifar10_model_1.pth')
wandb.finish()
