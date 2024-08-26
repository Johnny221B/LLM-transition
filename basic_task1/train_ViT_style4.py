import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import timm
import wandb
import time

wandb.init(project="gpu-performance-benchmark")

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True, num_workers=2)

test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False, num_workers=2)

class VisionTransformer(nn.Module):
    def __init__(self, model_name, num_classes, dropout_rate=0.5):
        super(VisionTransformer, self).__init__()
        self.model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.model(x)
        x = self.dropout(x)
        return x

def train_epoch(model, data_loader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        if batch_idx % 200 == 199:
            print(f'Batch {batch_idx + 1}, Loss: {epoch_loss / (batch_idx + 1):.3f}')
            wandb.log({"epoch_loss": epoch_loss / (batch_idx + 1)})

def validate_epoch(model, data_loader, criterion, device):
    model.eval()
    validation_loss = 0
    correct_predictions = 0
    total_samples = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            validation_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            total_samples += targets.size(0)
            correct_predictions += (preds == targets).sum().item()
    validation_accuracy = 100 * correct_predictions / total_samples
    avg_val_loss = validation_loss / len(data_loader)
    print(f'Validation Loss: {avg_val_loss:.3f}, Accuracy: {validation_accuracy:.2f}%')
    wandb.log({"validation_loss": avg_val_loss, "validation_accuracy": validation_accuracy})
    return avg_val_loss, validation_accuracy

model = VisionTransformer('vit_base_patch16_224', num_classes=10, dropout_rate=0.5)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

epochs = 70
start = time.time()

for epoch in range(epochs):
    print(f'Starting Epoch {epoch + 1}/{epochs}')
    train_epoch(model, train_loader, criterion, optimizer, device)
    validate_epoch(model, test_loader, criterion, device)
    scheduler.step()

end = time.time()
print(f'Training completed in {end - start:.2f} seconds')

wandb.log({"total_training_time": end - start})

torch.save(model.state_dict(), './vit_cifar10_model_2.pth')
wandb.finish()
