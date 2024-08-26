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

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)

class VisionTransformerWithDropout(nn.Module):
    def __init__(self, model_name, num_classes):
        super(VisionTransformerWithDropout, self).__init__()
        self.model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.model(x)
        x = self.dropout(x)
        return x

model = VisionTransformerWithDropout('vit_base_patch16_224', 10)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

num_epochs = 50  
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
            wandb.log({"epoch": epoch + 1, "batch": i + 1, "loss": avg_loss})

    scheduler.step() 

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
    wandb.log({"val_loss": val_loss, "val_accuracy": val_accuracy})

print('Finished Training')

end_time = time.time()
training_time = end_time - start_time
print('Training Time: {:.2f} seconds'.format(training_time))

wandb.log({"training_time": training_time})

PATH = './vit_cifar10.pth'
torch.save(model.state_dict(), PATH)

wandb.finish()
