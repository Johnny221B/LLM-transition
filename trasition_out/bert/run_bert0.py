import torch
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, AdamW
import wandb
import time
import GPUtil

wandb.init(project="gpu-usage-bert",name="bert0")

train_dataset = torch.load('data/train_dataset.pt')
test_dataset = torch.load('data/test_dataset.pt')

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

batch_size = 32
learning_rate = 1e-5
epochs = 60

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

optimizer = AdamW(model.parameters(), lr=learning_rate)

device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
model.to(device)

global_gpu_usages = []  

def train(model, dataloader, optimizer):
    model.train()
    running_loss = 0.0
    for i, batch in enumerate(dataloader):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        
        gpus = GPUtil.getGPUs()
        gpu_usage = gpus[0].load * 100
        global_gpu_usages.append(gpu_usage)
        
        if i % 10 == 0:
            print(f"Batch {i}, Loss: {loss.item()}")
    return running_loss / len(dataloader)

def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            running_loss += loss.item()
            predictions = torch.argmax(outputs.logits, dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            if i % 100 == 0:
                print(f"Batch {i}, Evaluation Loss: {loss.item()}")
            
            
            gpus = GPUtil.getGPUs()
            gpu_usage = gpus[0].load * 100
            global_gpu_usages.append(gpu_usage)

    accuracy = correct / total
    return accuracy, running_loss / len(dataloader)

start_time = time.time()
for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    train_loss = train(model, train_loader, optimizer)
    val_accuracy, val_loss = evaluate(model, test_loader)
    print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, Validation Loss: {val_loss:.4f}')
    
    wandb.log({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "val_accuracy": val_accuracy,
        "val_loss": val_loss
    })

end_time = time.time()
training_time = end_time - start_time
overall_avg_gpu_usage = sum(global_gpu_usages) / len(global_gpu_usages) if global_gpu_usages else 0
print('Training Time: {:.2f} seconds'.format(training_time))
print(f'Overall Average GPU Usage: {overall_avg_gpu_usage:.2f}%')

wandb.log({
    "training_time": training_time,
    "overall_avg_gpu_usage": overall_avg_gpu_usage
})

wandb.finish()
