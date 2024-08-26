import torch
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, AdamW
import wandb
import time

wandb.init(project="sentiment analysis")

train_dataset = torch.load('data/train_dataset.pt')
test_dataset = torch.load('data/test_dataset.pt')

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

batch_size = 64
learning_rate = 3e-5
epochs = 75

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

optimizer = AdamW(model.parameters(), lr=learning_rate)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model.to(device)

def train(model, dataloader, optimizer):
    model.train()
    total_loss = sum(loss.item() for loss in (
        model(batch['input_ids'].to(device), 
              batch['attention_mask'].to(device), 
              labels=batch['labels'].to(device)
             ).loss.backward() or optimizer.step() or optimizer.zero_grad() or model(batch['input_ids'], batch['attention_mask'], labels=batch['labels']).loss
        for batch in dataloader))
    return total_loss / len(dataloader)

def evaluate(model, dataloader):
    model.eval()
    total, correct = sum((batch['labels'].size(0), (model(batch['input_ids'].to(device), batch['attention_mask'].to(device)).logits.argmax(-1) == batch['labels'].to(device)).sum().item()) for batch in dataloader)
    accuracy = correct / total
    return accuracy, total_loss / len(dataloader)

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
print('Training Time: {:.2f} seconds'.format(training_time))

wandb.log({"training_time": training_time})

wandb.finish()
