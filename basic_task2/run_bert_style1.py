import torch

from torch.utils.data import DataLoader

from transformers import BertForSequenceClassification, AdamW

import wandb

import time

 

wandb.init(project="sentiment analysis",name="bert_style1")

 

# 数据集加载

train_dataset = torch.load('data/train_dataset.pt')

test_dataset = torch.load('data/test_dataset.pt')

 

# 模型初始化

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

 

# 超参数设置

batch_size = 64

learning_rate = 3e-5

epochs = 75

 

# 数据加载器

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_loader = DataLoader(test_dataset, batch_size=batch_size)

 

# 优化器

optimizer = AdamW(model.parameters(), lr=learning_rate)

 

# 设备设置

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

model.to(device)

 

# 训练函数

def train(model, dataloader, optimizer):

    model.train()

    total_loss = 0.0

    for batch_idx, batch in enumerate(dataloader):

        optimizer.zero_grad()

        inputs = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**inputs)

        loss = outputs.loss

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 10 == 0:

            print(f"Batch {batch_idx}, Loss: {loss.item()}")

    return total_loss / len(dataloader)

 

# 验证函数

def evaluate(model, dataloader):

    model.eval()

    correct_predictions = 0

    total_predictions = 0

    total_loss = 0.0

    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():

        for batch_idx, batch in enumerate(dataloader):

            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}

            labels = batch['labels'].to(device)

            outputs = model(**inputs)

            loss = criterion(outputs.logits, labels)

            total_loss += loss.item()

            predictions = torch.argmax(outputs.logits, dim=-1)

            correct_predictions += (predictions == labels).sum().item()

            total_predictions += labels.size(0)

            if batch_idx % 100 == 0:

                print(f"Batch {batch_idx}, Evaluation Loss: {loss.item()}")

    accuracy = correct_predictions / total_predictions

    return accuracy, total_loss / len(dataloader)

 

# 训练和验证循环

start_time = time.time()

for epoch in range(epochs):

    print(f"Epoch {epoch+1}/{epochs}")

    avg_train_loss = train(model, train_loader, optimizer)

    val_accuracy, avg_val_loss = evaluate(model, test_loader)

    print(f'Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, Validation Loss: {avg_val_loss:.4f}')

   

    wandb.log({

        "epoch": epoch + 1,

        "train_loss": avg_train_loss,

        "val_accuracy": val_accuracy,

        "val_loss": avg_val_loss

    })

 

end_time = time.time()

total_training_time = end_time - start_time
print(f'Training Time: {total_training_time:.2f} seconds')

 

wandb.log({"training_time": total_training_time})
wandb.finish()