import torch

from torch.utils.data import DataLoader

from transformers import BertForSequenceClassification, AdamW

import wandb

import time

 

# 初始化wandb

wandb.init(project="sentiment analysis",name="bert_style2")

 

# 数据集加载

def load_datasets(train_path, test_path):

    return torch.load(train_path), torch.load(test_path)

 

train_dataset, test_dataset = load_datasets('data/train_dataset.pt', 'data/test_dataset.pt')

 

# 模型初始化

def initialize_model(pretrained_model_name, num_labels):

    model = BertForSequenceClassification.from_pretrained(pretrained_model_name, num_labels=num_labels)

    return model

 

model = initialize_model('bert-base-uncased', num_labels=2)

 

# 设置参数

config = {

    "batch_size": 64,

    "learning_rate": 3e-5,

    "epochs": 75

}

 

# 数据加载器

train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)

test_loader = DataLoader(test_dataset, batch_size=config["batch_size"])

 

# 优化器

optimizer = AdamW(model.parameters(), lr=config["learning_rate"])

 

# 设置设备

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

model.to(device)

 

# 训练函数

def train_one_epoch(model, dataloader, optimizer, device):

    model.train()

    cumulative_loss = 0.0

    for batch_index, batch in enumerate(dataloader):

        optimizer.zero_grad()

        input_data = {key: value.to(device) for key, value in batch.items()}

        output = model(**input_data)

        loss = output.loss

        loss.backward()

        optimizer.step()

        cumulative_loss += loss.item()

        if batch_index % 10 == 0:

            print(f"Batch {batch_index}, Loss: {loss.item()}")

    return cumulative_loss / len(dataloader)

 

# 验证函数

def evaluate_model(model, dataloader, device):

    model.eval()

    total_correct = 0

    total_samples = 0

    total_loss = 0.0

    criterion = torch.nn.CrossEntropyLoss()

   

    with torch.no_grad():

        for batch_index, batch in enumerate(dataloader):

            inputs = {key: value.to(device) for key, value in batch.items() if key != 'labels'}

            labels = batch['labels'].to(device)

            output = model(**inputs)

            loss = criterion(output.logits, labels)

            total_loss += loss.item()

            predictions = torch.argmax(output.logits, dim=-1)

            total_correct += (predictions == labels).sum().item()

            total_samples += labels.size(0)

            if batch_index % 100 == 0:

                print(f"Batch {batch_index}, Evaluation Loss: {loss.item()}")

   

    accuracy = total_correct / total_samples

    avg_loss = total_loss / len(dataloader)

    return accuracy, avg_loss

 

# 训练和验证循环

def run_training(model, train_loader, test_loader, optimizer, config, device):

    start_time = time.time()

    for epoch in range(config["epochs"]):

        print(f"Epoch {epoch + 1}/{config['epochs']}")

        train_loss = train_one_epoch(model, train_loader, optimizer, device)

        val_accuracy, val_loss = evaluate_model(model, test_loader, device)

        print(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, Validation Loss: {val_loss:.4f}')

       

        wandb.log({

            "epoch": epoch + 1,

            "train_loss": train_loss,

            "val_accuracy": val_accuracy,

            "val_loss": val_loss

        })

   

    end_time = time.time()

    total_training_time = end_time - start_time

    print(f'Training Time: {total_training_time:.2f} seconds')

   

    wandb.log({"training_time": total_training_time})

    wandb.finish()

 

# 执行训练

run_training(model, train_loader, test_loader, optimizer, config, device)