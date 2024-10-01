import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
batch_size = 8
num_epochs = 2000
hidden_size = 512
learning_rate = 1e-4
weight_decay = 0
lambda_l1 = 1e-5
run_type = 'indist'
gpu_task = 'A100time' # A100time A100energy A6000time A6000energy
embedding_path = f'/home/jingxuan/linear_probing/embeddings_A100time_{run_type}/'
train_pic_file = f'A100time/training_predictions_A100time_{run_type}.png'
test_pic_file = f'A100time/testing_predictions_A100time_{run_type}.png'
code_name = f'A100time-bs{batch_size}-lr{learning_rate}-hs{hidden_size}-wd{weight_decay}-l1{lambda_l1}-{run_type}'

class EmbeddingDataset(Dataset):
    def __init__(self, embedding_dir):
        self.embedding_files = [os.path.join(embedding_dir, f) for f in sorted(os.listdir(embedding_dir)) if 'embedding' in f]
        self.scaler = torch.load(os.path.join(embedding_dir, 'scaler.pt'))

    def __len__(self):
        return len(self.embedding_files)

    def __getitem__(self, idx):
        embedding, time = torch.load(self.embedding_files[idx])
        return embedding.squeeze(), torch.tensor(time, dtype=torch.float).squeeze()

class LinearProbe(nn.Module):
    def __init__(self, input_dim, hidden_size1, hidden_size2):
        super(LinearProbe, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_size2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        self.batchnorm1 = nn.BatchNorm1d(hidden_size2)
        self.linear2 = nn.Linear(hidden_size2, hidden_size1)
        self.batchnorm2 = nn.BatchNorm1d(hidden_size1)
        self.linear3 = nn.Linear(hidden_size1, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear3(x)
        return x

def evaluate(model, data_loader, scaler):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for embeddings, time in data_loader:
            embeddings, time = embeddings.to(device), time.to(device)
            pred_time = model(embeddings)
            predictions.extend(pred_time.cpu().numpy().flatten())
            actuals.extend(time.cpu().numpy().flatten())
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    actuals = scaler.inverse_transform(np.array(actuals).reshape(-1, 1)).flatten()

    results = pd.DataFrame({
        'Predictions': predictions,
        'Actuals': actuals
    })

    pd.options.display.float_format = '{:.0f}'.format
    print(results)
    return predictions, actuals

def plot_predictions(predictions, actuals, title, filename):
    plt.figure(figsize=(8, 6))
    plt.scatter(actuals, predictions, alpha=0.5, label='Predicted vs Actual')
    plt.plot([min(actuals), max(actuals)], [min(actuals), max(actuals)], color='red', label='Ideal Prediction')
    plt.xlabel('Actual Execution Time')
    plt.ylabel('Predicted Execution Time')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.show()

# 读取数据集
train_dataset = EmbeddingDataset(os.path.join(embedding_path, 'train'))
test_dataset = EmbeddingDataset(os.path.join(embedding_path, 'test'))
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 交叉验证设置
k_folds = 4
kf = KFold(n_splits=k_folds, shuffle=True)
best_model = None
lowest_val_loss = float('inf')

# 交叉验证循环
for fold, (train_idx, val_idx) in enumerate(kf.split(train_dataset)):
    print(f'FOLD {fold}')
    print('--------------------------------')

    train_subsampler = Subset(train_dataset, train_idx)
    val_subsampler = Subset(train_dataset, val_idx)

    train_loader = DataLoader(train_subsampler, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subsampler, batch_size=batch_size, shuffle=False)

    model = LinearProbe(4608, hidden_size, 1024).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=30)

    for epoch in range(num_epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device).unsqueeze(1)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        if val_loss < lowest_val_loss:
            lowest_val_loss = val_loss
            best_model = model.state_dict()
        if (epoch + 1) % 50 == 0:
            print(f'Fold {fold}, Epoch {epoch+1}: Val Loss: {val_loss / len(val_loader):.4f}')

# 使用最佳模型在测试集上进行评估
model.load_state_dict(best_model)
test_predictions, test_actuals = evaluate(model, test_loader, test_dataset.scaler)
train_predictions, train_actuals = evaluate(model, train_loader, train_dataset.scaler)
plot_predictions(test_predictions, test_actuals, 'Testing Data Predictions', test_pic_file)
plot_predictions(train_predictions, train_actuals, 'Training Data Predictions', train_pic_file)
print(code_name)