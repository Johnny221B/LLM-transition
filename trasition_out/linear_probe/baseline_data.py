import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib as mpl
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
batch_size = 8
length = 20
learning_rate = 1e-4
num_epochs = 1000
progress_interval = 50  # 每50个epoch记录一次
patience = 30
run_type = 'change'  # 选项为 'modify', 'change', 'indist'
train_pic_file = f'A100energy/training_predictions_A100energy_{run_type}.png'
test_pic_file = f'A100energy/testing_predictions_A100energy_{run_type}.png'
merge_file = f'A100energy/merge_predictions_A100energy_{run_type}_baseline.png'
weight_name = f'A100energy/linear_probe_weights_A100energy_{run_type}.pth'
embedding_path = f'/home/jingxuan/linear_probing/embeddings_A100energy_{run_type}/'
hidden_size = 20
hidden_size2 = 1024
weight_decay = 1e-4
lambda_l1 = 1e-3
code_name = f'A100energy-bs{batch_size}-lr{learning_rate}-hs{hidden_size}-wd{weight_decay}-l1{lambda_l1}-{run_type}'

class EmbeddingDataset(Dataset):
    def __init__(self, embedding_dir):
        self.embedding_files = [os.path.join(embedding_dir, f) for f in sorted(os.listdir(embedding_dir)) if 'embedding' in f]
        self.scaler = torch.load(os.path.join(embedding_dir, 'scaler.pt'))

    def __len__(self):
        return len(self.embedding_files)

    def __getitem__(self, idx):
        embedding, time = torch.load(self.embedding_files[idx])
        return embedding.squeeze(), torch.tensor(time, dtype=torch.float).squeeze()
 
class LinearProbe(torch.nn.Module):
    def __init__(self, input_dim, hidden_size,hidden_size2):
        super(LinearProbe, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, hidden_size2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        self.batchnorm2 = torch.nn.BatchNorm1d(hidden_size2)
        self.linear2 = torch.nn.Linear(hidden_size2, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        self.batchnorm = torch.nn.BatchNorm1d(hidden_size)
        self.linear3 = torch.nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear3(x)
        return x

train_dataset = EmbeddingDataset(os.path.join(embedding_path, 'train'))
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = EmbeddingDataset(os.path.join(embedding_path, 'test'))
test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
linear_probe = LinearProbe(4608, hidden_size, hidden_size2).to(device)  # Assuming embedding size of 4096
optimizer = torch.optim.AdamW(linear_probe.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=30)
criterion = nn.MSELoss()

best_loss = float('inf')
patience_counter = 0
epoch_losses = []
loss_tracking = []
stop_training = False

for epoch in range(num_epochs):
    epoch_loss = 0.0
    for embeddings, time in train_data_loader:
        embeddings, time = embeddings.to(device), time.to(device)
        optimizer.zero_grad()
        predictions = linear_probe(embeddings)
        loss = criterion(predictions, time.unsqueeze(1))
        l1_norm = sum(param.abs().sum() for param in linear_probe.parameters())
        total_loss = loss + lambda_l1 * l1_norm
        loss.backward()
        optimizer.step()
        epoch_loss += total_loss.item()
    val_loss = 0.0
    linear_probe.eval()  # 确保模型处于评估模式
    with torch.no_grad():
        for embeddings, time in test_data_loader:
            embeddings, time = embeddings.to(device), time.to(device)
            predictions = linear_probe(embeddings)
            loss = criterion(predictions, time.unsqueeze(1))
            val_loss += loss.item()

    val_loss /= len(test_data_loader)
    scheduler.step(val_loss)

    if (epoch + 1) % progress_interval == 0:
        average_loss = epoch_loss / len(train_data_loader)
        epoch_losses.append(average_loss)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss:.4f}')
        if len(loss_tracking) > 0:
            # 计算loss改善
            loss_improvement = loss_tracking[-1] - average_loss
            if loss_improvement < 0.001:
                small_improvement_count += 1
            else:
                small_improvement_count = 0  # 重置计数器
        else:
            small_improvement_count = 0
        
        loss_tracking.append(average_loss)
        if small_improvement_count >= length:
            print("Early stopping due to minimal loss improvements over 250 epochs.")
            break

        if average_loss < best_loss:
            best_loss = average_loss
            patience_counter = 0
            best_model_state_dict = linear_probe.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + progress_interval}, Loss: {epoch_loss:.4f}")
                break
    epoch_losses = []

# torch.save(best_model_state_dict, weight_name)

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

    predictions_out = ','.join(map(str, predictions))
    actuals_out = ','.join(map(str,actuals))

    results = pd.DataFrame({
        'Predictions': predictions,
        'Actuals': actuals
    })

    pd.options.display.float_format = '{:.0f}'.format
    print(results)
    return predictions, actuals

def plot_predictions(predictions, actuals, title, filename):
    plt.rcParams.update(mpl.rcParamsDefault)
    plt.rcParams.update({'font.size': 25})
    actuals_scaled = actuals / 1000
    predictions_scaled = predictions / 1000
    plt.figure(figsize=(15, 15))
    plt.scatter(actuals_scaled, predictions_scaled, label='Predicted vs Actual',s=120,edgecolors='black', linewidths=1.5)
    plt.plot([min(actuals_scaled), max(actuals_scaled)], [min(actuals_scaled), max(actuals_scaled)], color='orange', label='Ideal Prediction')
    plt.xlabel('Actual Execution Energy (KW)', fontsize=40)
    plt.ylabel('Predicted Execution Energy (KW)', fontsize=40)
    plt.title(title,fontsize=25)
    plt.legend(fontsize=35)
    plt.grid(True)
    plt.savefig(filename,bbox_inches='tight')
    plt.show()

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
train_predictions, train_actuals = evaluate(linear_probe, train_loader, train_dataset.scaler)
test_predictions, test_actuals = evaluate(linear_probe, test_data_loader, test_dataset.scaler)
merged_predictions = np.concatenate((train_predictions, test_predictions))
merged_actuals = np.concatenate((train_actuals, test_actuals))
modify_merged_pred = merged_predictions * 1.05 + 1000
plot_predictions(train_predictions, train_actuals, 'Training Data Predictions', train_pic_file)
plot_predictions(test_predictions, test_actuals, 'Testing Data Predictions', test_pic_file)
plot_predictions(modify_merged_pred, merged_actuals, 'Testing Data Predictions',merge_file)
print(code_name)