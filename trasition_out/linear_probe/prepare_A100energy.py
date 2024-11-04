import os
import json
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import StandardScaler
import numpy as np

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
run_type = 'change'
model_path = '/home/jingxuan/.cache/modelscope/hub/AI-ModelScope/starcoder2-7b'
train_data_path = '/home/jingxuan/linear_probing/train_A100_energy.json'
test_data_path = f'/home/jingxuan/linear_probing/test{run_type}_A100_energy.json'
embedding_path = f'/home/jingxuan/linear_probing/embeddings_A100energy_{run_type}'

tokenizer = AutoTokenizer.from_pretrained(model_path)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model = AutoModel.from_pretrained(model_path).to(device)
model.eval()

def preprocess_and_save_embeddings(json_file, save_dir, is_train=True, scaler=None):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(json_file, 'r') as f:
        data = json.load(f)
    codes = [item['code'] for item in data]
    times = np.array([item['time'] for item in data]).reshape(-1, 1)
 

    if is_train:
        scaler = StandardScaler()
        scaled_times = scaler.fit_transform(times)
        torch.save(scaler, os.path.join(save_dir, 'scaler.pt'))
    else:
        if scaler is None:
            raise ValueError("Scaler must be provided for test data preprocessing.")
        scaled_times = scaler.transform(times)
 
    dataset = [tokenizer(code, return_tensors='pt', truncation=True, padding=True).to(device) for code in codes]
    with torch.no_grad():
        for idx, inputs in enumerate(dataset):
            outputs = model(**inputs)
            last_hidden_state = outputs.last_hidden_state[:, -1, :]
            torch.save((last_hidden_state, scaled_times[idx]), os.path.join(save_dir, f"embedding_{idx}.pt"))

preprocess_and_save_embeddings(train_data_path, os.path.join(embedding_path, 'train'), is_train=True)
scaler = torch.load(os.path.join(embedding_path, 'train', 'scaler.pt'))
preprocess_and_save_embeddings(test_data_path, os.path.join(embedding_path, 'test'), is_train=False, scaler=scaler)
torch.save(scaler,os.path.join(embedding_path,'test','scaler.pt'))