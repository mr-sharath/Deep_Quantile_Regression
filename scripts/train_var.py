# scripts/train_var.py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model.quantile_rnn import MogrifierLSTM, QuantileLoss

# --- Load Data ---
df = pd.read_csv("data/blk_processed.csv", index_col="Date", parse_dates=True)
data = df['Close_Scaled'].values.astype(np.float32)

# --- Create Sequences ---
window_size = 30
horizon = 1  # 1-day ahead
X, y = [], []

for i in range(len(data) - window_size - horizon):
    X.append(data[i:i+window_size])
    y.append(data[i+window_size+horizon-1])

X = np.expand_dims(np.array(X), axis=2)
y = np.array(y).reshape(-1, 1)

train_size = int(0.8 * len(X))
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

# --- Dataloaders ---
batch_size = 64
train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(torch.tensor(X_test), torch.tensor(y_test)), batch_size=batch_size)

# --- Train Model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MogrifierLSTM(input_dim=1, hidden_dim=32).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
quantile = 0.05
criterion = QuantileLoss(quantile=quantile)

num_epochs = 200
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.5f}")

import os
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/quantile_rnn_var.pth")
