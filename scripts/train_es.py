# scripts/train_es.py
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from model.gan_model import Generator, Discriminator

# Load scaled returns
data = pd.read_csv("data/blk_processed.csv", index_col="Date", parse_dates=True)
returns = data['Close_Scaled'].pct_change().dropna().values.astype(np.float32)

# Parameters
seq_len = 30
noise_dim = 20
batch_size = 64
num_epochs = 50
alpha = 0.05

# Real samples (tail events only)
threshold = np.quantile(returns, alpha)
tail_data = returns[returns <= threshold]
real_samples = np.array([tail_data[i:i+seq_len] for i in range(len(tail_data) - seq_len)])

# Prepare dataset
train_loader = DataLoader(TensorDataset(torch.tensor(real_samples)), batch_size=batch_size, shuffle=True)

# Models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
G = Generator(noise_dim, seq_len).to(device)
D = Discriminator(seq_len).to(device)

# Optimizers and Loss
opt_G = torch.optim.Adam(G.parameters(), lr=0.0002)
opt_D = torch.optim.Adam(D.parameters(), lr=0.0002)
criterion = nn.BCELoss()

# Training Loop
for epoch in range(num_epochs):
    G.train()
    for real in train_loader:
        real = real[0].to(device)
        batch_size = real.size(0)

        # Train Discriminator
        noise = torch.randn(batch_size, noise_dim).to(device)
        fake = G(noise).detach()
        d_loss_real = criterion(D(real), torch.ones(batch_size, 1).to(device))
        d_loss_fake = criterion(D(fake), torch.zeros(batch_size, 1).to(device))
        d_loss = d_loss_real + d_loss_fake

        opt_D.zero_grad()
        d_loss.backward()
        opt_D.step()

        # Train Generator
        noise = torch.randn(batch_size, noise_dim).to(device)
        fake = G(noise)
        g_loss = criterion(D(fake), torch.ones(batch_size, 1).to(device))

        opt_G.zero_grad()
        g_loss.backward()
        opt_G.step()

    print(f"Epoch [{epoch+1}/{num_epochs}] D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

# Save generator
import os
os.makedirs("models", exist_ok=True)
torch.save(G.state_dict(), "models/gan_generator_es.pth")
