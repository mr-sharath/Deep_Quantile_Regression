# scripts/model/gan_model.py
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, noise_dim, seq_len):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, seq_len)
        )

    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self, seq_len):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(seq_len, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
