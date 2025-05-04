# scripts/model/quantile_rnn.py
import torch
import torch.nn as nn

class MogrifierLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, mogrify_iters=5, num_layers=1):
        super(MogrifierLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.mogrify_iters = mogrify_iters

        self.x_transform = nn.Linear(input_dim, hidden_dim)
        self.h_transform = nn.Linear(hidden_dim, input_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.output = nn.Linear(hidden_dim, 1)

    def mogrify(self, x, h):
        for i in range(self.mogrify_iters):
            if i % 2 == 0:
                x = x * torch.sigmoid(self.h_transform(h))
            else:
                h = h * torch.sigmoid(self.x_transform(x))
        return x, h

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(1, x.size(0), self.hidden_dim).to(x.device)
        x, _ = self.lstm(x, (h0, c0))
        out = self.output(x[:, -1, :])
        return out

class QuantileLoss(nn.Module):
    def __init__(self, quantile):
        super(QuantileLoss, self).__init__()
        self.quantile = quantile

    def forward(self, preds, target):
        errors = target - preds
        return torch.mean(torch.max((self.quantile - 1) * errors, self.quantile * errors))
