# scripts/evaluate_models.py
import numpy as np
import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from model.quantile_rnn import MogrifierLSTM
from model.gan_model import Generator
from utils.metrics import var_backtest

# === Load Data ===
data = pd.read_csv("data/blk_processed.csv", index_col="Date", parse_dates=True)
scaled_close = data['Close_Scaled'].values.astype(np.float32)
returns = pd.Series(scaled_close).pct_change().dropna().values

# === Load VaR Model ===
window = 30
alpha = 0.05
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

var_model = MogrifierLSTM(input_dim=1, hidden_dim=32, num_layers=1).to(device)
var_model.load_state_dict(torch.load("models/quantile_rnn_var.pth"))
var_model.eval()

X_test = []
y_test = []
for i in range(window, len(scaled_close)):
    X_test.append(scaled_close[i - window:i])
    y_test.append(scaled_close[i])

X_test = torch.tensor(X_test).unsqueeze(-1).to(device)
y_test = torch.tensor(y_test).to(device)

with torch.no_grad():
    y_pred_var = var_model(X_test).squeeze()

# === Evaluate VaR ===
print("\nVaR Evaluation:")
var_backtest(y_test.cpu().numpy(), y_pred_var.cpu().numpy(), alpha)

# === Load ES Model ===
noise_dim = 20
G = Generator(noise_dim=noise_dim, seq_len=window).to(device)
G.load_state_dict(torch.load("models/gan_generator_es.pth"))
G.eval()

z = torch.randn(1000, noise_dim).to(device)
with torch.no_grad():
    samples = G(z).cpu().numpy()

# Calculate ES from bottom alpha% values
es_samples = [np.mean(np.sort(seq)[:int(alpha * window)]) for seq in samples]
print(f"\nEstimated Expected Shortfall (ES) at {alpha:.2f}: {np.mean(es_samples):.6f}")


# kupiec test
from statsmodels.stats.proportion import proportions_ztest
n = len(y_test)
x = (y_test < y_pred_var).sum()
stat, p_value = proportions_ztest(x, n, alpha)
print(f"Kupiec Test P-Value: {p_value:.4f}")
#If p_value > 0.05 → ✅ cannot reject correct calibration

#Naive Historical ES
es_naive = y_test[y_test < y_pred_var].mean()
print(f"Naive Historical ES: {es_naive:.6f}")
#Compare this to the GAN-generated ES. If your model’s ES is slightly more conservative than historical → ✅ good (shows risk-awareness).
