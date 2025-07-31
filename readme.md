# 📈 Deep Quantile Regression for VaR & ES Estimation

## 🌐 Project Overview

This project aims to improve the **accuracy and robustness** of market risk measures using deep learning techniques. It targets two key metrics:

* **Value at Risk (VaR)**: Probability-based downside risk estimate.
* **Expected Shortfall (ES)**: Average loss during the worst-case returns beyond VaR.

We use:

* ✨ **Quantile Regression with Mogrifier LSTM** for VaR
* ✨ **Generative Adversarial Network (GAN)** for ES

---

## 🗂️ Folder Structure

```
Deep_Quantile_Regression/
├── data/
│   └── raw/                 # Raw financial data (e.g., BLK.csv)
│
├── models/               # Trained PyTorch model files
│   ├── quantile_rnn_var.pth
│   └── gan_generator_es.pth
│
├── scripts/
│   ├── fetch_data.py       # Download financial data from Stooq
│   ├── preprocess.py        # Preprocess time series
│   ├── train_var_model.py   # Train Mogrifier-LSTM for VaR
│   ├── train_es_gan.py     # Train GAN for ES
│   ├── evaluate_models.py  # Evaluate both models
│   └── utils.py            # Helper functions
│
├── README.md             # Project instructions and details
└── report.md             # Detailed explanation of methods and findings
```

---

## 💡 Features & Models

### ✅ VaR Model: Mogrifier-LSTM Quantile Regressor

* Captures temporal dependencies with long memory and volatility clustering.
* Learns the conditional quantile function.
* Based on RNNs with **Mogrifier** linear layers for richer interactions.

**Hyperparameters:**

```python
input_dim = 1
hidden_dim = 32
num_layers = 2
sequence_length = 20
quantile_alpha = 0.05
epochs = 50
learning_rate = 0.001
```

### 🤖 ES Model: GAN-Based Tail Risk Generator

* Conditional GAN learns to generate tail risk samples beyond the VaR.
* Estimates expected shortfall by averaging generated loss scenarios.

**Hyperparameters:**

```python
noise_dim = 10
hidden_dim = 64
num_epochs = 300
batch_size = 64
learning_rate = 0.0002
```

---

## ⚙️ How to Setup and Run

### 1. 📁 Clone and Setup

```bash
git clone <repo-url>
cd Deep_Quantile_Regression
pip install -r requirements.txt
```

### 2. 📊 Download Data

```bash
python scripts/fetch_data.py
```

### 3. 📝 Preprocess Data

```bash
python scripts/preprocess.py
```

### 4. 🏋️ Train VaR Model

```bash
python scripts/train_var_model.py
```

### 5. 🏋️ Train ES Model

```bash
python scripts/train_es_gan.py
```

### 6. ✍️ Evaluate Models

```bash
python scripts/evaluate_models.py
```

---

## 🔄 Playing with Parameters

* Modify `alpha` for different quantile levels (e.g., 0.01, 0.10)
* Change `hidden_dim`, `sequence_length`, or `noise_dim` in training scripts
* Replace `BLK.csv` with other financial instruments

---

## 📊 Results Summary

### 🌐 Dataset: BlackRock Inc. (BLK) from 2020-05-04 onward

| Metric              | Value   |
| ------------------- | ------- |
| VaR Hit Ratio       | 0.0513  |
| Kupiec Test p-value | 0.8097  |
| Predicted ES        | -0.3366 |
| Historical ES       | 0.5037  |

### ✅ Interpretation:

* **VaR hit ratio** is close to 5%, indicating strong calibration.
* **Kupiec test** confirms statistically consistent performance.
* **ES model** is conservative, indicating realistic tail risk capture.

---

## 📄 Report Summary

### Goals:

* Improve risk estimation using deep learning.
* Align with financial properties like volatility clustering and long memory.

### Methodologies:

* Quantile loss for VaR
* GAN-based sampling for ES

### Best Settings:

* `hidden_dim = 32`, `seq_len = 20`, `alpha = 0.05`
* `GAN noise_dim = 10`, `epochs = 300`

### Contributions:

* Combines **modern sequence models** with **financial theory**.
* Practical approach to **regulatory risk quantification**.

---

## 🚀 Next Steps

* Add LightGBM or GARCH as baseline comparisons
* Use Christoffersen test for further statistical validation
* Extend to multiple stock indices

---

Made by Indu Sai Atla and Sharath Reddy
