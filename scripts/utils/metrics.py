# utils/metrics.py
import numpy as np

def var_hit_ratio(y_true, y_pred, alpha):
    """
    Calculates how often the actual return exceeds the VaR (should be approx alpha)
    """
    hits = y_true < y_pred
    return hits.sum() / len(hits)

def var_backtest(y_true, y_pred, alpha):
    hit_ratio = var_hit_ratio(y_true, y_pred, alpha)
    print(f"Hit Ratio (should be close to {alpha}): {hit_ratio:.4f}")
