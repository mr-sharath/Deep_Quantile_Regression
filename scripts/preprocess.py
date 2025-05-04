# scripts/preprocess.py
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler

RAW_DATA_PATH = 'data/BLK.csv'
PROCESSED_DATA_PATH = 'data/blk_processed.csv'

os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)

def preprocess_data():
    df = pd.read_csv(RAW_DATA_PATH, parse_dates=['Date'], index_col='Date')
    df = df[['Close']].copy()

    # Normalize closing prices
    scaler = MinMaxScaler()
    df['Close_Scaled'] = scaler.fit_transform(df[['Close']])

    # Save processed data
    df.to_csv(PROCESSED_DATA_PATH)
    print(f"Saved processed data to {PROCESSED_DATA_PATH}")

    return df, scaler

if __name__ == "__main__":
    preprocess_data()
