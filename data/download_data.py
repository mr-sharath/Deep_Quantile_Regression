# data/download_data.py

import os
import pandas as pd
from pandas_datareader import data as pdr

def fetch_data(ticker, start_date='2000-01-01', end_date='2023-12-31', save_dir='data'):
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        df = pdr.DataReader(ticker, 'stooq').sort_index(ascending=True)
        df = df[['Close']].resample('D').last().ffill()  # Daily closing prices
        save_path = os.path.join(save_dir, f"{ticker}.csv")
        df.to_csv(save_path)
        print(f"Saved {ticker} data to {save_path}")
        return df

    except Exception as e:
        print(f"Failed to fetch {ticker}: {e}")
        return None

# Fetch BLK data
fetch_data('BLK')
