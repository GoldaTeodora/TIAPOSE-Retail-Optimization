import pandas as pd
import numpy as np

def preprocess_store_data_final(filepath):
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    # ...existing code...
    return df
