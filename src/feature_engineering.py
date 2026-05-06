import pandas as pd
import numpy as np

def preprocess_store_data_final(filepath):
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    # 1. TRATAMENTO INICIAL
    df['Pct_On_Sale'] = df['Pct_On_Sale'].fillna(0)
    upper_limit = df['Num_Customers'].quantile(0.99)
    df['Num_Customers_Clean'] = df['Num_Customers'].clip(upper=upper_limit)
    
    # 2. DERIVAR FEATURES DE DATA
    df['Day_of_Week'] = df['Date'].dt.dayofweek + 1 
    df['Is_Weekend'] = df['Date'].dt.dayofweek.isin([5, 6]).astype(int)
    df['Month'] = df['Date'].dt.month
    df['Quarter'] = df['Date'].dt.quarter
    df['Day_of_Year'] = df['Date'].dt.dayofyear
    
    # 3. FERIADOS E EVENTOS
    black_fridays = ['2012-11-23', '2013-11-29']
    df['Is_Black_Friday'] = df['Date'].isin(pd.to_datetime(black_fridays)).astype(int)
    
    # Mapeamento do Bloco Memorial Day (Sexta a Segunda)
    memorial_weekend = [
        '2013-05-24', '2013-05-25', '2013-05-26', '2013-05-27',
        '2014-05-23', '2014-05-24', '2014-05-25', '2014-05-26'
    ]
    df['Is_Memorial_Day'] = df['Date'].isin(pd.to_datetime(memorial_weekend)).astype(int)

    # Feriados Americanos Fixos
    holidays = ['2012-09-03', '2012-10-08', '2012-11-12', '2013-01-01', 
                '2013-01-21', '2013-02-18', '2013-05-27', '2013-07-04']
    df['Is_Holiday'] = df['Date'].isin(pd.to_datetime(holidays)).astype(int)
    
    # Efeito Véspera e Pós-Feriado (Capta o movimento de viagem)
    df['Holiday_Nearby'] = df['Is_Holiday'].shift(1).fillna(0) + df['Is_Holiday'].shift(-1).fillna(0)
    df['Holiday_Nearby'] = (df['Holiday_Nearby'] > 0).astype(int)

    closed_days = ['2012-12-25', '2013-03-31', '2013-12-25', '2014-04-20']
    df.loc[df['Date'].isin(pd.to_datetime(closed_days)), 'Num_Customers_Clean'] = 0
    df['Is_Tourist_Event'] = df['TouristEvent'].map({'Yes': 1, 'No': 0}).fillna(0)
    df['Event_Nearby'] = df['Is_Tourist_Event'].shift(1).fillna(0) + df['Is_Tourist_Event'].shift(-1).fillna(0)
    df['Event_Nearby'] = (df['Event_Nearby'] > 0).astype(int)
    
    # 4. LAG FEATURES
    for lag in [1, 7, 14]:
        df[f'Lag_Customers_{lag}'] = df['Num_Customers_Clean'].shift(lag)

    # 5. ROLLING STATISTICS
    df['Rolling_Mean_7'] = df['Num_Customers_Clean'].shift(1).rolling(window=7).mean()
    df['Rolling_Std_7'] = df['Num_Customers_Clean'].shift(1).rolling(window=7).std()
    df['Rolling_Mean_30'] = df['Num_Customers_Clean'].shift(1).rolling(window=30).mean()
    
    return df

stores = ['baltimore', 'lancaster', 'philadelphia', 'richmond']
for s in stores:
    final_df = preprocess_store_data_final(f'data/raw/{s}.csv')
    final_df.to_csv(f'data/processed/{s}_clean.csv', index=False)
    print(f"Base corrigida para {s}: {final_df.shape[1]} colunas.")