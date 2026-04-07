import pandas as pd
import pickle
import numpy as np
import os

stores = ['baltimore', 'lancaster', 'philadelphia', 'richmond']

features = [
    'Day_of_Week','Is_Weekend','Month','Quarter','Day_of_Year',
    'Is_Black_Friday','Is_Tourist_Event','Is_Holiday','Is_Memorial_Day',
    'Event_Nearby','Holiday_Nearby',
    'Lag_Customers_7','Lag_Customers_14',
    'Rolling_Mean_7','Rolling_Std_7','Rolling_Mean_30'
]

all_results = []

for store in stores:
    print(f"\n>>> {store.upper()}")

    df = pd.read_csv(f"data/processed/{store}_clean.csv").dropna()
    df = df.tail(7)

    # modelo horizonte 1
    with open(f"models/{store}/xgb_h1.pkl", "rb") as f:
        model = pickle.load(f)

    X = df[features]
    y_real = df['Num_Customers']

    pred_log = model.predict(X)
    y_pred = np.expm1(pred_log)

    table = pd.DataFrame({
        'Loja': store,
        'Dia': df['Date'],
        'Clientes Reais': y_real.values,
        'Clientes Previstos': y_pred.round(0)
    })

    table['Erro'] = abs(table['Clientes Reais'] - table['Clientes Previstos'])

    print(table)

    all_results.append(table)

# juntar tudo
final_table = pd.concat(all_results)

# guardar para relatório
os.makedirs('reports', exist_ok=True)
final_table.to_csv('reports/forecast_validation_all_stores.csv', index=False)

print("\n✔ Tabela final guardada em reports/forecast_validation_all_stores.csv")