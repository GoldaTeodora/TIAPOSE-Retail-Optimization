import pandas as pd
import pickle
import numpy as np

def analysis_validation():
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
        df = pd.read_csv(f"data/processed/{store}_clean.csv").dropna().tail(7)
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
        print(table)
        all_results.append(table)
    return all_results

if __name__ == '__main__':
    analysis_validation()
