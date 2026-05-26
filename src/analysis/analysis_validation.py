import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd
import numpy as np

from core.config import REPORTS_PATH, XGBOOST_FEATURES, get_data_path, get_model_path

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
        df = pd.read_csv(get_data_path(store, raw=False)).dropna().tail(7)
        model_path = get_model_path(store, "xgboost")
        loaded = joblib.load(model_path)
        if isinstance(loaded, dict):
            model = loaded.get("model", loaded)
        elif isinstance(loaded, tuple):
            model = loaded[0]
        else:
            model = loaded

        X = df[[feature for feature in XGBOOST_FEATURES if feature in df.columns]].copy()
        for feature in XGBOOST_FEATURES:
            if feature not in X.columns:
                X[feature] = 0
        X = X[XGBOOST_FEATURES].ffill().bfill().fillna(0)
        y_real = df['Num_Customers']
        y_pred = np.maximum(model.predict(X), 0)
        table = pd.DataFrame({
            'Loja': store,
            'Dia': df['Date'],
            'Clientes Reais': y_real.values,
            'Clientes Previstos': y_pred.round(0)
        })
        print(table)
        all_results.append(table)

    final_table = pd.concat(all_results, ignore_index=True)
    REPORTS_PATH.mkdir(parents=True, exist_ok=True)
    output_path = REPORTS_PATH / 'forecast_validation_all_stores.csv'
    final_table.to_csv(output_path, index=False)
    print(f"\n✔ Tabela final guardada em {output_path}")

    return final_table

if __name__ == '__main__':
    analysis_validation()
