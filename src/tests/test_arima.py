import sys
import os

# garantir acesso ao src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from metrics import calculate_metrics
from pmdarima import auto_arima


def backtest_arima(df, n_iterations=50, horizon=7):
    results = []

    for i in range(n_iterations):
        train_end = len(df) - (90 - i)

        train = df['Num_Customers'].iloc[:train_end]
        test = df['Num_Customers'].iloc[train_end:train_end + horizon]

        if len(test) < horizon:
            break

        try:
            model = auto_arima(
                train,
                seasonal=True,
                m=7,
                suppress_warnings=True,
                error_action='ignore'
            )

            preds = model.predict(n_periods=horizon)

            metrics = calculate_metrics(test, preds)
            results.append(metrics)

        except:
            continue

    return pd.DataFrame(results).mean()


def run():
    stores = ['baltimore', 'lancaster', 'philadelphia', 'richmond']

    os.makedirs('reports', exist_ok=True)

    for store in stores:
        df = pd.read_csv(f'data/processed/{store}_clean.csv').dropna()

        print(f"\n===== ARIMA: {store.upper()} =====")

        res = backtest_arima(df)

        print(res)

        res.to_csv(f'reports/arima_{store}.csv')


if __name__ == "__main__":
    run()