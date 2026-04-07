import sys
import os

# 🔧 garantir que o Python encontra o src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from metrics import calculate_metrics


def backtest_baselines(df, n_iterations=500, horizon=7):
    results = {
        'Naive': [],
        'S_Naive': [],
        'MA_7': []
    }

    for i in range(n_iterations):
        train_end = len(df) - (90 - i)

        train = df.iloc[:train_end]
        test = df.iloc[train_end:train_end + horizon]

        if len(test) < horizon:
            break

        # --- NAIVE ---
        y_pred_naive = [train['Num_Customers'].iloc[-1]] * horizon
        results['Naive'].append(
            calculate_metrics(test['Num_Customers'], y_pred_naive)
        )

        # --- SEASONAL NAIVE ---
        y_pred_s_naive = df['Lag_Customers_7'].iloc[train_end:train_end + horizon]
        results['S_Naive'].append(
            calculate_metrics(test['Num_Customers'], y_pred_s_naive)
        )

        # --- MOVING AVERAGE ---
        y_pred_ma = df['Rolling_Mean_7'].iloc[train_end:train_end + horizon]
        results['MA_7'].append(
            calculate_metrics(test['Num_Customers'], y_pred_ma)
        )

    # média dos resultados
    final_results = {}

    for model in results:
        df_res = pd.DataFrame(results[model])
        final_results[model] = df_res.mean()

    return final_results


def run():
    stores = ['baltimore', 'lancaster', 'philadelphia', 'richmond']

    os.makedirs('reports', exist_ok=True)

    for store in stores:
        df = pd.read_csv(f'data/processed/{store}_clean.csv').dropna()

        res = backtest_baselines(df)

        print(f"\n===== BASELINES: {store.upper()} =====")

        res_df = pd.DataFrame(res).T
        print(res_df)

        res_df.to_csv(f'reports/baseline_{store}.csv')


if __name__ == "__main__":
    run()