import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from core.config import STORES, XGBOOST_FEATURES
from forecasting.ml_model import XGBoostForecaster
from optimization.methods_global import OptimizationMethodsGlobal
from core.profit_calculator import calculate_daily_profit


# =========================
# FORECAST
# =========================
def generate_forecasts():
    forecasts = {}

    for store in STORES:
        print(f"[Forecast] {store}")

        df = pd.read_csv(f'data/enriched/{store}_features.csv')

        features = [f for f in XGBOOST_FEATURES if f in df.columns]

        X = df[features].ffill().bfill().fillna(0)
        y = df['Num_Customers']

        X_train = X.iloc[:-7]
        y_train = y.iloc[:-7]
        X_test = X.iloc[-7:]

        model = XGBoostForecaster(store)
        model.train(X_train, y_train)

        forecast = model.predict(X_test, n_periods=7)
        forecasts[store] = np.array(forecast)

    return forecasts


# =========================
# EXTRACT RESULTS
# =========================
def extract_daily_results(solution, forecasts, scenario):
    n_stores = len(STORES)
    n = n_stores * 7

    J = solution[0:n]
    X = solution[n:2*n]
    PR = solution[2*n:3*n]

    idx = 0
    rows = []

    for store in STORES:
        for d in range(7):

            daily = {
                'num_customers': forecasts[store][d],
                'J': int(J[idx]),
                'X': int(X[idx]),
                'PR': PR[idx],
                'is_weekend': d >= 5
            }

            profit = calculate_daily_profit(
                 daily['num_customers'],
                 int(J[idx]),
                 int(X[idx]),
                 PR[idx],
                 d >= 5,
                 store
            )[0]

            rows.append({
                'store': store,
                'day': d + 1,
                'scenario': scenario,
                'customers': int(forecasts[store][d]),
                'J': int(J[idx]),
                'X': int(X[idx]),
                'PR': round(PR[idx], 2),
                'profit': round(profit, 2)
            })

            idx += 1

    return pd.DataFrame(rows)


# =========================
# RUN SCENARIOS
# =========================
def run():

    forecasts = generate_forecasts()

    all_results = []

    scenarios = ['O1', 'O2', 'O3']

    for scenario in scenarios:

        print(f"\n=== RUNNING {scenario} ===")

        optimizer = OptimizationMethodsGlobal(
            stores=STORES,
            forecasts=forecasts,
            objective=scenario,
            seed=0
        )

        if scenario == 'O1':
            result = optimizer.particle_swarm(max_iter=100)

        elif scenario == 'O2':
            result = optimizer.particle_swarm(max_iter=100)

        elif scenario == 'O3':
            result = optimizer.particle_swarm(max_iter=100)

        df = extract_daily_results(result['solution'], forecasts, scenario)

        all_results.append(df)

    # juntar tudo
    final_df = pd.concat(all_results)

    # guardar
    final_df.to_csv("daily_results_all_scenarios.csv", index=False)

    print("\n[OK] Ficheiro criado: daily_results_all_scenarios.csv")

    # mostrar Philadelphia
    df_phil = final_df[final_df['store'] == 'philadelphia']

    print("\n=== PHILADELPHIA ===")
    print(df_phil)


if __name__ == "__main__":
    run()