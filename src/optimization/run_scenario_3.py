import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import time

from core.config import STORES, XGBOOST_FEATURES
from forecasting.ml_model import XGBoostForecaster
from optimization.methods_global import OptimizationMethodsGlobal


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


def compute_profit_per_store(solution, stores, forecasts):
    n_stores = len(stores)
    n = n_stores * 7

    J = solution[0:n]
    X = solution[n:2*n]
    PR = solution[2*n:3*n]

    idx = 0
    results = []

    from core.profit_calculator import calculate_weekly_profit

    for store in stores:
        daily_plans = []

        for d in range(7):
            daily_plans.append({
                'num_customers': forecasts[store][d],
                'J': int(J[idx]),
                'X': int(X[idx]),
                'PR': PR[idx],
                'is_weekend': d >= 5
            })
            idx += 1

        weekly = calculate_weekly_profit(daily_plans, store)

        results.append({
            'store': store,
            'weekly_profit': weekly['weekly_profit'],
            'units': weekly['total_units'],
            'hr_cost': weekly['total_hr']
        })

    return pd.DataFrame(results)


# =========================
# EXPERIMENTO O3
# =========================
def run_experiment():

    print("\n=== CENÁRIO 3 (O3) ===")

    forecasts = generate_forecasts()

    methods = ['genetic', 'pso']   # 🔥 só os bons
    seeds = [0, 1, 2]

    results = []

    best_overall = None
    best_value = -np.inf

    for method in methods:
        for seed in seeds:

            print(f"\nRunning {method} | seed {seed}")

            optimizer = OptimizationMethodsGlobal(
                stores=STORES,
                forecasts=forecasts,
                objective='O3',
                seed=seed
            )

            start = time.time()

            if method == 'genetic':
                result = optimizer.genetic_algorithm(
                    population_size=50,
                    generations=100
                )

            elif method == 'pso':
                result = optimizer.particle_swarm(
                    n_particles=30,
                    max_iter=100
                )

            elapsed = time.time() - start

            print(f"Value: {result['value']:.2f} | Time: {elapsed:.2f}s")

            if result['value'] > best_value:
                 best_value = result['value']
                 best_overall = result['solution']

            results.append({
                'method': method,
                'seed': seed,
                'value': result['value'],
                'time': elapsed
            })

    df = pd.DataFrame(results)
    df.to_csv("scenario_3_results.csv", index=False)

    print("\n[OK] Resultados guardados em scenario_3_results.csv")
    print("\n=== MELHOR SOLUÇÃO GLOBAL (O3) ===")
    extract_solution(best_overall, STORES, forecasts)

    df_profit = compute_profit_per_store(best_overall, STORES, forecasts)

    print("\n=== LUCRO POR LOJA ===")
    print(df_profit)

    df_profit.to_csv("scenario_3_profit_per_store.csv", index=False)


def extract_solution(solution, stores, forecasts):
    n_stores = len(stores)
    n = n_stores * 7

    J = solution[0:n]
    X = solution[n:2*n]
    PR = solution[2*n:3*n]

    idx = 0

    for store in stores:
        print(f"\n=== {store.upper()} ===")

        for d in range(7):
            print(f"Day {d+1}: "
                  f"J={int(J[idx])}, "
                  f"X={int(X[idx])}, "
                  f"PR={round(PR[idx],2)}, "
                  f"Customers={int(forecasts[store][d])}")
            idx += 1

if __name__ == "__main__":
    run_experiment()