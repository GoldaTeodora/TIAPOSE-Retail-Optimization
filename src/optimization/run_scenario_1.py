import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import time

from core.config import STORES, XGBOOST_FEATURES
from forecasting.ml_model import XGBoostForecaster
from optimization.methods import OptimizationMethods


# =========================
# CONFIG
# =========================
methods = ['random', 'hill_climbing', 'genetic', 'pso']
seeds = [0, 1, 2]

configs = {
    'random': {'n_iter': 500},
    'hill_climbing': {'max_iter': 200},
    'genetic': {'generations': 100},
    'pso': {'max_iter': 100}
}

# =========================
# EXPERIMENTO
# =========================
results = []

for store in STORES:

    print(f"\n=== STORE: {store} ===")

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
    forecast = np.array(forecast)

    for method in methods:
        for seed in seeds:

            print(f"\nRunning {method} | seed {seed}")

            np.random.seed(seed)

            
            optimizer = OptimizationMethods(
                store_name=store,
                customers_forecast=forecast,
                seed=seed
            )

            start = time.time()

            if method == 'random':
                result = optimizer.random_search(**configs[method])
            elif method == 'hill_climbing':
                result = optimizer.hill_climbing(**configs[method])
            elif method == 'genetic':
                result = optimizer.genetic_algorithm(**configs[method])
            elif method == 'pso':
                result = optimizer.particle_swarm(**configs[method])

            elapsed = time.time() - start

            print(f"Value: {result['value']:.2f} | Time: {elapsed:.2f}s")

            results.append({
                'store': store,
                'method': method,
                'seed': seed,
                'value': result['value'],
                'time': elapsed,
                'iterations': list(configs[method].values())[0]
            })    

# =========================
# SAVE RESULTS
# =========================
df = pd.DataFrame(results)
df.to_csv("scenario_1_results.csv", index=False)

print("\n[OK] Resultados guardados em scenario_1_results.csv")