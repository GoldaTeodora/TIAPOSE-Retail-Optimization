import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.config import STORES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from core.config import XGBOOST_FEATURES
from optimization.methods import OptimizationMethods
from forecasting.ml_model import XGBoostForecaster

store = STORES[0]  # ex: baltimore

df = pd.read_csv(f'data/enriched/{store}_features.csv')


available_features = [f for f in XGBOOST_FEATURES if f in df.columns]

print("Features usadas:", available_features)

X = df[available_features].ffill().bfill().fillna(0)
y = df['Num_Customers']

# split simples
X_train = X.iloc[:-7]
y_train = y.iloc[:-7]
X_test = X.iloc[-7:]

# modelo
xgb = XGBoostForecaster(store)
xgb.train(X_train, y_train)

forecast = xgb.predict(X_test, n_periods=7)
forecast = np.array(forecast)

methods = ['random', 'hill_climbing', 'simulated_annealing', 'genetic', 'pso']
seeds = [0, 1, 2, 3, 4]

configs = {
    'low': {
        'random': {'n_iter': 100},
        'hill_climbing': {'max_iter': 100},
        'simulated_annealing': {'max_iter': 200},
        'genetic': {'generations': 50},
        'pso': {'max_iter': 50}
    },
    'high': {
        'random': {'n_iter': 300},
        'hill_climbing': {'max_iter': 300},
        'simulated_annealing': {'max_iter': 500},
        'genetic': {'generations': 100},
        'pso': {'max_iter': 100}
    }
}

results = []

for level, cfg in configs.items():
    for method in methods:
        for seed in seeds:

            np.random.seed(seed)

            optimizer = OptimizationMethods(
                store_name=store,
                customers_forecast=forecast
       )
            start = time.time()

            if method == 'random':
                result = optimizer.random_search(**cfg[method])
            elif method == 'hill_climbing':
                result = optimizer.hill_climbing(**cfg[method])
            elif method == 'simulated_annealing':
                result = optimizer.simulated_annealing(**cfg[method])
            elif method == 'genetic':
                result = optimizer.genetic_algorithm(**cfg[method])
            elif method == 'pso':
                result = optimizer.particle_swarm(**cfg[method])

            elapsed = time.time() - start

            results.append({
                'level': level,
                'method': method,
                'seed': seed,
                'value': result['value'],
                'time': elapsed
            })

df = pd.DataFrame(results)

summary = df.groupby(['level', 'method']).agg({
    'value': ['mean', 'std'],
    'time': 'mean'
}).reset_index()
summary.columns = ['level', 'method', 'mean', 'std', 'avg_time']
summary['level'] = pd.Categorical(summary['level'], categories=['low','high'], ordered=True)
summary = summary.sort_values(['method', 'level'])
summary['efficiency'] = summary['mean'] / (summary['avg_time'] + 1e-10)
summary.to_csv("optimization_experiment_comparison.csv", index=False)

plt.figure(figsize=(8,5))

order = ['low', 'high']

for method in methods:
    subset = summary[summary['method'] == method]
    subset = subset.set_index('level').loc[order].reset_index()

    plt.errorbar(
        subset['level'],
        subset['mean'],
        yerr=subset['std'],
        marker='o',
        capsize=4,
        label=method
    )
    
plt.xlabel("Configuração")
plt.ylabel("Lucro médio")
plt.title("Impacto das iterações nos algoritmos")
plt.legend()
plt.grid(alpha=0.3)

plt.savefig("optimization_iterations_impact.png", dpi=300, bbox_inches='tight')
plt.show()