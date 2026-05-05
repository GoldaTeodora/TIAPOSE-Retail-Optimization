import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from core.config import STORES
from optimization.methods import OptimizationMethods

store = STORES[0]  # ex: baltimore
forecast = np.array([100, 110, 120, 130, 140, 150, 160])

algorithms = [
    'random',
    'hill_climbing',
    'simulated_annealing',
    'genetic',
    'pso'
]

for algo in algorithms:
    print(f"\n=== Testing {algo} ===")

    try:
        opt = OptimizationMethods(store, forecast, 'O1', method=algo)
        result = opt.optimize()

        print(f"Value: {result.get('value'):.2f}")
        print(f"HR: {result.get('hr', 0)}")
        print(f"Units: {result.get('units', 0)}")

    except Exception as e:
        print("ERRO:", e)