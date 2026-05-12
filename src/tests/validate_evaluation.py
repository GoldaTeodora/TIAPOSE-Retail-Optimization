import sys
from pathlib import Path

# adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from core.profit_calculator import (
    calculate_daily_profit,
    calculate_weekly_profit,
    evaluate_solution,
    evaluate_solution_global
)

# =========================================================
# TESTE 1 — DAILY PROFIT VALIDATION
# =========================================================

print("\n=== TESTE 1: DAILY PROFIT VALIDATION ===")

daily_profit, units, hr = calculate_daily_profit(
    num_customers=100,
    J=2,
    X=5,
    PR=0.1,
    is_weekend=False,
    store_name='baltimore'
)

print(f"Profit : {daily_profit}")
print(f"Units  : {units}")
print(f"HR Cost: {hr}")


# =========================================================
# TESTE 2 — WEEKLY PROFIT VALIDATION
# =========================================================

print("\n=== TESTE 2: WEEKLY PROFIT VALIDATION ===")

daily_plans = []

for i in range(7):
    daily_plans.append({
        'num_customers': 100,
        'J': 2,
        'X': 5,
        'PR': 0.1,
        'is_weekend': i >= 5
    })

weekly = calculate_weekly_profit(
    daily_plans,
    'baltimore'
)

print(f"Weekly Profit: {weekly['weekly_profit']}")
print(f"Total Units  : {weekly['total_units']}")
print(f"Total HR     : {weekly['total_hr']}")


# =========================================================
# TESTE 3 — O1 VALIDATION
# =========================================================

print("\n=== TESTE 3: O1 VALIDATION ===")

J = np.array([2] * 7)
X = np.array([5] * 7)
PR = np.array([0.1] * 7)
forecast = np.array([100] * 7)

result = evaluate_solution(
    J,
    X,
    PR,
    forecast,
    'baltimore',
    objective='O1'
)

print(result)


# =========================================================
# TESTE 4 — GLOBAL O2 FEASIBLE VALIDATION
# =========================================================

print("\n=== TESTE 4: GLOBAL O2 FEASIBLE VALIDATION ===")

small_forecasts = {
    'baltimore': np.array([20] * 7),
    'lancaster': np.array([20] * 7),
    'philadelphia': np.array([20] * 7),
    'richmond': np.array([20] * 7)
}

small_J = {
    store: np.array([1] * 7)
    for store in small_forecasts
}

small_X = {
    store: np.array([1] * 7)
    for store in small_forecasts
}

small_PR = {
    store: np.array([0.05] * 7)
    for store in small_forecasts
}

result = evaluate_solution_global(
    small_J,
    small_X,
    small_PR,
    small_forecasts,
    objective='O2'
)

print(result)


# =========================================================
# TESTE 5 — GLOBAL O2 INFEASIBLE VALIDATION
# =========================================================

print("\n=== TESTE 5: GLOBAL O2 INFEASIBLE VALIDATION ===")

big_forecasts = {
    'baltimore': np.array([1000] * 7),
    'lancaster': np.array([1000] * 7),
    'philadelphia': np.array([1000] * 7),
    'richmond': np.array([1000] * 7)
}

big_J = {
    store: np.array([20] * 7)
    for store in big_forecasts
}

big_X = {
    store: np.array([20] * 7)
    for store in big_forecasts
}

big_PR = {
    store: np.array([0.3] * 7)
    for store in big_forecasts
}

result = evaluate_solution_global(
    big_J,
    big_X,
    big_PR,
    big_forecasts,
    objective='O2'
)

print(result)


# =========================================================
# TESTE 6 — GLOBAL O3 FEASIBLE VALIDATION
# =========================================================

print("\n=== TESTE 6: GLOBAL O3 FEASIBLE VALIDATION ===")

result = evaluate_solution_global(
    small_J,
    small_X,
    small_PR,
    small_forecasts,
    objective='O3',
    max_profit=100000,
    max_hr=50000
)

print(result)


# =========================================================
# TESTE 7 — PDF DEBUG VALIDATION (BALTIMORE)
# Slides 17 — apenas validação matemática/debug
# =========================================================

print("\n=== TESTE 7: PDF DEBUG VALIDATION - BALTIMORE ===")

PDF_PR = np.array([0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30])

PDF_X = np.array([4, 0, 8, 20, 0, 4, 3])

PDF_J = np.array([0, 10, 4, 0, 5, 5, 4])

forecast = np.array([97, 61, 65, 71, 65, 89, 125])

result = evaluate_solution(
    PDF_J,
    PDF_X,
    PDF_PR,
    forecast,
    'baltimore',
    objective='O1'
)

print(result)


# =========================================================
# TESTE 8 — PDF DEBUG VALIDATION (PHILADELPHIA)
# Slides 18 — apenas validação matemática/debug
# =========================================================

print("\n=== TESTE 8: PDF DEBUG VALIDATION - PHILADELPHIA ===")

forecast = np.array([230, 144, 154, 168, 154, 211, 298])

result = evaluate_solution(
    PDF_J,
    PDF_X,
    PDF_PR,
    forecast,
    'philadelphia',
    objective='O1'
)

print(result)


# =========================================================
# NOTAS IMPORTANTES
# =========================================================

print("\n=== NOTAS ===")

print("""
1. Slides 17 e 18 servem apenas para validação matemática
   das fórmulas de lucro, HR e unidades.

2. Os slides NÃO validam O2/O3 globais.

3. O2 e O3 usam constraint global:
   máximo de 10.000 unidades para as 4 lojas.

4. O2 infeasible deve retornar:
   {'best_value': -inf}

5. O3 depende de O2.
""")