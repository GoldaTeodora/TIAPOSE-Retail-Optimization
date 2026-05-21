#!/usr/bin/env python
"""
Script para testar O2 e O3_WEIGHTED com constraint relaxado.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd
from core.config import STORES
from optimization.optimize_profit import optimize_global_profit, generate_forecasts

print("=" * 70)
print("TESTE: O2 e O3_WEIGHTED com constraint relaxado")
print("=" * 70)

# Gerar previsões
forecasts = generate_forecasts(STORES, forecast_horizon=7)

# Testar O2
print("\n[TEST 1] O2 - 2 seeds...")
optimize_global_profit(
    stores=STORES,
    forecasts=forecasts,
    objective='O2',
    n_runs=2
)

print("\n[TEST 2] O3_WEIGHTED - 2 seeds...")
optimize_global_profit(
    stores=STORES,
    forecasts=forecasts,
    objective='O3_WEIGHTED',
    n_runs=2
)

# Verificar resultados
print("\n[CHECK] Verificando plans...")
report_dir = Path("reports")

for objective in ['O2', 'O3_WEIGHTED']:
    print(f"\n  {objective}:")
    total_profit = 0
    for store in STORES:
        pattern = f"global_plan_{store}_{objective}"
        matching = list(report_dir.glob(f"{pattern}_*.csv"))
        if matching:
            df = pd.read_csv(matching[-1])  # Last file
            lucro = df['Lucro'].sum()
            total_profit += lucro
            print(f"    {store}: ${lucro:,.0f}")
    print(f"    TOTAL: ${total_profit:,.0f}")

print("\n" + "=" * 70)
