#!/usr/bin/env python
"""
Script para testar o fluxo completo de otimização global e geração de reports.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
from core.config import STORES
from optimization.optimize_profit import optimize_global_profit, generate_forecasts

print("=" * 70)
print("TESTE COMPLETO: Otimização Global O3_NS")
print("=" * 70)

# Gerar previsões
print("\n[PASSO 1] Gerando previsões...")
forecasts = generate_forecasts(STORES, forecast_horizon=7)

for store in STORES:
    print(f"  {store.upper()}: {forecasts[store]}")

# Rodar otimização
print("\n[PASSO 2] Rodando otimização O3_NS (1 run para teste)...")
optimize_global_profit(
    stores=STORES,
    forecasts=forecasts,
    objective='O3_NS',
    n_runs=1
)

# Verificar reports gerados
print("\n[PASSO 3] Verificando reports gerados...")
report_dir = Path("reports")

for store in STORES:
    plan_file = report_dir / f"global_plan_{store}_O3_NS_1runs.csv"
    if plan_file.exists():
        import pandas as pd
        df = pd.read_csv(plan_file)
        print(f"\n  {store.upper()} plan:")
        print(df.to_string(index=False))
        
        # Verificar se há zeros
        if (df['Juniores'] == 0).all() or (df['Experts'] == 0).all():
            print(f"  ⚠️  AVISO: Alguns valores estão em zero!")
        else:
            print(f"  ✓ Valores OK")
    else:
        print(f"  ✗ Arquivo não encontrado: {plan_file}")

summary_file = report_dir / "global_summary_O3_NS_1runs.csv"
if summary_file.exists():
    import pandas as pd
    df = pd.read_csv(summary_file)
    print(f"\nResumo global:")
    print(df.to_string(index=False))

print("\n" + "=" * 70)
print("TESTE COMPLETO FINALIZADO")
print("=" * 70)
