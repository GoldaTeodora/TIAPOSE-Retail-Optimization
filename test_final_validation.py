#!/usr/bin/env python
"""
Script para rodar otimização completa com múltiplos seeds e validar dados.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd
from core.config import STORES
from optimization.optimize_profit import optimize_global_profit, generate_forecasts

print("=" * 70)
print("TESTE FINAL: Otimização Global com 3 seeds")
print("=" * 70)

# Gerar previsões
forecasts = generate_forecasts(STORES, forecast_horizon=7)

# Rodar otimização com 3 seeds
print("\n[STEP 1] Rodando otimização O3_NS com 3 seeds...")
optimize_global_profit(
    stores=STORES,
    forecasts=forecasts,
    objective='O3_NS',
    n_runs=3
)

print("\n[STEP 2] Validando integridade dos dados...")
report_dir = Path("reports")

total_profit_global = 0
valid_stores = 0

for store in STORES:
    plan_file = report_dir / f"global_plan_{store}_O3_NS_3runs.csv"
    if plan_file.exists():
        df = pd.read_csv(plan_file)
        
        # Validações
        has_zeros_jr = (df['Juniores'] == 0).all()
        has_zeros_ex = (df['Experts'] == 0).all()
        has_negative = (df['Lucro'] < 0).any()
        total_profit = df['Lucro'].sum()
        
        print(f"\n  {store.upper()}:")
        print(f"    Total Lucro Semanal: ${total_profit:,.0f}")
        print(f"    Lucro Diário (min/max): ${df['Lucro'].min():.0f} / ${df['Lucro'].max():.0f}")
        print(f"    Atendimento: {df['Atendidos'].sum()}/{df['Clientes'].sum()} clientes ({100*df['Atendidos'].sum()/df['Clientes'].sum():.1f}%)")
        
        # Status
        status = "✓"
        if has_zeros_jr:
            print(f"    ⚠️  Aviso: Todos os Juniores são 0")
            status = "!"
        if has_zeros_ex:
            print(f"    ⚠️  Aviso: Todos os Experts são 0")
            status = "!"
        if has_negative:
            print(f"    ⚠️  Aviso: Lucro negativo em alguns dias")
            status = "!"
        
        print(f"    Status: {status}")
        
        total_profit_global += total_profit
        valid_stores += 1
    else:
        print(f"  ✗ Arquivo não encontrado: {plan_file}")

print(f"\n[STEP 3] Resumo Final:")
print(f"  Lojas processadas: {valid_stores}/{len(STORES)}")
print(f"  Lucro Total Semanal Global: ${total_profit_global:,.0f}")

# Verificar summary
summary_file = report_dir / "global_summary_O3_NS_3runs.csv"
if summary_file.exists():
    df_summary = pd.read_csv(summary_file)
    print(f"\n  Resumo por Algoritmo:")
    for idx, row in df_summary.iterrows():
        mean_profit = row.get('mean_profit', row.get('best_profit', 0))
        print(f"    {row['algorithm']}: ${mean_profit:,.0f}")

# Verificar pareto front
pareto_file = report_dir / "pareto_front_O3_NS_3runs.csv"
if pareto_file.exists():
    df_pareto = pd.read_csv(pareto_file)
    print(f"\n[STEP 4] Pareto Front (multi-objetivo):")
    print(f"  Soluções no front: {len(df_pareto)}")
    if len(df_pareto) > 0:
        print(f"  Lucro (min/max): ${df_pareto['profit'].min():,.0f} / ${df_pareto['profit'].max():,.0f}")
        print(f"  HR (min/max): {df_pareto['hr'].min():.0f} / {df_pareto['hr'].max():.0f}")

print("\n" + "=" * 70)
print("VALIDAÇÃO CONCLUÍDA")
print("=" * 70)
