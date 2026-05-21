#!/usr/bin/env python
"""
Script de teste para validar a função _repair_solution() e o fluxo de otimização.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
from core.config import STORES
from core.profit_calculator import calculate_weekly_profit
from optimization.methods_global import OptimizationMethodsGlobal

# Previsões de exemplo
forecasts = {
    'baltimore': np.array([93, 77, 87, 113, 99, 107, 119]),
    'lancaster': np.array([109, 88, 104, 150, 136, 147, 163]),
    'philadelphia': np.array([120, 100, 110, 140, 130, 145, 155]),
    'richmond': np.array([85, 95, 105, 120, 110, 125, 135])
}

print("=" * 70)
print("TESTE: OptimizationMethodsGlobal - _repair_solution()")
print("=" * 70)

# Teste 1: Verificar se a geração de soluções viáveis funciona
print("\n[TESTE 1] Gerando soluções viáveis...")
opt = OptimizationMethodsGlobal(STORES, forecasts, objective='O3_NS', seed=42)

feasible_count = 0
total_tests = 10

for i in range(total_tests):
    solution = opt._repair_solution(opt._generate_feasible_solution())
    J, X, PR = opt._split_solution(solution)
    
    # Calcular total de unidades
    total_units = 0
    for s, store in enumerate(STORES):
        start = s * 7
        daily_plans = []
        for d in range(7):
            daily_plans.append({
                'num_customers': forecasts[store][d],
                'J': int(J[start + d]),
                'X': int(X[start + d]),
                'PR': PR[start + d],
                'is_weekend': d >= 5
            })
        result = calculate_weekly_profit(daily_plans, store)
        total_units += result['total_units']
    
    is_feasible = total_units <= 10000
    feasible_count += is_feasible
    print(f"  Tentativa {i+1}: {total_units:,} unidades - {'✓ VIÁVEL' if is_feasible else '✗ INVIÁVEL'}")

print(f"\n  Taxa de viabilidade: {feasible_count}/{total_tests} ({100*feasible_count/total_tests:.1f}%)")

# Teste 2: Verificar NSGA2 com parametrização nova
print("\n[TESTE 2] Rodando NSGA2 com nova parametrização...")
opt2 = OptimizationMethodsGlobal(STORES, forecasts, objective='O3_NS', seed=123)
result = opt2.nsga2(population_size=60, generations=50)  # Reduzido para teste rápido

print(f"  Melhor valor: {result['best_value']:.2f}")
print(f"  Tamanho do Pareto front: {len(result.get('pareto_front', []))}")

if 'pareto_solutions' in result:
    print(f"\n  Soluções no Pareto front:")
    for i, sol in enumerate(result['pareto_solutions']):
        print(f"    {i+1}. Lucro: {sol['profit']:.0f}, HR: {sol['hr']:.0f}")

# Teste 3: Verificar planos gerados
print("\n[TESTE 3] Validando planos por loja...")
solution = result['solution']
J_all, X_all, PR_all = opt2._split_solution(solution)

for i, store in enumerate(STORES):
    start = i * 7
    J = J_all[start:start+7].astype(int)
    X = X_all[start:start+7].astype(int)
    PR = PR_all[start:start+7]
    
    daily_plans = []
    for d in range(7):
        daily_plans.append({
            'num_customers': forecasts[store][d],
            'J': int(J[d]),
            'X': int(X[d]),
            'PR': PR[d],
            'is_weekend': d >= 5
        })
    
    result_store = calculate_weekly_profit(daily_plans, store)
    total_attended = sum(min(7*X[d] + 6*J[d], forecasts[store][d]) for d in range(7))
    
    print(f"\n  {store.upper()}:")
    print(f"    Total clientes previstos: {sum(forecasts[store])}")
    print(f"    Total atendidos: {total_attended}")
    print(f"    Semana: Lucro = {result_store['weekly_profit']:.0f}, Unidades = {result_store['total_units']:.0f}, HR = {result_store['total_hr']:.0f}")
    print(f"    Juniores (J): {J}")
    print(f"    Experts (X):  {X}")

print("\n" + "=" * 70)
print("TESTES CONCLUÍDOS")
print("=" * 70)
