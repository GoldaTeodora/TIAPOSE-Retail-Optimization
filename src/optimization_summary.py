import pandas as pd
import os

# carregar resultados da otimização
df = pd.read_csv('reports/weekly_plan_scenarios.csv')

# -----------------------------
# AGRUPAR
# -----------------------------
summary = (
    df
    .groupby(['Loja', 'Cenario'])
    .agg({
        'Lucro': 'sum',
        'J': 'mean',
        'X': 'mean',
        'PR': 'mean'
    })
    .reset_index()
)

# -----------------------------
# ARREDONDAR
# -----------------------------
summary['J'] = summary['J'].round(2)
summary['X'] = summary['X'].round(2)
summary['PR'] = summary['PR'].round(2)

# -----------------------------
# ORDENAR
# -----------------------------
summary = summary.sort_values(by='Lucro', ascending=False)

# -----------------------------
# GUARDAR
# -----------------------------
os.makedirs('reports', exist_ok=True)
summary.to_csv('reports/summary_optimization.csv', index=False)

# -----------------------------
# PRINT
# -----------------------------
print("\n--- RESUMO FINAL DA OTIMIZAÇÃO ---")
print(summary.to_string(index=False))