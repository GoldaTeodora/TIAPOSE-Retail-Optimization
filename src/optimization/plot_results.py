import pandas as pd
import matplotlib.pyplot as plt

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("daily_results_all_scenarios.csv")

# =========================
# AGREGAR LUCRO
# =========================
summary = df.groupby(['store', 'scenario'])['profit'].sum().reset_index()

# Pivot para gráfico
pivot = summary.pivot(index='store', columns='scenario', values='profit')

print("\n=== LUCRO POR LOJA E CENÁRIO ===")
print(pivot)

# =========================
# PLOT
# =========================
pivot.plot(kind='bar')

plt.title("Lucro Total por Loja e Cenário")
plt.ylabel("Lucro (€)")
plt.xlabel("Loja")
plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig("profit_comparison.png")

plt.show()

import pandas as pd

df = pd.read_csv("daily_results_all_scenarios.csv")

summary = df.groupby(['store', 'scenario']).agg(
    total_profit=('profit', 'sum'),
    avg_J=('J', 'mean'),
    avg_X=('X', 'mean'),
    avg_PR=('PR', 'mean')
).reset_index()

# arredondar
summary['total_profit'] = summary['total_profit'].round(2)
summary['avg_J'] = summary['avg_J'].round(2)
summary['avg_X'] = summary['avg_X'].round(2)
summary['avg_PR'] = summary['avg_PR'].round(2)

print(summary)

summary.to_csv("summary_results.csv", index=False)