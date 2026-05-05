import pandas as pd
import matplotlib.pyplot as plt

# carregar tabela resumo (a que já geraste)
files = {
    "Philadelphia": "reports/tabela_philadelphia_FINAL.csv",
    "Lancaster": "reports/tabela_lancaster_FINAL.csv",
    "Baltimore": "reports/tabela_baltimore_FINAL.csv",
    "Richmond": "reports/tabela_richmond_FINAL.csv"
}

dfs = []

for loja, path in files.items():
    df = pd.read_csv(path)
    df["Loja"] = loja
    dfs.append(df)

df_all = pd.concat(dfs, ignore_index=True)

# agrupar (igual à tabela 21)
resumo = df_all.groupby(["Loja", "Cenário"])["Lucro"].sum().reset_index()

# pivot para gráfico
pivot = resumo.pivot(index="Loja", columns="Cenário", values="Lucro")

# plot
pivot.plot(kind="bar")

plt.title("Lucro Total por Cenário e Loja")
plt.xlabel("Loja")
plt.ylabel("Lucro (€)")
plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig("reports/grafico_lucro.png")
plt.show()