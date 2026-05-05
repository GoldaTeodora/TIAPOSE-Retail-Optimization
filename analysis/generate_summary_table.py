import pandas as pd

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

# juntar tudo
df_all = pd.concat(dfs, ignore_index=True)

# agrupar
resumo = df_all.groupby(["Loja", "Cenário"]).agg({
    "Lucro": "sum",
    "J": "mean",
    "X": "mean",
    "PR": "mean"
}).reset_index()

# nomes finais
resumo.columns = ["Loja", "Cenário", "Lucro Total (€)", "J Médio", "X Médio", "PR Médio"]

# arredondar
resumo = resumo.round(2)

print("\n✅ TABELA 21 (RESUMO FINAL):\n")
print(resumo)