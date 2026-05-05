import pandas as pd
import ast

# =============================
# Função para converter listas
# =============================
def safe_eval(x):
    if isinstance(x, list):
        return x
    return ast.literal_eval(x)

# =============================
# LOJAS DO PROJETO
# =============================
lojas = ["philadelphia", "lancaster", "baltimore", "richmond"]

# =============================
# LOOP PARA TODAS AS LOJAS
# =============================
for loja in lojas:
    print(f"\n📍 A processar loja: {loja.upper()}")

    files = {
        "O1": f"reports/rolling_optimum_{loja}_O1.csv",
        "O2": f"reports/rolling_optimum_{loja}_O2.csv",
        "O3": f"reports/rolling_optimum_{loja}_O3.csv"
    }

    tabelas = []

    # =========================
    # EXPANDIR DADOS
    # =========================
    for scenario, path in files.items():
        df = pd.read_csv(path)

        for col in ["J", "X", "PR", "clientes_previstos"]:
            df[col] = df[col].apply(safe_eval)

        for _, row in df.iterrows():
            base_date = pd.to_datetime(row["start_date"])
            dias = len(row["J"])

            for i in range(dias):
                tabelas.append({
                    "Data": (base_date + pd.Timedelta(days=i)).date(),
                    "Cenário": scenario,
                    "Clientes": round(row["clientes_previstos"][i]),
                    "J": int(row["J"][i]),
                    "X": int(row["X"][i]),
                    "PR": round(row["PR"][i], 2),
                    "Lucro": round(row["lucro_otimo"] / dias, 2)
                })

    # =========================
    # DATAFRAME FINAL
    # =========================
    final = pd.DataFrame(tabelas)

    final["Data"] = pd.to_datetime(final["Data"]).dt.date

    # =========================
    # REMOVER DUPLICADOS
    # =========================
    final = (
        final
        .sort_values("Lucro", ascending=False)
        .groupby(["Data", "Cenário"], as_index=False)
        .first()
    )

    # =========================
    # ORDENAR
    # =========================
    final = final.sort_values(["Cenário", "Data"]).reset_index(drop=True)

    # =========================
    # GUARDAR
    # =========================
    output_path = f"reports/tabela_{loja}_FINAL.csv"
    final.to_csv(output_path, index=False)

    print(f"✅ Guardado: {output_path}")

print("\n🎉 Todas as tabelas foram geradas!")