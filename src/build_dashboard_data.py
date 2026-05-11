from pathlib import Path
import pandas as pd

REPORT_DIR = Path(__file__).resolve().parent.parent / "reports"

# ==========================================
# WEEKLY PLAN
# ==========================================
files = list(REPORT_DIR.glob("plan_*_20runs.csv"))

dfs = []

for f in files:
    df = pd.read_csv(f)

    # Nome: plan_baltimore_O1_20runs.csv
    parts = f.stem.split("_")

    loja = parts[1]
    cenario = parts[2]

    df["Loja"] = loja
    df["Cenario"] = cenario

    # criar colunas compatíveis
    if "Profit" in df.columns:
        df["Lucro"] = df["Profit"]

    if "Dia" in df.columns:
        df["Data"] = df["Dia"]

    dfs.append(df)

if dfs:
    final_weekly = pd.concat(dfs, ignore_index=True)
    final_weekly.to_csv(REPORT_DIR / "weekly_plan_scenarios.csv", index=False)

# ==========================================
# SUMMARY
# ==========================================
files = list(REPORT_DIR.glob("summary_*_20runs.csv"))

summary_rows = []

for f in files:
    df = pd.read_csv(f)

    parts = f.stem.split("_")

    loja = parts[1]
    cenario = parts[2]

    if "mean_profit" in df.columns:
        best = df.loc[df["mean_profit"].idxmax()]

        summary_rows.append({
            "Loja": loja,
            "Cenario": cenario,
            "Lucro": best["mean_profit"]
        })

if summary_rows:
    final_summary = pd.DataFrame(summary_rows)
    final_summary.to_csv(REPORT_DIR / "summary_optimization.csv", index=False)

print("✅ Ficheiros do dashboard criados!")