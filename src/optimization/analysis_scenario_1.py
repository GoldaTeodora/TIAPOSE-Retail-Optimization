import pandas as pd

# Ler resultados
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]

df = pd.read_csv(BASE_DIR / "scenario_1_results.csv")

# Criar summary
summary = df.groupby("method").agg(
    avg_value=("value", "mean"),
    best_value=("value", "max"),
    avg_time=("time", "mean"),
    avg_iterations=("iterations", "mean")
).reset_index()

# Ordenar por desempenho
summary = summary.sort_values("avg_value", ascending=False)

# Mostrar
print("\n=== SUMMARY CENÁRIO 1 ===")
print(summary)

# Guardar para relatório
summary.to_csv("scenario_1_summary.csv", index=False)