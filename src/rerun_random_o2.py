"""
Re-corre apenas o Random Search para O2 global com n_iter=300 (corrigido).
Atualiza os ficheiros de resultados sem tocar nos outros algoritmos.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

from core.config import STORES
from optimization.methods_global import OptimizationMethodsGlobal

REPORTS = Path("reports")
MAX_RUNS = 100
CHECKPOINTS = [50, 100]


def load_forecasts():
    path = REPORTS / "future_forecast_7_days.csv"
    df = pd.read_csv(path)
    df["Loja"] = df["Loja"].str.lower()
    forecasts = {}
    for store in STORES:
        rows = df[df["Loja"] == store].sort_values("Data")
        forecasts[store] = rows["Clientes_Previstos"].values[:7].astype(int)
    return forecasts


def rerun_random_o2(forecasts, max_runs=100, checkpoints=(50, 100)):
    print("\n" + "="*60)
    print(" RE-RUN: Random Search O2 (n_iter=300 corrigido)")
    print("="*60)

    values = []

    for run_idx in range(max_runs):
        opt = OptimizationMethodsGlobal(
            stores=STORES,
            forecasts=forecasts,
            objective="O2",
            seed=run_idx,
        )
        res = opt.optimize("random")
        val = res["best_value"]
        values.append(val)

        if (run_idx + 1) % 10 == 0:
            print(f"  Run {run_idx+1}/{max_runs} | valor atual: {val:.2f} | média: {np.mean(values):.2f}")

        if (run_idx + 1) in checkpoints:
            n = run_idx + 1
            _update_checkpoint(values[:n], n)
            print(f"  [OK] checkpoint {n} runs actualizado")

    print(f"\nRandom Search O2 concluído.")
    print(f"  Média final ({max_runs} runs): {np.mean(values):.2f}")
    print(f"  Melhor: {np.max(values):.2f} | Pior: {np.min(values):.2f}")


def _update_checkpoint(values, n_runs):
    new_row = {
        "objective":    "O2",
        "algorithm":    "random",
        "mean_profit":  np.mean(values),
        "std_profit":   np.std(values),
        "best_profit":  np.max(values),
        "worst_profit": np.min(values),
    }

    # Actualizar global_comparison_O2_{n}runs.csv
    comp_path = REPORTS / f"global_comparison_O2_{n_runs}runs.csv"
    if comp_path.exists():
        df = pd.read_csv(comp_path)
        df = df[df["algorithm"] != "random"]  # remover linha antiga
    else:
        df = pd.DataFrame()

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df = df.sort_values("mean_profit", ascending=False)
    df.to_csv(comp_path, index=False)

    # Actualizar global_summary_O2_{n}runs.csv
    summary_path = REPORTS / f"global_summary_O2_{n_runs}runs.csv"
    if summary_path.exists():
        ds = pd.read_csv(summary_path)
        ds = ds[ds["algorithm"] != "random"]
    else:
        ds = pd.DataFrame()

    summary_row = {
        "algorithm":   "random",
        "mean_profit": np.mean(values),
        "std_profit":  np.std(values),
    }
    ds = pd.concat([ds, pd.DataFrame([summary_row])], ignore_index=True)
    ds = ds.sort_values("mean_profit", ascending=False)
    ds.to_csv(summary_path, index=False)


if __name__ == "__main__":
    t0 = time.time()
    forecasts = load_forecasts()
    print("Previsões carregadas:")
    for s, f in forecasts.items():
        print(f"  {s}: {f}")

    rerun_random_o2(forecasts, max_runs=MAX_RUNS, checkpoints=CHECKPOINTS)

    elapsed = time.time() - t0
    print(f"\nConcluído em {elapsed:.1f} segundos")
    print("Ficheiros actualizados:")
    for n in CHECKPOINTS:
        print(f"  reports/global_comparison_O2_{n}runs.csv")
        print(f"  reports/global_summary_O2_{n}runs.csv")
