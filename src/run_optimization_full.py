"""
Pipeline completo de otimização com múltiplos runs.
Executa O1 (por loja) + O2 + O3_WEIGHTED + O3_NS (global).
Guarda checkpoints a 50 e 100 runs.

Uso:
  python src/run_optimization_full.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

from core.config import STORES
from optimization.optimize_profit import (
    optimize_store_profit,
    optimize_global_profit,
    compute_daily_metrics,
)

REPORTS = Path("reports")
REPORTS.mkdir(exist_ok=True)

CHECKPOINTS = [50, 100]


# ─────────────────────────────────────────────
# FORECASTS
# ─────────────────────────────────────────────
def load_forecasts():
    path = REPORTS / "future_forecast_7_days.csv"
    df = pd.read_csv(path)
    df["Loja"] = df["Loja"].str.lower()
    forecasts = {}
    for store in STORES:
        rows = df[df["Loja"] == store].sort_values("Data")
        forecasts[store] = (
            rows["Clientes_Previstos"].values[:7].astype(int)
        )
    return forecasts


# ─────────────────────────────────────────────
# O1 — por loja, com checkpoint
# ─────────────────────────────────────────────
def run_o1_all_stores(forecasts, max_runs=100, checkpoints=(50, 100)):
    """Corre O1 para todas as lojas com checkpoint a cada N runs."""
    from optimization.methods import OptimizationMethods

    PR_scenarios = {
        "coarse": np.array([0, 0.1, 0.2, 0.3]),
        "fine":   np.arange(0, 0.31, 0.05),
    }
    methods = [
        "random", "hill_climbing", "simulated_annealing",
        "genetic", "pso", "de",
    ]

    for store in STORES:
        print(f"\n{'='*60}")
        print(f" O1 — {store.upper()}")
        print(f"{'='*60}")
        forecast = forecasts[store]

        # acumula resultados por método/pr_type
        all_values = {
            (m, pr): [] for m in methods for pr in PR_scenarios
        }
        all_histories = {m: [] for m in methods}
        best_global = {"value": -np.inf, "solution": None, "method": None}

        for run_idx in range(max_runs):
            for pr_name, pr_vals in PR_scenarios.items():
                for method in methods:
                    opt = OptimizationMethods(
                        store, forecast, objective="O1",
                        method=method, seed=run_idx,
                    )
                    opt.PR_values = pr_vals
                    res = opt.optimize()
                    val = res["best_value"]
                    all_values[(method, pr_name)].append(val)

                    if pr_name == "coarse" and res.get("history"):
                        all_histories[method].append(res["history"])

                    if val > best_global["value"]:
                        best_global["value"] = val
                        best_global["solution"] = res["solution"].copy()
                        best_global["method"] = method

            if (run_idx + 1) in checkpoints:
                n = run_idx + 1
                _save_o1_checkpoint(
                    store, all_values, all_histories,
                    best_global, forecast, n, PR_scenarios, methods,
                )
                print(f"  [OK] checkpoint {n} runs guardado ({store})")

        print(f"  Melhor método: {best_global['method']} | valor: {best_global['value']:.2f}")


def _save_o1_checkpoint(
    store, all_values, all_histories,
    best_global, forecast, n_runs,
    PR_scenarios, methods,
):
    rows = []
    for pr_name in PR_scenarios:
        for method in methods:
            vals = all_values[(method, pr_name)][:n_runs]
            if not vals:
                continue
            rows.append({
                "store": store, "objective": "O1",
                "pr_type": pr_name, "algorithm": method,
                "mean_profit": np.mean(vals),
                "std_profit":  np.std(vals),
            })

    df_comp = pd.DataFrame(rows)
    df_comp.to_csv(REPORTS / f"comparison_{store}_O1_{n_runs}runs.csv", index=False)

    # summary (média de coarse+fine)
    summary = (
        df_comp.groupby("algorithm")["mean_profit"]
        .mean()
        .reset_index()
        .rename(columns={"mean_profit": "mean_profit"})
        .sort_values("mean_profit", ascending=False)
    )
    summary["std_profit"] = (
        df_comp.groupby("algorithm")["mean_profit"]
        .std()
        .reindex(summary["algorithm"])
        .values
    )
    summary.to_csv(REPORTS / f"summary_{store}_O1_{n_runs}runs.csv", index=False)

    # summary_pr
    summary_pr = (
        df_comp.groupby(["algorithm", "pr_type"])["mean_profit"]
        .mean()
        .reset_index()
    )
    summary_pr.to_csv(REPORTS / f"summary_pr_{store}_O1_{n_runs}runs.csv", index=False)

    # plano ótimo
    if best_global["solution"] is not None:
        sol = best_global["solution"]
        J  = np.round(sol[0:7]).astype(int)
        X  = np.round(sol[7:14]).astype(int)
        PR = sol[14:21]
        plan_df = compute_daily_metrics(store, forecast, J, X, PR)
        plan_df.to_csv(REPORTS / f"plan_{store}_O1_{n_runs}runs.csv", index=False)

    # convergência (coarse, média sobre runs)
    _save_convergence_png(store, "O1", all_histories, n_runs)


def _save_convergence_png(store, objective, all_histories, n_runs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {
        "random": "#636EFA",
        "hill_climbing": "#EF553B",
        "simulated_annealing": "#00CC96",
        "genetic": "#AB63FA",
        "pso": "#FFA15A",
        "de": "#19D3F3",
    }
    labels = {
        "random": "Random Search",
        "hill_climbing": "Hill Climbing",
        "simulated_annealing": "Simulated Annealing",
        "genetic": "Genetic Algorithm",
        "pso": "PSO",
        "de": "Differential Evolution",
    }

    for method, histories in all_histories.items():
        if not histories:
            continue
        min_len = min(len(h) for h in histories)
        trimmed = [h[:min_len] for h in histories]
        mean_h = np.mean(trimmed, axis=0)
        ax.plot(
            mean_h,
            label=labels.get(method, method),
            color=colors.get(method, None),
            linewidth=2,
        )

    ax.set_xlabel("Iteração", fontsize=12)
    ax.set_ylabel("Melhor Lucro Encontrado ($)", fontsize=12)
    ax.set_title(
        f"Convergência — {store.capitalize()} | O1 | {n_runs} runs",
        fontsize=13, fontweight="bold",
    )
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(
        REPORTS / f"convergence_{store}_{objective}_{n_runs}runs_coarse.png",
        dpi=120,
    )
    plt.close(fig)


# ─────────────────────────────────────────────
# GLOBAL — O2 / O3_WEIGHTED / O3_NS
# ─────────────────────────────────────────────
def run_global_objectives(forecasts, max_runs=100, checkpoints=(50, 100)):
    """Corre O2, O3_WEIGHTED e O3_NS com checkpoint."""
    from optimization.methods_global import OptimizationMethodsGlobal
    from core.config import STORE_PARAMS

    objectives_config = {
        "O2":          ["random", "hill_climbing", "simulated_annealing",
                        "genetic", "pso", "de"],
        "O3_WEIGHTED": ["genetic", "pso", "de"],
        "O3_NS":       ["nsga2"],
    }

    for objective, methods in objectives_config.items():
        print(f"\n{'='*60}")
        print(f" GLOBAL — {objective}")
        print(f"{'='*60}")

        all_values   = {m: [] for m in methods}
        all_histories = {m: [] for m in methods}
        best_global = {"value": -np.inf, "result": None, "optimizer": None, "method": None}

        for run_idx in range(max_runs):
            for method in methods:
                opt = OptimizationMethodsGlobal(
                    stores=STORES,
                    forecasts=forecasts,
                    objective=objective,
                    seed=run_idx,
                )
                res = opt.optimize(method)
                val = res["best_value"]
                all_values[method].append(val)

                if res.get("history"):
                    all_histories[method].append(res["history"])

                if val > best_global["value"]:
                    best_global["value"] = val
                    best_global["result"] = res
                    best_global["optimizer"] = opt
                    best_global["method"] = method

            if (run_idx + 1) in checkpoints:
                n = run_idx + 1
                _save_global_checkpoint(
                    objective, methods, all_values,
                    all_histories, best_global, forecasts, n,
                )
                print(f"  [OK] checkpoint {n} runs guardado ({objective})")

        print(f"  Melhor método: {best_global['method']} | valor: {best_global['value']:.2f}")


def _save_global_checkpoint(
    objective, methods, all_values,
    all_histories, best_global, forecasts, n_runs,
):
    rows = []
    for method in methods:
        vals = all_values[method][:n_runs]
        if not vals:
            continue
        rows.append({
            "objective":    objective,
            "algorithm":    method,
            "mean_profit":  np.mean(vals),
            "std_profit":   np.std(vals),
            "best_profit":  np.max(vals),
            "worst_profit": np.min(vals),
        })

    df_comp = pd.DataFrame(rows).sort_values("mean_profit", ascending=False)
    df_comp.to_csv(
        REPORTS / f"global_comparison_{objective}_{n_runs}runs.csv",
        index=False,
    )

    summary = df_comp[["algorithm", "mean_profit", "std_profit"]].copy()
    summary.to_csv(
        REPORTS / f"global_summary_{objective}_{n_runs}runs.csv",
        index=False,
    )

    # planos por loja
    opt = best_global["optimizer"]
    res = best_global["result"]
    if opt is not None and res is not None:
        try:
            sol = opt._repair_solution(opt._fix_solution(res["solution"].copy()))
            J_all, X_all, PR_all = opt._split_solution(sol)

            for i, store in enumerate(STORES):
                start = i * 7
                J  = J_all[start:start+7].astype(int)
                X  = X_all[start:start+7].astype(int)
                PR = PR_all[start:start+7]
                plan_df = compute_daily_metrics(store, forecasts[store], J, X, PR)
                plan_df.to_csv(
                    REPORTS / f"global_plan_{store}_{objective}_{n_runs}runs.csv",
                    index=False,
                )
        except Exception as e:
            print(f"  [WARN] Não foi possível guardar planos: {e}")

        # Pareto front (O3_NS)
        if objective == "O3_NS" and "pareto_solutions" in res:
            pareto = res["pareto_solutions"]
            if pareto:
                pd.DataFrame([
                    {"profit": s["profit"], "hr": s["hr"]}
                    for s in pareto
                ]).to_csv(
                    REPORTS / f"pareto_front_{objective}_{n_runs}runs.csv",
                    index=False,
                )

    # convergência global
    _save_global_convergence_png(objective, all_histories, n_runs)


def _save_global_convergence_png(objective, all_histories, n_runs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {
        "random": "#636EFA", "hill_climbing": "#EF553B",
        "simulated_annealing": "#00CC96", "genetic": "#AB63FA",
        "pso": "#FFA15A", "de": "#19D3F3", "nsga2": "#FF6692",
    }
    for method, histories in all_histories.items():
        if not histories:
            continue
        min_len = min(len(h) for h in histories)
        trimmed = [h[:min_len] for h in histories]
        mean_h = np.mean(trimmed, axis=0)
        ax.plot(mean_h, label=method.upper(),
                color=colors.get(method), linewidth=2)

    ax.set_xlabel("Iteração", fontsize=12)
    ax.set_ylabel("Melhor Valor", fontsize=12)
    ax.set_title(
        f"Convergência Global — {objective} | {n_runs} runs",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(
        REPORTS / f"global_convergence_{objective}_{n_runs}runs.png",
        dpi=120,
    )
    plt.close(fig)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    t0 = time.time()
    print("\n" + "="*60)
    print(" TIAPOSE — OTIMIZAÇÃO COMPLETA (50 + 100 runs)")
    print("="*60)

    forecasts = load_forecasts()
    print("\nPrevisões carregadas:")
    for s, f in forecasts.items():
        print(f"  {s}: {f}")

    # O1 — por loja
    print("\n\n>>> FASE 1: O1 (por loja)")
    run_o1_all_stores(forecasts, max_runs=100, checkpoints=(50, 100))

    # Global
    print("\n\n>>> FASE 2: GLOBAL (O2, O3_WEIGHTED, O3_NS)")
    run_global_objectives(forecasts, max_runs=100, checkpoints=(50, 100))

    elapsed = time.time() - t0
    print(f"\n\n{'='*60}")
    print(f" CONCLUÍDO em {elapsed/60:.1f} minutos")
    print(f"{'='*60}")
    print("\nFicheiros gerados em reports/:")
    for f in sorted(REPORTS.glob("*50runs*")) + sorted(REPORTS.glob("*100runs*")):
        print(f"  {f.name}")
