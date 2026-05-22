"""Gera gráficos do plano ótimo semanal para cada loja (O1, 100 runs)."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPORTS = Path("reports")
STORES = ["baltimore", "lancaster", "philadelphia", "richmond"]
DAY_LABELS = ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"]

W_S = {"baltimore": 700, "lancaster": 730, "philadelphia": 760, "richmond": 800}


def load_plan(store, n_runs=100):
    for n in [n_runs, 50, 20]:
        p = REPORTS / f"plan_{store}_O1_{n}runs.csv"
        if p.exists():
            return pd.read_csv(p), n
    return None, None


def plot_store_plan(store):
    df, n = load_plan(store)
    if df is None:
        print(f"  [SKIP] {store}: sem dados")
        return

    dias = DAY_LABELS[:len(df)]
    x = np.arange(len(df))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(
        f"Plano Ótimo — {store.capitalize()} | O1 | {n} runs",
        fontsize=14, fontweight="bold",
    )

    # --- esquerda: trabalhadores por dia ---
    bars_j = ax1.bar(x - width/2, df["Juniores"], width,
                     label="Juniores", color="#4C78A8", alpha=0.9)
    bars_x = ax1.bar(x + width/2, df["Experts"], width,
                     label="Experts", color="#F58518", alpha=0.9)

    # label numérico em cima de cada barra
    for bar in bars_j:
        h = bar.get_height()
        if h > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, h + 0.3,
                     str(int(h)), ha="center", va="bottom", fontsize=9)
    for bar in bars_x:
        h = bar.get_height()
        if h > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, h + 0.3,
                     str(int(h)), ha="center", va="bottom", fontsize=9)

    ax1.set_xticks(x)
    ax1.set_xticklabels(dias)
    ax1.set_ylabel("Nº Trabalhadores", fontsize=11)
    ax1.set_title("Alocação de Recursos Humanos", fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_ylim(0, max(df["Juniores"].max(), df["Experts"].max()) * 1.25 + 2)

    # --- direita: lucro diário e promoção ---
    colors = ["#E45756" if p > 0 else "#72B7B2"
              for p in df["Promocao"]]  # vermelho = com promoção
    bars_l = ax2.bar(x, df["Lucro"], color="#54A24B", alpha=0.85, label="Lucro diário ($)")

    for bar, lucro in zip(bars_l, df["Lucro"]):
        if lucro > 0:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                     f"${int(lucro)}", ha="center", va="bottom", fontsize=8)

    # promoção como linha secundária
    ax2b = ax2.twinx()
    ax2b.plot(x, df["Promocao"] * 100, "D--", color="#B279A2",
              linewidth=1.8, markersize=6, label="Promoção (%)")
    ax2b.set_ylabel("Promoção (%)", fontsize=10, color="#B279A2")
    ax2b.tick_params(axis="y", labelcolor="#B279A2")
    ax2b.set_ylim(-2, 35)

    ax2.set_xticks(x)
    ax2.set_xticklabels(dias)
    ax2.set_ylabel("Lucro ($)", fontsize=11)
    ax2.set_title("Lucro Diário e Promoções", fontsize=11)
    ax2.grid(axis="y", alpha=0.3)

    # legenda combinada
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2b.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, fontsize=9, loc="upper left")

    # anotação lucro líquido semanal
    lucro_liq = df["Lucro"].sum() - W_S[store]
    fig.text(0.99, 0.02,
             f"Lucro líquido semanal: ${lucro_liq:,.0f}  |  W_s = ${W_S[store]}",
             ha="right", fontsize=9, style="italic", color="gray")

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    out = REPORTS / f"plan_chart_{store}_O1_{n}runs.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  [OK] {out.name}")


if __name__ == "__main__":
    print("Gerando gráficos dos planos ótimos...\n")
    for store in STORES:
        plot_store_plan(store)
    print("\nConcluído.")
