import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


REPORTS_PATH = Path("reports")


# 📊 1. Comparação de algoritmos (por loja)
def plot_algorithm_comparison(store_name):
    file_path = REPORTS_PATH / f"results_{store_name}.csv"

    if not file_path.exists():
        print(f"[ERRO] Ficheiro não encontrado: {file_path}")
        return

    df = pd.read_csv(file_path)

    plt.figure()

    for scenario in df['scenario'].unique():
        subset = df[df['scenario'] == scenario]

        plt.bar(
            subset['method'] + "_" + scenario,
            subset['profit']
        )

    plt.xlabel("Algoritmo")
    plt.ylabel("Lucro")
    plt.title(f"Comparação de Algoritmos - {store_name}")
    plt.xticks(rotation=45)

    plt.tight_layout()

    output_file = REPORTS_PATH / f"plot_algorithms_{store_name}.png"
    plt.savefig(output_file)

    print(f"[OK] Gráfico salvo: {output_file}")

    plt.show()


# 📊 2. Antes vs Depois (tuning)
def plot_before_after():
    algorithms = ['HC', 'SA', 'GA', 'PSO']

    # 👉 usa valores reais (ou médios)
    before = [-2703, -458, 2041, 2176]
    after = [2305, 1297, 2100, 3000]

    x = range(len(algorithms))

    plt.figure()

    width = 0.4
    plt.bar([i - width/2 for i in x], before, width=width, label="Antes")
    plt.bar([i + width/2 for i in x], after, width=width, label="Depois")

    plt.xticks(x, algorithms)
    plt.xlabel("Algoritmos")
    plt.ylabel("Lucro")
    plt.title("Antes vs Depois do Tuning")
    plt.legend()

    plt.tight_layout()

    output_file = REPORTS_PATH / "plot_before_after.png"
    plt.savefig(output_file)

    print(f"[OK] Gráfico salvo: {output_file}")

    plt.show()


# 📊 3. Convergência (opcional - nível 20)
def plot_convergence(history, method_name="Algoritmo"):
    plt.figure()

    plt.plot(history)

    plt.xlabel("Iterações")
    plt.ylabel("Melhor lucro")
    plt.title(f"Convergência - {method_name}")

    plt.tight_layout()
    plt.show()


# ▶️ MAIN
if __name__ == "__main__":

    stores = ["baltimore", "lancaster", "philadelphia", "richmond"]

    print("\n=== GERANDO GRÁFICOS ===\n")

    for store in stores:
        plot_algorithm_comparison(store)

    plot_before_after()