import pandas as pd
import matplotlib.pyplot as plt
import os


def plot_ets_detailed():
    stores = ['baltimore', 'lancaster', 'philadelphia', 'richmond']

    os.makedirs('reports/figures', exist_ok=True)

    for store in stores:
        df = pd.read_csv(f'reports/ets_{store}.csv', index_col=0)

        # métricas principais
        metrics = ['MAE', 'RMSE', 'NMAE']
        values = [df.loc[m].values[0] for m in metrics]

        plt.figure(figsize=(6, 4))
        plt.bar(metrics, values)

        plt.title(f'ETS - Métricas ({store.capitalize()})')
        plt.ylabel('Erro')

        # valores nas barras
        for i, v in enumerate(values):
            plt.text(i, v, f"{v:.3f}", ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(f'reports/figures/ets_detailed_{store}.png')
        plt.close()

        print(f"[OK] Gráfico detalhado ETS criado para {store}")


if __name__ == "__main__":
    plot_ets_detailed()