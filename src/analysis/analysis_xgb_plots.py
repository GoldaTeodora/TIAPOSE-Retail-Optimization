import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_xgboost_detailed():
    stores = ['baltimore', 'lancaster', 'philadelphia', 'richmond']
    os.makedirs('reports/figures', exist_ok=True)
    for store in stores:
        df = pd.read_csv(f'reports/xgb_{store}.csv', index_col=0)
        metrics = ['MAE', 'RMSE', 'NMAE']
        values = [df.loc[m].values[0] for m in metrics]
        plt.figure(figsize=(6, 4))
        plt.bar(metrics, values)
        plt.title(f'XGBoost - Métricas ({store.capitalize()})')
        plt.ylabel('Erro')
        for i, v in enumerate(values):
            plt.text(i, v, f"{v:.3f}", ha='center', va='bottom')
        plt.tight_layout()
        plt.savefig(f'reports/figures/xgb_detailed_{store}.png')
        plt.close()
        print(f"[OK] Gráfico detalhado XGBoost criado para {store}")

if __name__ == "__main__":
    plot_xgboost_detailed()
