import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import matplotlib.pyplot as plt

from core.config import FIGURES_PATH, get_report_path

def plot_arima_detailed():
    stores = ['baltimore', 'lancaster', 'philadelphia', 'richmond']
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)
    for store in stores:
        df = pd.read_csv(get_report_path(f'arima_{store}.csv'), index_col=0)
        metrics = ['MAE', 'RMSE', 'NMAE']
        values = [df.loc[m].values[0] for m in metrics]
        plt.figure(figsize=(6, 4))
        plt.bar(metrics, values)
        plt.title(f'ARIMA - Métricas ({store.capitalize()})')
        plt.ylabel('Erro')
        for i, v in enumerate(values):
            plt.text(i, v, f"{v:.3f}", ha='center', va='bottom')
        plt.tight_layout()
        plt.savefig(FIGURES_PATH / f'arima_detailed_{store}.png')
        plt.close()
        print(f"[OK] Gráfico detalhado ARIMA criado para {store}")

if __name__ == "__main__":
    plot_arima_detailed()
