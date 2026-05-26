import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import matplotlib.pyplot as plt

from core.config import FIGURES_PATH, get_report_path

def plot_baseline_mae():
    stores = ['baltimore', 'lancaster', 'philadelphia', 'richmond']

    FIGURES_PATH.mkdir(parents=True, exist_ok=True)

    for store in stores:
        df = pd.read_csv(get_report_path(f'baseline_{store}.csv'), index_col=0)
        plt.figure()
        df['MAE'].plot(kind='bar')
        plt.title(f'Baseline Comparison (MAE) - {store.capitalize()}')
        plt.ylabel('MAE')
        plt.xlabel('Model')
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(FIGURES_PATH / f'baseline_mae_{store}.png')
        plt.close()

    print("✔ Gráficos baseline gerados com sucesso!")


if __name__ == '__main__':
    plot_baseline_mae()
