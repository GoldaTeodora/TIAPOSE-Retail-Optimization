import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import matplotlib.pyplot as plt

from core.config import FIGURES_PATH, get_report_path

def plot_residuals_summary():
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)
    stores = ['baltimore', 'lancaster', 'philadelphia', 'richmond']
    for store in stores:
        residual_path = get_report_path(f'{store}_residuals.csv')
        df = pd.read_csv(residual_path) if residual_path.exists() else None
        if df is None:
            continue
        plt.figure(figsize=(8, 4))
        plt.hist(df['residuals'], bins=20)
        plt.title(f'Resíduos - {store.capitalize()}')
        plt.tight_layout()
        plt.savefig(FIGURES_PATH / f'residuals_{store}.png')
        plt.close()

if __name__ == '__main__':
    plot_residuals_summary()
