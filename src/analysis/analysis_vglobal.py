import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt

from core.config import FIGURES_PATH

def plot_global_comparison():
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)
    data = {
        'Baltimore': {'Naive': 0.5003, 'S_Naive': 0.3615, 'MA_7': 0.4030, 'ARIMA': 0.3001, 'ETS': 0.4827, 'XGBoost': 0.1412},
        'Lancaster': {'Naive': 0.5007, 'S_Naive': 0.3624, 'MA_7': 0.4033, 'ARIMA': 0.3458, 'ETS': 0.5394, 'XGBoost': 0.1801},
        'Philadelphia': {'Naive': 0.4990, 'S_Naive': 0.3533, 'MA_7': 0.3989, 'ARIMA': 0.2116, 'ETS': 0.4350, 'XGBoost': 0.1319},
        'Richmond': {'Naive': 0.4995, 'S_Naive': 0.3613, 'MA_7': 0.4024, 'ARIMA': 0.2608, 'ETS': 0.4611, 'XGBoost': 0.1512}
    }
    for store, results in data.items():
        plt.figure(figsize=(8, 4))
        plt.bar(results.keys(), results.values())
        plt.title(f'Comparação Global de NMAE - {store}')
        plt.ylabel('NMAE')
        plt.tight_layout()
        plt.savefig(FIGURES_PATH / f'global_nmae_{store.lower()}.png')
        plt.close()
        print(f"[OK] Gráfico global NMAE criado para {store}")

if __name__ == '__main__':
    plot_global_comparison()
