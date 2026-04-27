import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_residuals_summary():
    os.makedirs('reports/figures', exist_ok=True)
    stores = ['baltimore', 'lancaster', 'philadelphia', 'richmond']
    for store in stores:
        df = pd.read_csv(f'reports/{store}_residuals.csv') if os.path.exists(f'reports/{store}_residuals.csv') else None
        if df is None:
            continue
        plt.figure(figsize=(8, 4))
        plt.hist(df['residuals'], bins=20)
        plt.title(f'Resíduos - {store.capitalize()}')
        plt.tight_layout()
        plt.savefig(f'reports/figures/residuals_{store}.png')
        plt.close()

if __name__ == '__main__':
    plot_residuals_summary()
