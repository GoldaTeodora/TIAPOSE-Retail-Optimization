import pandas as pd
import matplotlib.pyplot as plt
import os

stores = ['baltimore', 'lancaster', 'philadelphia', 'richmond']

os.makedirs('reports/figures', exist_ok=True)

for store in stores:
    df = pd.read_csv(f'reports/baseline_{store}.csv', index_col=0)

    plt.figure()
    df['MAE'].plot(kind='bar')

    plt.title(f'Baseline Comparison (MAE) - {store.capitalize()}')
    plt.ylabel('MAE')
    plt.xlabel('Model')
    plt.xticks(rotation=0)

    plt.tight_layout()
    plt.savefig(f'reports/figures/baseline_mae_{store}.png')
    plt.close()

print("✔ Gráficos baseline gerados com sucesso!")