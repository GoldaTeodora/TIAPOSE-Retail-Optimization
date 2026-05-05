import pandas as pd
import matplotlib.pyplot as plt
import os


def plot_global_comparison():
    os.makedirs('reports/figures', exist_ok=True)

    # dados organizados
    data = {
        'Baltimore': {
            'Naive': 0.5003,
            'S_Naive': 0.3615,
            'MA_7': 0.4030,
            'ARIMA': 0.3001,
            'ETS': 0.4827,
            'XGBoost': 0.1412
        },
        'Lancaster': {
            'Naive': 0.5007,
            'S_Naive': 0.3624,
            'MA_7': 0.4033,
            'ARIMA': 0.3458,
            'ETS': 0.5394,
            'XGBoost': 0.1801
        },
        'Philadelphia': {
            'Naive': 0.4990,
            'S_Naive': 0.3533,
            'MA_7': 0.3989,
            'ARIMA': 0.2116,
            'ETS': 0.4350,
            'XGBoost': 0.1319
        },
        'Richmond': {
            'Naive': 0.4995,
            'S_Naive': 0.3613,
            'MA_7': 0.4024,
            'ARIMA': 0.2608,
            'ETS': 0.4611,
            'XGBoost': 0.1296
        }
    }

    df = pd.DataFrame(data)

    # transpor para facilitar o plot
    df = df.T

    # criar gráfico
    df.plot(kind='bar', figsize=(10, 6))

    plt.title('Comparação Global dos Modelos (NMAE)')
    plt.ylabel('NMAE')
    plt.xlabel('Lojas')
    plt.xticks(rotation=0)

    plt.legend(title='Modelos')
    plt.tight_layout()

    plt.savefig('reports/figures/global_model_comparison.png')
    plt.close()

    print("[OK] Gráfico global criado com sucesso!")


if __name__ == "__main__":
    plot_global_comparison()