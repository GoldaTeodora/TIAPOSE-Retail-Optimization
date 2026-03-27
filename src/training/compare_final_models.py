"""
Runner final para comparar os 5 métodos de forecasting.

Fluxo:
1. Treinar modelos por loja
2. Avaliar com rolling window
3. Consolidar resultados finais em CSV
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from core.config import STORES, get_report_path
from training.train_all import train_all_models
from forecasting.rolling_window import evaluate_all_stores


def summarize_global_results():
    rows = []

    for store in STORES:
        path = get_report_path(f'rolling_window_{store}.csv')
        df = pd.read_csv(path)
        df['store'] = store
        rows.append(df)

    all_df = pd.concat(rows, ignore_index=True)

    metric_cols = [
        'MAE_mean', 'RMSE_mean', 'MAPE_mean', 'NMAE_mean', 'R2_mean',
        'MAE_median', 'RMSE_median', 'MAPE_median', 'NMAE_median', 'R2_median'
    ]

    global_df = (
        all_df.groupby('method', as_index=False)[metric_cols]
        .mean(numeric_only=True)
        .sort_values('R2_mean', ascending=False)
        .reset_index(drop=True)
    )

    global_path = get_report_path('rolling_window_global_comparison.csv')
    global_df.to_csv(global_path, index=False)

    return global_df, global_path


def main(verbose=True):
    if verbose:
        print("\n" + "=" * 80)
        print("COMPARAÇÃO FINAL DOS 5 MÉTODOS")
        print("=" * 80)

    train_all_models(verbose=verbose)
    evaluate_all_stores()

    global_df, global_path = summarize_global_results()

    if verbose:
        print("\n" + "=" * 80)
        print("RANKING GLOBAL (por R2_mean)")
        print("=" * 80)
        print(global_df[['method', 'R2_mean', 'MAE_mean', 'RMSE_mean']].to_string(index=False))
        print(f"\n[OK] Comparação global salva em: {global_path}")

    return global_df


if __name__ == '__main__':
    main(verbose=True)
