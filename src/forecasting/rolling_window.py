"""
Avaliação rápida de forecasting sem rolling window.

Métodos avaliados:
- seasonal_naive (baseline)
- arima
- ets
- xgboost
- arimax
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from forecasting.naive_model import SeasonalNaiveForecaster
from forecasting.arima_model import ARIMAForecaster
from forecasting.ets_model import ETSForecaster
from forecasting.ml_model import XGBoostForecaster
from forecasting.arimax_model import ARIMAXForecaster
from core.metrics import calculate_metrics
from core.config import STORES, get_data_path, get_report_path, XGBOOST_FEATURES


def _evaluate_store_fast(store_name, train_ratio=0.85):
    df = pd.read_csv(get_data_path(store_name, raw=False))
    y = df['Num_Customers'].values
    X = df[XGBOOST_FEATURES].fillna(0).copy()

    split_idx = int(len(df) * train_ratio)
    y_train, y_test = y[:split_idx], y[split_idx:]
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]

    horizon = len(y_test)
    if horizon < 7:
        raise ValueError(f"Store {store_name} com holdout muito pequeno")

    results = []

    naive = SeasonalNaiveForecaster(store_name, seasonal_period=7)
    naive.train(None, y_train)
    naive_pred = naive.predict(n_periods=horizon)
    naive_m = calculate_metrics(y_test, naive_pred)
    results.append({'method': 'seasonal_naive', **naive_m})

    arima = ARIMAForecaster(store_name)
    arima.train(None, y_train)
    arima_pred = arima.predict(n_periods=horizon)
    arima_m = calculate_metrics(y_test, arima_pred)
    results.append({'method': 'arima', **arima_m})

    ets = ETSForecaster(store_name)
    ets.train(None, y_train)
    ets_pred = ets.predict(n_periods=horizon)
    ets_m = calculate_metrics(y_test, ets_pred)
    results.append({'method': 'ets', **ets_m})

    xgb = XGBoostForecaster(store_name)
    xgb.train(X_train, y_train)
    xgb_pred = xgb.predict(X_test, n_periods=horizon)
    xgb_m = calculate_metrics(y_test, xgb_pred)
    results.append({'method': 'xgboost', **xgb_m})

    arimax = ARIMAXForecaster(store_name)
    arimax.train(X_train, y_train)
    arimax_pred = arimax.predict(X_test, n_periods=horizon)
    arimax_m = calculate_metrics(y_test, arimax_pred)
    results.append({'method': 'arimax', **arimax_m})

    return pd.DataFrame(results)


def evaluate_all_stores(train_ratio=0.85):
    """Executa comparação rápida para todas as lojas."""
    all_frames = []

    print("\n" + "=" * 70)
    print("FORECASTING RÁPIDO (SEM ROLLING WINDOW)")
    print("=" * 70)

    for store in STORES:
        print(f"\n[{store.upper()}] avaliando baseline + 4 modelos...")
        store_df = _evaluate_store_fast(store, train_ratio=train_ratio)
        store_df.insert(0, 'store', store)
        all_frames.append(store_df)

        out_path = get_report_path(f'fast_eval_{store}.csv')
        store_df.to_csv(out_path, index=False)
        print(f"  [OK] salvo em: {out_path}")

    combined = pd.concat(all_frames, ignore_index=True)
    combined_path = get_report_path('fast_eval_all_stores.csv')
    combined.to_csv(combined_path, index=False)

    summary = (
        combined
        .groupby('method', as_index=False)[['MAE', 'RMSE', 'MAPE', 'NMAE', 'R2']]
        .mean()
        .sort_values('R2', ascending=False)
        .reset_index(drop=True)
    )
    summary_path = get_report_path('fast_eval_summary.csv')
    summary.to_csv(summary_path, index=False)

    print(f"\n[OK] comparação combinada: {combined_path}")
    print(f"[OK] resumo final: {summary_path}")

    return {
        'combined': combined,
        'summary': summary,
        'combined_path': str(combined_path),
        'summary_path': str(summary_path),
    }
