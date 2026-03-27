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


def _easter_sunday(year):
    """Computa a data da Páscoa (calendário gregoriano)."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return pd.Timestamp(year=year, month=month, day=day)


def _known_closed_day_mask(dates):
    """Mascara de dias de loja fechada por calendário conhecido."""
    dates = pd.to_datetime(dates)
    christmas = (dates.dt.month == 12) & (dates.dt.day == 25)
    easter = dates == dates.dt.year.map(_easter_sunday)
    return (christmas | easter).to_numpy(dtype=bool)


def _apply_closed_day_override(y_pred, test_dates):
    """Força previsão zero em dias de loja fechada conhecidos por calendário."""
    y_adj = np.asarray(y_pred, dtype=float).copy()
    closed_mask = _known_closed_day_mask(pd.Series(test_dates))
    y_adj[closed_mask] = 0.0
    return np.maximum(y_adj, 0)


def _build_feature_frame(df):
    feat = df.copy()

    if 'Date' in feat.columns:
        dt = pd.to_datetime(feat['Date'])
        feat['Year'] = dt.dt.year
        feat['WeekOfYear'] = dt.dt.isocalendar().week.astype(int)
        feat['DayOfMonth'] = dt.dt.day
        feat['Is_Christmas'] = ((dt.dt.month == 12) & (dt.dt.day == 25)).astype(int)
        feat['Is_Easter_Sunday'] = (dt == dt.dt.year.map(_easter_sunday)).astype(int)
        feat['Is_Known_Closed_Day'] = _known_closed_day_mask(dt).astype(int)

    if 'TouristEvent' in feat.columns and 'Is_Tourist_Event' not in feat.columns:
        feat['Is_Tourist_Event'] = feat['TouristEvent'].astype(str).str.lower().map({'yes': 1, 'no': 0}).fillna(0)

    if 'Num_Customers' in feat.columns:
        feat['Lag_Customers_1'] = feat['Num_Customers'].shift(1)
        feat['Lag_Customers_7'] = feat['Num_Customers'].shift(7)
        feat['Lag_Customers_14'] = feat['Num_Customers'].shift(14)
        feat['Lag_Customers_28'] = feat['Num_Customers'].shift(28)
        feat['Rolling_Mean_14'] = feat['Num_Customers'].rolling(14).mean()
        feat['Rolling_Std_14'] = feat['Num_Customers'].rolling(14).std()

    for col in XGBOOST_FEATURES:
        if col not in feat.columns:
            feat[col] = 0

    X = feat[XGBOOST_FEATURES].copy()
    return X.ffill().bfill().fillna(0)


def _evaluate_store_fast(store_name, train_ratio=0.85):
    df = pd.read_csv(get_data_path(store_name, raw=False))
    y = df['Num_Customers'].values
    X = _build_feature_frame(df)

    split_idx = int(len(df) * train_ratio)
    y_train, y_test = y[:split_idx], y[split_idx:]
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]

    horizon = len(y_test)
    if horizon < 7:
        raise ValueError(f"Store {store_name} com holdout muito pequeno")

    results = []
    error_rows = []

    date_col = pd.to_datetime(df['Date']) if 'Date' in df.columns else pd.RangeIndex(start=0, stop=len(df))
    test_dates = date_col.iloc[split_idx:] if hasattr(date_col, 'iloc') else date_col[split_idx:]

    def add_errors(model_name, y_pred):
        for d, actual, pred in zip(test_dates, y_test, y_pred):
            err = actual - pred
            error_rows.append({
                'store': store_name,
                'model': model_name,
                'date': str(d),
                'actual': float(actual),
                'predicted': float(pred),
                'error': float(err),
                'abs_error': float(abs(err)),
            })

    naive = SeasonalNaiveForecaster(store_name, seasonal_period=7)
    naive.train(None, y_train)
    naive_pred = _apply_closed_day_override(naive.predict(n_periods=horizon), test_dates)
    naive_m = calculate_metrics(y_test, naive_pred)
    results.append({'method': 'seasonal_naive', **naive_m})
    add_errors('seasonal_naive', naive_pred)

    arima = ARIMAForecaster(store_name)
    arima.train(None, y_train)
    arima_pred = _apply_closed_day_override(arima.predict(n_periods=horizon), test_dates)
    arima_m = calculate_metrics(y_test, arima_pred)
    results.append({'method': 'arima', **arima_m})
    add_errors('arima', arima_pred)

    ets = ETSForecaster(store_name)
    ets.train(None, y_train)
    ets_pred = _apply_closed_day_override(ets.predict(n_periods=horizon), test_dates)
    ets_m = calculate_metrics(y_test, ets_pred)
    results.append({'method': 'ets', **ets_m})
    add_errors('ets', ets_pred)

    xgb = XGBoostForecaster(store_name)
    xgb.train(X_train, y_train)
    xgb_pred = _apply_closed_day_override(xgb.predict(X_test, n_periods=horizon), test_dates)
    xgb_m = calculate_metrics(y_test, xgb_pred)
    results.append({'method': 'xgboost', **xgb_m})
    add_errors('xgboost', xgb_pred)

    arimax = ARIMAXForecaster(store_name)
    arimax.train(X_train, y_train)
    arimax_pred = _apply_closed_day_override(arimax.predict(X_test, n_periods=horizon), test_dates)
    arimax_m = calculate_metrics(y_test, arimax_pred)
    results.append({'method': 'arimax', **arimax_m})
    add_errors('arimax', arimax_pred)

    return pd.DataFrame(results), pd.DataFrame(error_rows)


def evaluate_all_stores(train_ratio=0.85):
    """Executa comparação rápida para todas as lojas."""
    all_frames = []
    all_errors = []

    print("\n" + "=" * 70)
    print("FORECASTING RÁPIDO (SEM ROLLING WINDOW)")
    print("=" * 70)

    for store in STORES:
        print(f"\n[{store.upper()}] avaliando baseline + 4 modelos...")
        store_df, store_errors = _evaluate_store_fast(store, train_ratio=train_ratio)
        store_df.insert(0, 'store', store)
        all_frames.append(store_df)
        all_errors.append(store_errors)

        out_path = get_report_path(f'fast_eval_{store}.csv')
        store_df.to_csv(out_path, index=False)
        err_path = get_report_path(f'fast_eval_errors_{store}.csv')
        store_errors.sort_values('abs_error', ascending=False).to_csv(err_path, index=False)
        print(f"  [OK] salvo em: {out_path}")
        print(f"  [OK] erros salvos em: {err_path}")

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

    errors_df = pd.concat(all_errors, ignore_index=True)
    errors_path = get_report_path('fast_eval_errors_all.csv')
    errors_df.to_csv(errors_path, index=False)

    top_errors = (
        errors_df
        .sort_values('abs_error', ascending=False)
        .head(50)
        .reset_index(drop=True)
    )
    top_errors_path = get_report_path('fast_eval_top50_errors.csv')
    top_errors.to_csv(top_errors_path, index=False)

    top_errors_by_model = (
        errors_df
        .sort_values('abs_error', ascending=False)
        .groupby('model', as_index=False)
        .head(10)
        .reset_index(drop=True)
    )
    top_errors_by_model_path = get_report_path('fast_eval_top10_errors_by_model.csv')
    top_errors_by_model.to_csv(top_errors_by_model_path, index=False)

    print(f"\n[OK] comparação combinada: {combined_path}")
    print(f"[OK] resumo final: {summary_path}")
    print(f"[OK] erros completos: {errors_path}")
    print(f"[OK] top 50 erros: {top_errors_path}")
    print(f"[OK] top 10 por modelo: {top_errors_by_model_path}")

    return {
        'combined': combined,
        'summary': summary,
        'errors': errors_df,
        'top_errors': top_errors,
        'combined_path': str(combined_path),
        'summary_path': str(summary_path),
    }
