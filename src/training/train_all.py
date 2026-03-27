"""
Treino essencial de modelos de forecasting (sem ensemble).

Modelos treinados por loja:
- ARIMA
- ETS
- XGBoost
- ARIMAX
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from core.config import STORES, get_data_path, get_model_path, XGBOOST_FEATURES
from forecasting.arima_model import ARIMAForecaster
from forecasting.ets_model import ETSForecaster
from forecasting.ml_model import XGBoostForecaster
from forecasting.arimax_model import ARIMAXForecaster


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
    return (christmas | easter).astype(int)


def _build_feature_frame(df):
    feat = df.copy()

    if 'Date' in feat.columns:
        dt = pd.to_datetime(feat['Date'])
        feat['Year'] = dt.dt.year
        feat['WeekOfYear'] = dt.dt.isocalendar().week.astype(int)
        feat['DayOfMonth'] = dt.dt.day
        feat['Is_Christmas'] = ((dt.dt.month == 12) & (dt.dt.day == 25)).astype(int)
        feat['Is_Easter_Sunday'] = (dt == dt.dt.year.map(_easter_sunday)).astype(int)
        feat['Is_Known_Closed_Day'] = _known_closed_day_mask(dt)

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

    return feat[XGBOOST_FEATURES].ffill().bfill().fillna(0)


def train_store_models(store_name, verbose=True):
    """Treina os 4 modelos principais para uma loja.

    Args:
        store_name: Nome da loja
        verbose: Imprimir progresso

    Returns:
        Dict com caminhos de modelos salvos
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"TREINAMENTO DE MODELOS: {store_name.upper()}")
        print(f"{'='*70}")

    path = get_data_path(store_name, raw=False)
    df = pd.read_csv(path)

    y = df['Num_Customers'].values
    X = _build_feature_frame(df)

    train_end = len(df) - 7
    y_train = y[:train_end]
    X_train = X.iloc[:train_end]

    if verbose:
        print(f"\nDados carregados: {len(df)} registros")
        print(f"Treino até índice: {train_end - 1}")

    saved_models = {}

    trainers = [
        ('arima', ARIMAForecaster(store_name), None),
        ('ets', ETSForecaster(store_name), None),
        ('xgboost', XGBoostForecaster(store_name), X_train),
        ('arimax', ARIMAXForecaster(store_name), X_train),
    ]

    for idx, (model_name, model, x_for_train) in enumerate(trainers, 1):
        if verbose:
            print(f"\n{idx}. Treinando {model_name.upper()}...")
        try:
            model.train(x_for_train, y_train)
            model_path = get_model_path(store_name, model_name)
            model.save(model_path)
            saved_models[model_name] = str(model_path)
            if verbose:
                print(f"   [OK] {model_name.upper()} salvo em: {model_path}")
        except Exception as e:
            if verbose:
                print(f"   [FAIL] Erro ao treinar {model_name.upper()}: {e}")

    return saved_models


def train_all_models(verbose=True):
    """Treina todos os modelos para todas as lojas."""
    if verbose:
        print("\n" + "=" * 70)
        print("FASE PRELIMINAR: TREINAMENTO DOS 4 MODELOS PRINCIPAIS")
        print("=" * 70)

    all_results = {}

    for store in STORES:
        try:
            all_results[store] = train_store_models(store, verbose=verbose)
        except Exception as e:
            if verbose:
                print(f"\n[FAIL] Erro ao treinar {store}: {e}")
            all_results[store] = {}

    if verbose:
        print("\n" + "=" * 70)
        print("RESUMO DO TREINAMENTO")
        print("=" * 70)
        for store, models in all_results.items():
            print(f"\n{store.upper()}:")
            for model_type, path in models.items():
                print(f"  [OK] {model_type:14s}: {path}")

    return all_results
