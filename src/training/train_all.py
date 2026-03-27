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
    X = df[XGBOOST_FEATURES].fillna(0).copy()

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
