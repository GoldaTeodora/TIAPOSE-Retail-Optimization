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
    from forecasting.advanced_features import AdvancedFeatureEngineer
    engineer = AdvancedFeatureEngineer(store_name, verbose=False)
    X, _ = engineer.process_features_complete(df, y=y, outlier_detection=False)
    # Garante que todas as features esperadas existem
    for col in XGBOOST_FEATURES:
        if col not in X.columns:
            X[col] = 0
    X = X[XGBOOST_FEATURES].ffill().bfill().fillna(0)

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
            print(f"[TRAIN] Treinando {model_name} para {store_name}...")
            model.train(x_for_train, y_train)
            model_path = get_model_path(store_name, model_name)
            print(f"[TRAIN] Salvando modelo {model_name} em: {model_path}")
            model.save(model_path)
            saved_models[model_name] = str(model_path)
            if verbose:
                print(f"   [OK] {model_name.upper()} salvo em: {model_path}")
        except Exception as e:
            print(f"[TRAIN] ERRO ao treinar/salvar {model_name} para {store_name}: {e}")
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

if __name__ == '__main__':
    print("[DEBUG] Script train_all.py iniciado!")
    train_all_models(verbose=True)
