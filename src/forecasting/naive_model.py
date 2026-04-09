"""
Modelo Seasonal Naive para baseline de previsão.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pickle
from forecasting.base import BaseForecaster


class SeasonalNaiveForecaster(BaseForecaster):
    """Baseline sazonal: repete os últimos m valores observados."""

    def __init__(self, store_name, seasonal_period=7):
        super().__init__(store_name)
        self.seasonal_period = seasonal_period
        self.last_season = None

    def train(self, X_train, y_train):
        """Armazena os últimos valores da sazonalidade.

        Args:
            X_train: Ignorado
            y_train: Série temporal de treino
        """
        y = y_train.values if hasattr(y_train, 'values') else np.asarray(y_train)
        if len(y) == 0:
            raise ValueError("y_train vazio para SeasonalNaive")

        m = min(self.seasonal_period, len(y))
        self.last_season = np.asarray(y[-m:], dtype=float)
        self.model = {'seasonal_period': self.seasonal_period}
        self.is_trained = True

    def predict(self, X_test=None, n_periods=7):
        """Repete o padrão sazonal para n períodos à frente."""
        if not self.is_trained or self.last_season is None:
            raise RuntimeError("Modelo não foi treinado")

        reps = int(np.ceil(n_periods / len(self.last_season)))
        forecast = np.tile(self.last_season, reps)[:n_periods]
        return np.maximum(forecast, 0)

    def save(self, path):
        """Salva modelo Seasonal Naive."""
        print(f"[NAIVE SAVE] Salvando modelo em: {path}")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(path, 'wb') as f:
                pickle.dump((self.seasonal_period, self.last_season), f)
            print(f"[NAIVE SAVE] Sucesso ao salvar: {path}")
        except Exception as e:
            print(f"[NAIVE SAVE] ERRO ao salvar {path}: {e}")

    def load(self, path):
        """Carrega modelo Seasonal Naive."""
        with open(path, 'rb') as f:
            self.seasonal_period, self.last_season = pickle.load(f)
        self.model = {'seasonal_period': self.seasonal_period}
        self.is_trained = True
