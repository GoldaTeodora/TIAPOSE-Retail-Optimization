"""
Modelo ARIMAX (ARIMA com variáveis exógenas).
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pickle
from pmdarima import auto_arima
from forecasting.base import BaseForecaster


class ARIMAXForecaster(BaseForecaster):
    """ARIMAX com seleção automática de ordem e sazonalidade semanal."""

    def train(self, X_train, y_train):
        """Treina ARIMAX.

        Args:
            X_train: Features exógenas de treino
            y_train: Série alvo de treino
        """
        if X_train is None:
            raise ValueError("X_train não pode ser None para ARIMAX")

        y = y_train.values if hasattr(y_train, 'values') else np.asarray(y_train)
        X = X_train.values if hasattr(X_train, 'values') else np.asarray(X_train)

        self.model = auto_arima(
            y,
            exogenous=X,
            seasonal=True,
            m=7,
            max_p=3,
            max_q=3,
            max_d=2,
            max_P=2,
            max_Q=2,
            max_D=1,
            stepwise=True,
            suppress_warnings=True,
            error_action='ignore',
            trace=False
        )
        self.is_trained = True

    def predict(self, X_test=None, n_periods=7):
        """Prevê os próximos n_periods com exógenas.

        Args:
            X_test: Features exógenas para horizonte futuro
            n_periods: Número de dias a prever
        """
        if not self.is_trained:
            raise RuntimeError("Modelo não foi treinado")
        if X_test is None:
            raise ValueError("X_test não pode ser None para ARIMAX")

        X = X_test.values if hasattr(X_test, 'values') else np.asarray(X_test)
        forecast = self.model.predict(n_periods=n_periods, exogenous=X)
        return np.maximum(forecast, 0)

    def save(self, path):
        """Salva modelo ARIMAX em pickle."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)

    def load(self, path):
        """Carrega modelo ARIMAX de pickle."""
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
        self.is_trained = True
