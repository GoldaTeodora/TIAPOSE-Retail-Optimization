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

        if len(y) < 40:
            self.model = auto_arima(
                y,
                exogenous=X,
                seasonal=False,
                suppress_warnings=True,
                error_action='ignore',
                trace=False,
            )
            self.is_trained = True
            return

        val_size = min(28, max(14, len(y) // 6))
        y_fit = y[:-val_size]
        y_val = y[-val_size:]
        X_fit = X[:-val_size]
        X_val = X[-val_size:]

        candidates = [
            {'seasonal': True, 'm': 7, 'max_p': 3, 'max_q': 3},
            {'seasonal': False, 'm': 1, 'max_p': 5, 'max_q': 5},
        ]

        best_cfg = None
        best_mae = np.inf

        for cfg in candidates:
            try:
                model = auto_arima(
                    y_fit,
                    exogenous=X_fit,
                    seasonal=cfg['seasonal'],
                    m=cfg['m'],
                    max_p=cfg['max_p'],
                    max_q=cfg['max_q'],
                    max_d=2,
                    max_P=2,
                    max_Q=2,
                    max_D=1,
                    stepwise=True,
                    suppress_warnings=True,
                    error_action='ignore',
                    trace=False,
                )
                pred = model.predict(n_periods=val_size, exogenous=X_val)
                mae = np.mean(np.abs(y_val - pred))
                if mae < best_mae:
                    best_mae = mae
                    best_cfg = cfg
            except Exception:
                continue

        if best_cfg is None:
            best_cfg = {'seasonal': True, 'm': 7, 'max_p': 3, 'max_q': 3}

        self.model = auto_arima(
            y,
            exogenous=X,
            seasonal=best_cfg['seasonal'],
            m=best_cfg['m'],
            max_p=best_cfg['max_p'],
            max_q=best_cfg['max_q'],
            max_d=2,
            max_P=2,
            max_Q=2,
            max_D=1,
            stepwise=True,
            suppress_warnings=True,
            error_action='ignore',
            trace=False,
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
