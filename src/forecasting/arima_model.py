"""
Modelo ARIMA com treino e salvamento.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pickle
from pmdarima import auto_arima
from forecasting.base import BaseForecaster


class ARIMAForecaster(BaseForecaster):
    """ARIMA com auto-identificação de parâmetros."""
    
    def train(self, X_train, y_train):
        """Treina ARIMA com auto_arima.
        
        Args:
            X_train: Ignorado (ARIMA é univariado)
            y_train: Série temporal de treino
        """
        y = y_train.values if hasattr(y_train, 'values') else np.asarray(y_train)
        if len(y) < 30:
            self.model = auto_arima(
                y,
                seasonal=False,
                suppress_warnings=True,
                trace=False,
                error_action='ignore'
            )
            self.is_trained = True
            return

        val_size = min(28, max(14, len(y) // 6))
        y_fit = y[:-val_size]
        y_val = y[-val_size:]

        candidates = [
            {'seasonal': True, 'm': 7},
            {'seasonal': False, 'm': 1},
        ]

        best_model = None
        best_mae = np.inf

        for cfg in candidates:
            try:
                model = auto_arima(
                    y_fit,
                    seasonal=cfg['seasonal'],
                    m=cfg['m'],
                    max_p=5,
                    max_q=5,
                    max_d=2,
                    max_P=2,
                    max_Q=2,
                    max_D=1,
                    maxiter=200,
                    stepwise=True,
                    suppress_warnings=True,
                    error_action='ignore',
                    trace=False,
                )
                pred = model.predict(n_periods=val_size)
                mae = np.mean(np.abs(y_val - pred))
                if mae < best_mae:
                    best_mae = mae
                    best_model = cfg
            except Exception:
                continue

        if best_model is None:
            best_model = {'seasonal': True, 'm': 7}

        self.model = auto_arima(
            y,
            seasonal=best_model['seasonal'],
            m=best_model['m'],
            max_p=5,
            max_q=5,
            max_d=2,
            max_P=2,
            max_Q=2,
            max_D=1,
            maxiter=200,
            stepwise=True,
            suppress_warnings=True,
            error_action='ignore',
            trace=False,
        )
        self.is_trained = True
    
    def predict(self, X_test=None, n_periods=7):
        """Prevê os próximos n_periods.
        
        Args:
            X_test: Ignorado
            n_periods: Número de dias a prever
        
        Returns:
            Array com previsões
        """
        if not self.is_trained:
            raise RuntimeError("Modelo não foi treinado")
        
        forecast = self.model.predict(n_periods=n_periods)
        # ARIMA pode gerar valores negativos com séries com muitas zeros
        return np.maximum(forecast, 0)
    
    def save(self, path):
        """Salva modelo ARIMA em pickle.
        
        Args:
            path: Caminho do arquivo
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
    
    def load(self, path):
        """Carrega modelo ARIMA de pickle.
        
        Args:
            path: Caminho do arquivo
        """
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
        self.is_trained = True
