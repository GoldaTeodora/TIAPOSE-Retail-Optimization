"""
Modelo ETS (Exponential Smoothing) com treino e salvamento.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pickle
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from forecasting.base import BaseForecaster


class ETSForecaster(BaseForecaster):
    """Exponential Smoothing (Holt-Winters)."""
    
    def train(self, X_train, y_train):
        """Treina modelo ETS.
        
        Args:
            X_train: Ignorado (ETS é univariado)
            y_train: Série temporal de treino
        """
        y = y_train.values if hasattr(y_train, 'values') else y_train
        
        # Evitar séries muito pequenas ou com variância zero
        if len(y) < 14:  # Precisa de pelo menos 2*m
            self.model = None
            self.is_trained = True
            self._fallback_forecast = np.mean(y[-7:])
            return
        
        try:
            val_size = min(14, max(7, len(y) // 5))
            y_fit = y[:-val_size]
            y_val = y[-val_size:]

            candidates = [
                {'trend': 'add', 'seasonal': 'add', 'damped_trend': False},
                {'trend': 'add', 'seasonal': 'add', 'damped_trend': True},
                {'trend': None, 'seasonal': 'add', 'damped_trend': False},
                {'trend': 'add', 'seasonal': None, 'damped_trend': False},
                {'trend': None, 'seasonal': None, 'damped_trend': False},
            ]

            best_model = None
            best_mae = np.inf

            for cfg in candidates:
                try:
                    fit = ExponentialSmoothing(
                        y_fit,
                        seasonal_periods=7,
                        trend=cfg['trend'],
                        seasonal=cfg['seasonal'],
                        damped_trend=cfg['damped_trend'],
                        initialization_method='estimated'
                    ).fit(optimized=True)
                    pred = fit.forecast(steps=val_size)
                    mae = np.mean(np.abs(y_val - pred))
                    if mae < best_mae:
                        best_mae = mae
                        best_model = cfg
                except Exception:
                    continue

            if best_model is None:
                raise RuntimeError("Nenhuma configuração ETS convergiu")

            self.model = ExponentialSmoothing(
                y,
                seasonal_periods=7,
                trend=best_model['trend'],
                seasonal=best_model['seasonal'],
                damped_trend=best_model['damped_trend'],
                initialization_method='estimated'
            ).fit(optimized=True)
            self._fallback_forecast = None
            self.is_trained = True
        except Exception as e:
            # Fallback para média móvel se ETS falhar
            print(f"  [ETS {self.store_name}] Falha ao treinar: {e}. Usando fallback.")
            self.model = None
            self._fallback_forecast = np.mean(y[-7:])
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
        
        if self.model is None:
            # Fallback: repetir última média
            return np.full(n_periods, self._fallback_forecast)
        
        # forecast() retorna apenas as previsões (sem incluir série histórica)
        forecast = self.model.forecast(steps=n_periods)
        # ETS pode gerar valores negativos
        return np.maximum(forecast, 0)
    
    def save(self, path):
        """Salva modelo ETS em pickle.
        
        Args:
            path: Caminho do arquivo
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump((self.model, self._fallback_forecast), f)
    
    def load(self, path):
        """Carrega modelo ETS de pickle.
        
        Args:
            path: Caminho do arquivo
        """
        with open(path, 'rb') as f:
            self.model, self._fallback_forecast = pickle.load(f)
        self.is_trained = True
