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
        # auto_arima: busca (p,d,q) e (P,D,Q) com sazonalidade
        self.model = auto_arima(
            y_train.values if hasattr(y_train, 'values') else y_train,
            seasonal=True,
            m=7,  # Sazonalidade semanal
            max_p=5, max_q=5, max_d=2,
            maxiter=100,
            suppress_warnings=True,
            trace=False
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
