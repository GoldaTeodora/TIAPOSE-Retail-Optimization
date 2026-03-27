"""
Classe base para modelos de forecasting.
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseForecaster(ABC):
    """Classe base para todos os modelos de forecasting."""
    
    def __init__(self, store_name):
        """Inicializa forecaster.
        
        Args:
            store_name: Nome da loja ('baltimore', 'lancaster', etc.)
        """
        self.store_name = store_name
        self.model = None
        self.is_trained = False
    
    @abstractmethod
    def train(self, X_train, y_train):
        """Treina o modelo.
        
        Args:
            X_train: Features de treino (pode ser None para métodos univariados)
            y_train: Target de treino
        """
        pass
    
    @abstractmethod
    def predict(self, X_test=None, n_periods=7):
        """Faz previsões.
        
        Args:
            X_test: Features de teste (pode ser None para métodos univariados)
            n_periods: Número de períodos a prever
        
        Returns:
            Array com previsões
        """
        pass
    
    @abstractmethod
    def save(self, path):
        """Salva o modelo.
        
        Args:
            path: Caminho para salvar
        """
        pass
    
    @abstractmethod
    def load(self, path):
        """Carrega o modelo.
        
        Args:
            path: Caminho para carregar
        """
        pass
    
    def forecast_next_week(self, df):
        """Faz previsão para os próximos 7 dias com base em df.
        
        Args:
            df: DataFrame com dados históricos
        
        Returns:
            Array com 7 previsões
        """
        return self.predict(n_periods=7)
