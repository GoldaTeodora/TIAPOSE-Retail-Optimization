"""
Modelo XGBoost multivariado com treino e salvamento.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pickle
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from xgboost import XGBRegressor
from forecasting.base import BaseForecaster


class XGBoostForecaster(BaseForecaster):
    """XGBoost com features engineered."""
    
    def __init__(self, store_name):
        """Inicializa XGBoost forecaster.
        
        Args:
            store_name: Nome da loja
        """
        super().__init__(store_name)
        self.feature_names = None
    
    def train(self, X_train, y_train):
        """Treina modelo XGBoost.
        
        Args:
            X_train: DataFrame/array com features
            y_train: Target de treino
        """
        base_model = XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='reg:squarederror',
            random_state=42,
            verbosity=0
        )

        # Tuning com TimeSeriesSplit quando há dados suficientes.
        if len(y_train) >= 180:
            tscv = TimeSeriesSplit(n_splits=4)
            param_dist = {
                'n_estimators': [300, 500, 700],
                'max_depth': [4, 5, 6, 8],
                'learning_rate': [0.01, 0.03, 0.05, 0.08],
                'subsample': [0.75, 0.85, 1.0],
                'colsample_bytree': [0.75, 0.85, 1.0],
                'min_child_weight': [1, 3, 5],
                'reg_alpha': [0.0, 0.1, 0.5],
                'reg_lambda': [1.0, 1.5, 2.0],
            }
            search = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=param_dist,
                n_iter=20,
                scoring='neg_mean_absolute_error',
                cv=tscv,
                random_state=42,
                n_jobs=-1,
                verbose=0,
            )
            search.fit(X_train, y_train)
            self.model = search.best_estimator_
        else:
            self.model = base_model
            self.model.fit(X_train, y_train)

        self.feature_names = X_train.columns.tolist() if hasattr(X_train, 'columns') else None
        self.is_trained = True
    
    def predict(self, X_test=None, n_periods=7):
        """Prevê os próximos n_periods usando X_test.
        
        Args:
            X_test: DataFrame/array com features dos próximos 7 dias
            n_periods: Ignorado (usa shape de X_test)
        
        Returns:
            Array com previsões
        """
        if not self.is_trained or self.model is None:
            raise RuntimeError("Modelo não foi treinado")
        
        if X_test is None:
            raise ValueError("X_test não pode ser None para XGBoost")
        
        forecast = self.model.predict(X_test)
        # XGBoost pode gerar valores negativos
        return np.maximum(forecast, 0)
    
    def save(self, path):
        """Salva modelo XGBoost.
        
        Args:
            path: Caminho do arquivo
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump((self.model, self.feature_names), f)
    
    def load(self, path):
        """Carrega modelo XGBoost.
        
        Args:
            path: Caminho do arquivo
        """
        with open(path, 'rb') as f:
            self.model, self.feature_names = pickle.load(f)
        self.is_trained = True
