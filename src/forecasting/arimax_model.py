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

    def train(self, X_train, y_train, use_stl=True):
        """Treina ARIMAX com suavização de outliers e decomposição STL opcional.
        Args:
            X_train: Features exógenas de treino
            y_train: Série alvo de treino
            use_stl: Se True, aplica decomposição STL antes do ARIMAX
        """
        import numpy as np
        if X_train is None:
            raise ValueError("X_train não pode ser None para ARIMAX")

        y = y_train.values if hasattr(y_train, 'values') else np.asarray(y_train)
        X = X_train.values if hasattr(X_train, 'values') else np.asarray(X_train)

        # Suavizar outliers antes do treino
        try:
            from forecasting.advanced_features import AdvancedFeatureEngineer
            engineer = AdvancedFeatureEngineer(self.store_name, verbose=False)
            y_clean, outlier_mask = engineer.detect_and_handle_outliers(y, method='iqr_zscore', threshold_iqr=1.5, threshold_z=3)
            y = y_clean
        except Exception as e:
            print(f"  [ARIMAX {self.store_name}] Falha ao suavizar outliers: {e}. Prosseguindo com série original.")

        # Decomposição STL (opcional, testa 7 e 14)
        self.stl_trend = None
        self.stl_seasonal = None
        stl_periods = [7, 14]
        self.stl_period_used = None
        if use_stl:
            for period in stl_periods:
                try:
                    from statsmodels.tsa.seasonal import STL
                    stl = STL(y, period=period, robust=True)
                    res = stl.fit()
                    self.stl_trend = res.trend
                    self.stl_seasonal = res.seasonal
                    y = res.resid
                    self.stl_period_used = period
                    break
                except Exception as e:
                    print(f"  [ARIMAX {self.store_name}] Falha STL(period={period}): {e}")
                    self.stl_trend = None
                    self.stl_seasonal = None

        # Fallback robusto para séries curtas ou zeros
        if len(y) < 40 or np.all(y == 0):
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
            {'seasonal': True, 'm': 14, 'max_p': 3, 'max_q': 3},
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
        """Prevê os próximos n_periods com exógenas, somando tendência/sazonalidade STL se usada.
        """
        if not self.is_trained:
            raise RuntimeError("Modelo não foi treinado")
        if X_test is None:
            raise ValueError("X_test não pode ser None para ARIMAX")

        X = X_test.values if hasattr(X_test, 'values') else np.asarray(X_test)
        forecast = self.model.predict(n_periods=n_periods, exogenous=X)
        # Se usou STL, somar tendência e sazonalidade previstas
        if hasattr(self, 'stl_trend') and self.stl_trend is not None and self.stl_seasonal is not None:
            trend_forecast = np.full(n_periods, self.stl_trend[-1])
            period = self.stl_period_used if hasattr(self, 'stl_period_used') and self.stl_period_used else 7
            seasonal_cycle = self.stl_seasonal[-period:]
            seasonal_forecast = np.resize(seasonal_cycle, n_periods)
            forecast = forecast + trend_forecast + seasonal_forecast
        return np.maximum(forecast, 0)

    def save(self, path):
        """Salva modelo ARIMAX em pickle."""
        print(f"[ARIMAX SAVE] Salvando modelo em: {path}")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(path, 'wb') as f:
                pickle.dump(self.model, f)
            print(f"[ARIMAX SAVE] Sucesso ao salvar: {path}")
        except Exception as e:
            print(f"[ARIMAX SAVE] ERRO ao salvar {path}: {e}")

    def load(self, path):
        """Carrega modelo ARIMAX de pickle."""
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
        self.is_trained = True
