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
    
    def train(self, X_train, y_train, use_stl=True):
        """Treina modelo ETS com suavização de outliers e decomposição STL opcional.
        Ajuste de pico: salva histórico para ajuste futuro.
        """
        import pandas as pd
        y = y_train.values if hasattr(y_train, 'values') else y_train
        self.y_hist = y.copy()  # Salva histórico para ajuste de pico
        self.dow_hist = None
        if hasattr(y_train, 'index') or hasattr(y_train, 'Date'):
            # Tenta extrair dia da semana
            if hasattr(y_train, 'index') and isinstance(y_train.index, pd.DatetimeIndex):
                self.dow_hist = y_train.index.dayofweek
            elif hasattr(y_train, 'Date'):
                self.dow_hist = pd.to_datetime(y_train['Date']).dayofweek

        # Suavizar outliers antes do treino (robustez, sem overfitting)
        try:
            from forecasting.advanced_features import AdvancedFeatureEngineer
            engineer = AdvancedFeatureEngineer(self.store_name, verbose=False)
            y_clean, outlier_mask = engineer.detect_and_handle_outliers(y, method='iqr_zscore', threshold_iqr=1.5, threshold_z=3)
            y = y_clean
        except Exception as e:
            print(f"  [ETS {self.store_name}] Falha ao suavizar outliers: {e}. Prosseguindo com série original.")

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
                    print(f"  [ETS {self.store_name}] Falha STL(period={period}): {e}")
                    self.stl_trend = None
                    self.stl_seasonal = None

        # Evitar séries muito pequenas ou com variância zero
        if len(y) < 14 or np.all(y == 0):  # Precisa de pelo menos 2*m e não pode ser tudo zero
            self.model = None
            self.is_trained = True
            # Fallback: média dos últimos 7 não nulos, senão zero
            vals = y[-14:][y[-14:] > 0] if np.any(y[-14:] > 0) else y[-14:]
            self._fallback_forecast = np.mean(vals) if len(vals) > 0 else 0.0
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
    
    def predict(self, X_test=None, n_periods=7, test_dates=None):
        """Prevê os próximos n_periods, com ajuste para Is_Peak_Day se possível.
        
        Args:
            X_test: Ignorado
            n_periods: Número de dias a prever
            test_dates: Datas dos períodos previstos (opcional, para ajuste de pico)
        
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

        # Se usou STL, somar tendência e sazonalidade previstas
        if hasattr(self, 'stl_trend') and self.stl_trend is not None and self.stl_seasonal is not None:
            # Projeta tendência e sazonalidade para frente (repete último valor ou ciclo)
            trend_forecast = np.full(n_periods, self.stl_trend[-1])
            period = self.stl_period_used if hasattr(self, 'stl_period_used') and self.stl_period_used else 7
            seasonal_cycle = self.stl_seasonal[-period:]
            seasonal_forecast = np.resize(seasonal_cycle, n_periods)
            forecast = forecast + trend_forecast + seasonal_forecast

        # Ajuste simples para Is_Peak_Day (se datas fornecidas)
        if test_dates is not None:
            import pandas as pd
            # Regra: se Is_Peak_Day (ex: feriado/evento), aumenta previsão para média dos picos anteriores
            # (sem usar ML, só regra)
            try:
                from forecasting.advanced_features import AdvancedFeatureEngineer
                # Carregar histórico de picos
                # Aqui, para simplificação, assume-se que o usuário pode passar um DataFrame com coluna Is_Peak_Day
                if isinstance(test_dates, pd.DataFrame) and 'Is_Peak_Day' in test_dates.columns:
                    peak_mask = test_dates['Is_Peak_Day'].values.astype(bool)
                    # Média dos top 5 valores históricos do mesmo dia da semana (ou geral)
                    pico_valor = None
                    if hasattr(self, 'y_hist') and self.y_hist is not None:
                        y_hist = np.array(self.y_hist)
                        if hasattr(self, 'dow_hist') and self.dow_hist is not None and len(self.dow_hist) == len(y_hist):
                            # Se test_dates tem datas, tenta extrair dia da semana
                            if 'Date' in test_dates.columns:
                                test_dow = pd.to_datetime(test_dates['Date']).dt.dayofweek.values
                                for i, is_peak in enumerate(peak_mask):
                                    if is_peak:
                                        dow = test_dow[i]
                                        mask = (self.dow_hist == dow)
                                        top_picos = np.sort(y_hist[mask])[-5:] if np.sum(mask) >= 5 else y_hist[mask]
                                        pico_valor = np.mean(top_picos) if len(top_picos) > 0 else np.mean(y_hist[-7:])
                                        forecast[i] = pico_valor
                            else:
                                # Fallback: média dos top 5 gerais
                                top_picos = np.sort(y_hist)[-5:] if len(y_hist) >= 5 else y_hist
                                pico_valor = np.mean(top_picos)
                                forecast[peak_mask] = pico_valor
                        else:
                            # Fallback: média dos top 5 gerais
                            top_picos = np.sort(y_hist)[-5:] if len(y_hist) >= 5 else y_hist
                            pico_valor = np.mean(top_picos)
                            forecast[peak_mask] = pico_valor
            except Exception as e:
                print(f"  [ETS {self.store_name}] Falha no ajuste Is_Peak_Day: {e}")

        # ETS pode gerar valores negativos
        return np.maximum(forecast, 0)
    
    def save(self, path):
        """Salva modelo ETS em pickle.
        Args:
            path: Caminho do arquivo
        """
        print(f"[ETS SAVE] Salvando modelo em: {path}")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(path, 'wb') as f:
                pickle.dump((self.model, self._fallback_forecast), f)
            print(f"[ETS SAVE] Sucesso ao salvar: {path}")
        except Exception as e:
            print(f"[ETS SAVE] ERRO ao salvar {path}: {e}")
    
    def load(self, path):
        """Carrega modelo ETS de pickle.
        
        Args:
            path: Caminho do arquivo
        """
        with open(path, 'rb') as f:
            self.model, self._fallback_forecast = pickle.load(f)
        self.is_trained = True
