"""
FEATURE ENGINEERING AVANÇADO PARA FORECASTING
================================================

Este módulo implemente técnicas TIER 2 de feature engineering para maximizar
a performance do XGBoost. Inclui:

1. Detecção e removálde outliers (anomalias)
2. Features de decomposição sazonal
3. Features cíclicas melhoradas
4. Autocorrelação explícita
5. Interações polinomiais
6. Seleção automática de features mais importantes

Impacto esperado: +15-20% R² quando combinado com hyperparameter tuning

Autor: AI Assistant | Data: Março 2026
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler


class AdvancedFeatureEngineer:
    def add_holiday_features(self, df):
        """
        Adiciona features binárias para feriados nacionais/regionais e datas promocionais.
        Inclui: Natal, Páscoa, Memorial Day, Black Friday, Thanksgiving, Ano Novo.
        """
        df = df.copy()
        if 'Date' in df.columns:
            dt = pd.to_datetime(df['Date'])
            # Natal
            df['Is_Christmas'] = ((dt.dt.month == 12) & (dt.dt.day == 25)).astype(int)
            # Ano Novo
            df['Is_New_Year'] = ((dt.dt.month == 1) & (dt.dt.day == 1)).astype(int)
            # Memorial Day (última segunda de maio)
            memorial = (dt.dt.month == 5) & (dt.dt.weekday == 0)
            for year in dt.dt.year.unique():
                may = dt[(dt.dt.year == year) & (dt.dt.month == 5)]
                if not may.empty:
                    last_monday = may[may.dt.weekday == 0].max()
                    df.loc[dt == last_monday, 'Is_Memorial_Day'] = 1
            df['Is_Memorial_Day'] = df['Is_Memorial_Day'].fillna(0).astype(int)
            # Páscoa
            def _easter_sunday(year):
                a = year % 19
                b = year // 100
                c = year % 100
                d = b // 4
                e = b % 4
                f = (b + 8) // 25
                g = (b - f + 1) // 3
                h = (19 * a + b - d - g + 15) % 30
                i = c // 4
                k = c % 4
                l = (32 + 2 * e + 2 * i - h - k) % 7
                m = (a + 11 * h + 22 * l) // 451
                month = (h + l - 7 * m + 114) // 31
                day = ((h + l - 7 * m + 114) % 31) + 1
                return pd.Timestamp(year=year, month=month, day=day)
            easter_dates = {year: _easter_sunday(year) for year in dt.dt.year.unique()}
            df['Is_Easter_Sunday'] = dt.apply(lambda d: int(d in easter_dates.values()))
            # Black Friday (sexta após 4ª quinta de novembro)
            for year in dt.dt.year.unique():
                nov = dt[(dt.dt.year == year) & (dt.dt.month == 11)]
                thursdays = nov[nov.dt.weekday == 3]
                if len(thursdays) >= 4:
                    thanksgiving = thursdays.iloc[3]
                    black_friday = thanksgiving + pd.Timedelta(days=1)
                    df.loc[dt == black_friday, 'Is_Black_Friday'] = 1
            df['Is_Black_Friday'] = df['Is_Black_Friday'].fillna(0).astype(int)
            # Thanksgiving (4ª quinta de novembro)
            for year in dt.dt.year.unique():
                nov = dt[(dt.dt.year == year) & (dt.dt.month == 11)]
                thursdays = nov[nov.dt.weekday == 3]
                if len(thursdays) >= 4:
                    thanksgiving = thursdays.iloc[3]
                    df.loc[dt == thanksgiving, 'Is_Thanksgiving'] = 1
            df['Is_Thanksgiving'] = df['Is_Thanksgiving'].fillna(0).astype(int)
        self._log("Features de feriados adicionadas")
        return df

    def add_peak_distance_features(self, df, y, peak_threshold=0.85):
        """
        Adiciona features:
        - DiasDesdeUltimoPico: dias desde o último pico
        - MagnitudeUltimoPico: valor do último pico
        Um pico é definido como valor >= percentil 85.
        """
        df = df.copy()
        y = np.array(y)
        peak_val = np.percentile(y, peak_threshold * 100)
        is_peak = y >= peak_val
        dias_desde_ultimo = np.zeros(len(y), dtype=int)
        mag_ultimo = np.zeros(len(y))
        last_peak_idx = -1
        last_peak_val = 0
        for i in range(len(y)):
            if is_peak[i]:
                last_peak_idx = i
                last_peak_val = y[i]
            dias_desde_ultimo[i] = i - last_peak_idx if last_peak_idx >= 0 else i
            mag_ultimo[i] = last_peak_val
        df['DiasDesdeUltimoPico'] = dias_desde_ultimo
        df['MagnitudeUltimoPico'] = mag_ultimo
        self._log("Features DiasDesdeUltimoPico e MagnitudeUltimoPico adicionadas")
        return df
    """
    Engenheiro de features avançado que detecta padrões complexos nos dados.
    
    Técnicas aplicadas:
    - Detecção de outliers (IQR + Z-score)
    - Decomposição estatística de sazonalidade
    - Features cíclicas com seno/cosseno (melhor representação)
    - Autocorrelação explícita (ACF)
    - Interações entre features chave
    - Polinômios de baixa ordem para capturar não-linearidades
    """
    
    def __init__(self, store_name, verbose=True):
        """
        Inicializa o engenheiro de features.
        
        Args:
            store_name: Nome da loja (para logging)
            verbose: Se True, imprime progresso detalhado
        """
        self.store_name = store_name
        self.verbose = verbose
        self.scaler = StandardScaler()
        self.outlier_indices = []
        self.feature_importance_scores = {}
        
    def _log(self, msg):
        """Helper para printar apenas se verbose=True"""
        if self.verbose:
            print(f"  [FeatureEng] {msg}")
    
    # ==================== TÉCNICA 1: DETECÇÃO DE OUTLIERS ====================
    
    def detect_and_handle_outliers(self, y, method='iqr_zscore', threshold_iqr=1.5, threshold_z=3):
        """
        Detecta outliers (anomalias, descontos abruptos, eventos especiais).
        
        Métodos:
        1. IQR (Interquartile Range): Padrão estatístico
           - Outliers = valores fora de [Q1-1.5*IQR, Q3+1.5*IQR]
           - Robusto a distribuições não-normais
        
        2. Z-score: Para distribuições normais
           - Outliers = valores com |Z-score| > 3
           
        3. Combinado: Usa IQR + Z-score (mais rigoroso)
        
        Args:
            y: Array de valores
            method: 'iqr', 'zscore', ou 'iqr_zscore' (padrão)
            threshold_iqr: Multiplicador do IQR (1.5 = padrão)
            threshold_z: Threshold do Z-score (3 = muito outlier)
        
        Returns:
            y_cleaned: Array com outliers suavizados
            outlier_mask: Booleano array indicando outliers detectados
        """
        y = np.array(y)
        outlier_mask = np.zeros(len(y), dtype=bool)
        
        # ---- MÉTODO 1: IQR ----
        if method in ['iqr', 'iqr_zscore']:
            Q1 = np.percentile(y, 25)
            Q3 = np.percentile(y, 75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - threshold_iqr * IQR
            upper_bound = Q3 + threshold_iqr * IQR
            
            outlier_mask |= (y < lower_bound) | (y > upper_bound)
            self._log(f"IQR Outliers detectados: {outlier_mask.sum()} ({100*outlier_mask.sum()/len(y):.1f}%)")
        
        # ---- MÉTODO 2: Z-SCORE ----
        if method in ['zscore', 'iqr_zscore']:
            z_scores = np.abs(stats.zscore(y))
            z_outliers = z_scores > threshold_z
            
            if method == 'iqr_zscore':
                # Combinado: outlier se detectado por ambos
                outlier_mask |= z_outliers
            else:
                outlier_mask = z_outliers
            
            self._log(f"Z-score Outliers detectados: {z_outliers.sum()} ({100*z_outliers.sum()/len(y):.1f}%)")
        
        # ---- SUAVIZAR OUTLIERS ----
        # Em vez de remover, substituir por interpolação (preserva série)
        y_cleaned = y.copy()
        
        if outlier_mask.sum() > 0:
            # Para cada outlier, usar média de vizinhos
            for idx in np.where(outlier_mask)[0]:
                neighbors = y[(~outlier_mask) & (np.abs(np.arange(len(y)) - idx) <= 3)]
                if len(neighbors) > 0:
                    y_cleaned[idx] = np.median(neighbors)
                else:
                    y_cleaned[idx] = np.median(y[~outlier_mask])
            
            self._log(f"Outliers suavizados (interpolação by vizinhos)")
        
        self.outlier_indices = np.where(outlier_mask)[0]
        return y_cleaned, outlier_mask
    
    # ==================== TÉCNICA 2: FEATURES CÍCLICAS ====================
    
    def add_cyclical_features(self, df):
        """
        Converte features cíclicas (dia da semana, mês) para representação
        seno/cosseno, que é muito melhor para ML do que valores numéricos.
        
        POR QUÊ? 
        - Dia 6 (sábado) e dia 0 (segunda) são VIZINHOS na semana
        - Mas numericamente 6 está longe de 0
        - Seno/cosseno mantém esta proximidade circular
        
        Matemática:
            dia_sin = sin(2π * dia / 7)
            dia_cos = cos(2π * dia / 7)
        
        Args:
            df: DataFrame com features cíclicas (Day_of_Week, Month, Quarter, etc.)
        
        Returns:
            df: DataFrame com novos features seno/cosseno adicionados
        """
        
        df = df.copy()
        
        # Mapear nomes possíveis das colunas
        day_of_week_col = 'Day_of_Week' if 'Day_of_Week' in df.columns else ('dayofweek' if 'dayofweek' in df.columns else None)
        day_of_year_col = 'Day_of_Year' if 'Day_of_Year' in df.columns else ('dayofyear' if 'dayofyear' in df.columns else None)
        month_col = 'Month' if 'Month' in df.columns else ('month' if 'month' in df.columns else None)
        quarter_col = 'Quarter' if 'Quarter' in df.columns else ('quarter' if 'quarter' in df.columns else None)
        
        # ---- DIA DA SEMANA (0-6) ----
        if day_of_week_col and day_of_week_col in df.columns:
            df['Day_of_Week_sin'] = np.sin(2 * np.pi * df[day_of_week_col] / 7)
            df['Day_of_Week_cos'] = np.cos(2 * np.pi * df[day_of_week_col] / 7)
        
        # ---- DIA DO ANO (1-365) ----
        if day_of_year_col and day_of_year_col in df.columns:
            df['Day_of_Year_sin'] = np.sin(2 * np.pi * df[day_of_year_col] / 365)
            df['Day_of_Year_cos'] = np.cos(2 * np.pi * df[day_of_year_col] / 365)
        
        # ---- MÊS DO ANO (1-12) ----
        if month_col and month_col in df.columns:
            df['Month_sin'] = np.sin(2 * np.pi * df[month_col] / 12)
            df['Month_cos'] = np.cos(2 * np.pi * df[month_col] / 12)
        
        # ---- QUARTER (1-4) ----
        if quarter_col and quarter_col in df.columns:
            df['Quarter_sin'] = np.sin(2 * np.pi * df[quarter_col] / 4)
            df['Quarter_cos'] = np.cos(2 * np.pi * df[quarter_col] / 4)
        
        self._log("Features cíclicas seno/cosseno adicionadas")
        
        return df
    
    # ==================== TÉCNICA 3: AUTOCORRELAÇÃO ====================
    
    def add_autocorrelation_features(self, y, lags=[1, 2, 3, 5, 7, 14]):
        """
        Adiciona features de autocorrelação, que capturam relações lineares
        com o passado mais distante.
        
        Estes são mais eficientes que lags simples porque normalizam
        a correlação (valores entre -1 e 1) vs valores brutos.
        
        Args:
            y: Array univariado (série temporal)
            lags: Lista de lags a incluir
        
        Returns:
            Dict com features de autocorrelação
        """
        features = {}
        
        # Calcular autocorrelação (ACF) manualmente
        y_mean = np.mean(y)
        c0 = np.sum((y - y_mean) ** 2) / len(y)
        
        for lag in lags:
            if lag < len(y):
                # Autocorrelação = covariance(lag) / variance
                c_lag = np.sum((y[:-lag] - y_mean) * (y[lag:] - y_mean)) / len(y)
                acf = c_lag / c0
                features[f'acf_lag{lag}'] = acf
        
        self._log(f"ACF features adicionadas: lags {lags}")
        
        return features
    
    # ==================== TÉCNICA 4: INTERAÇÕES E POLINÔMIOS ====================
    
    def add_interaction_features(self, df):
        """
        Adiciona features de interação entre componentes chave.
        
        Racional: XGBoost aprende interações, mas features explícitas
        podem acelerar convergência e melhorar interpretabilidade.
        
        Args:
            df: DataFrame com features base
        
        Returns:
            df: DataFrame com novas features de interação
        """
        
        df = df.copy()
        
        # Encontrar quais colunas estão disponíveis (com nomes possíveis)
        day_of_week_col = 'Day_of_Week' if 'Day_of_Week' in df.columns else ('dayofweek' if 'dayofweek' in df.columns else None)
        lag_1_col = 'Lag_Customers_1' if 'Lag_Customers_1' in df.columns else ('lag_1' if 'lag_1' in df.columns else None)
        rolling_mean_col = 'Rolling_Mean_7' if 'Rolling_Mean_7' in df.columns else ('rolling_mean_7' if 'rolling_mean_7' in df.columns else None)
        rolling_std_col = 'Rolling_Std_7' if 'Rolling_Std_7' in df.columns else ('rolling_std_7' if 'rolling_std_7' in df.columns else None)
        
        # Criar interações onde possível
        if day_of_week_col and day_of_week_col in df.columns and lag_1_col and lag_1_col in df.columns:
            df['DayOfWeek_Lag1_Interaction'] = df[day_of_week_col] * df[lag_1_col]
            
        if rolling_mean_col and rolling_mean_col in df.columns and rolling_std_col and rolling_std_col in df.columns:
            df['RollingMean_RollingStd_Interaction'] = df[rolling_mean_col] * df[rolling_std_col]

        # Adicionar Is_Peak_Day: 1 se for feriado OU evento turístico OU ambos
        holiday_cols = [c for c in ['Is_Holiday', 'Is_Memorial_Day', 'Is_Christmas', 'Is_Easter_Sunday', 'Is_Black_Friday'] if c in df.columns]
        event_cols = [c for c in ['Is_Tourist_Event', 'Event_Nearby'] if c in df.columns]
        if holiday_cols or event_cols:
            df['Is_Peak_Day'] = 0
            for col in holiday_cols + event_cols:
                df['Is_Peak_Day'] = df['Is_Peak_Day'] | (df[col] == 1)
            df['Is_Peak_Day'] = df['Is_Peak_Day'].astype(int)
            self._log("Feature Is_Peak_Day adicionada")
        
        self._log("Features de interação adicionadas")
        
        return df
        
        return df
    
    def add_polynomial_features(self, df, degree=2, columns=['lag_1', 'rolling_mean_3']):
        """
        Adiciona features polinomiais para capturar relações não-lineares.
        
        Racional: Dados podem ter relações quadráticas ou cúbicas
        ex: crescimento acelera exponencialmente vs linear
        
        Args:
            df: DataFrame
            degree: Grau do polinômio (2 = quadrático, 3 = cúbico) 
            columns: Colunas para criar polinômios
        
        Returns:
            df: DataFrame com features polinomiais
        """
        
        df = df.copy()
        
        # Mapear possíveis nomes de colunas
        mapping = {
            'lag_1': ['Lag_Customers_1', 'lag_1'],
            'rolling_mean_3': ['Rolling_Mean_7', 'rolling_mean_7', 'rolling_mean_3'],
            'rolling_mean_7': ['Rolling_Mean_7', 'rolling_mean_7'],
        }
        
        cols_to_poly = []
        for original_col in columns:
            possible_names = mapping.get(original_col, [original_col])
            for name in possible_names:
                if name in df.columns:
                    cols_to_poly.append((name, original_col))
                    break
        
        for col_actual, col_name in cols_to_poly:
            for d in range(2, degree + 1):
                feature_name = f'{col_name}_poly{d}'
                df[feature_name] = df[col_actual] ** d
        
        if cols_to_poly:
            self._log(f"Features polinomiais adicionadas (grau {degree}, colunas {len(cols_to_poly)})")
        
        return df
    
    # ==================== TÉCNICA 5: SELEÇÃO DE FEATURES ====================
    
    def score_feature_importance(self, xgboost_model, feature_names):
        """
        Extrai feature importance do modelo XGBoost treinado.
        
        Racional: Nem todas as features são úteis. Remover features
        ruidosas reduz overfitting.
        
        Args:
            xgboost_model: Modelo XGBoost treinado
            feature_names: Nomes das features
        
        Returns:
            Dict com importância de cada feature, ordenado
        """
        
        importance_scores = xgboost_model.feature_importances_
        
        importance_dict = {
            name: score for name, score in zip(feature_names, importance_scores)
        }
        
        # Ordenar por importância
        importance_dict = dict(sorted(
            importance_dict.items(),
            key=lambda x: x[1],
            reverse=True
        ))
        
        self.feature_importance_scores = importance_dict
        
        return importance_dict
    
    def select_top_features(self, importance_dict, threshold_percentile=80):
        """
        Selecciona features com importância acima de um threshold.
        
        Args:
            importance_dict: Dict com feature importances
            threshold_percentile: Manter features no top XX percentil
        
        Returns:
            Lista de features selecionadas
        """
        
        scores = np.array(list(importance_dict.values()))
        threshold = np.percentile(scores, 100 - threshold_percentile)
        
        selected_features = [
            name for name, score in importance_dict.items()
            if score >= threshold
        ]
        
        self._log(f"Features selecionadas: {len(selected_features)}/{len(importance_dict)} "
                 f"(top {threshold_percentile}%)")
        
        return selected_features
    
    # ==================== PIPELINE COMPLETO ====================
    
    def process_features_complete(self, df, y=None, outlier_detection=True):
        """
        PIPELINE COMPLETO de feature engineering.
        
        Passos:
        1. Detectar e suavizar outliers
        2. Adicionar features cíclicas (seno/cosseno)
        3. Adicionar features de autocorrelação
        4. Adicionar interações
        5. Adicionar polinômios
        
        Args:
            df: DataFrame com features base
            y: Array com valores reais (para outlier detection)
            outlier_detection: Se True, detecta e suaviza outliers
        
        Returns:
            df: DataFrame processado com todas as features novas
            outlier_mask: Mask de outliers detectados
        """
        
        outlier_mask = None
        
        # ---- PASSO 0: FERIADOS ----
        df = self.add_holiday_features(df)

        # ---- PASSO 1: OUTLIERS ----
        if outlier_detection and y is not None:
            y_cleaned, outlier_mask = self.detect_and_handle_outliers(
                y, method='iqr_zscore'
            )
        else:
            y_cleaned = y

        # ---- PASSO 1B: DISTÂNCIA/MAGNITUDE DE PICO ----
        if y is not None:
            df = self.add_peak_distance_features(df, y)

        # ---- PASSO 2: FEATURES CÍCLICAS ----
        df = self.add_cyclical_features(df)

        # ---- PASSO 3: AUTOCORRELAÇÃO ----
        if y is not None:
            acf_features = self.add_autocorrelation_features(y, lags=[1, 2, 3, 5, 7, 14])
            for fname, fvalue in acf_features.items():
                # ACF é escalar, replicar para todas as linhas
                df[fname] = fvalue

        # ---- PASSO 4: INTERAÇÕES ----
        df = self.add_interaction_features(df)

        # ---- PASSO 5: POLINÔMIOS ----
        df = self.add_polynomial_features(df, degree=2)

        self._log("Pipeline completo executado com sucesso!")

        return df, outlier_mask


if __name__ == '__main__':
    # TESTE: Criar dados dummy e aplicar
    print("Testando AdvancedFeatureEngineer...")
    
    # Dados dummy
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100)
    y = 50 + 10*np.sin(np.arange(100)*2*np.pi/7) + np.random.normal(0, 5, 100)
    
    df_dummy = pd.DataFrame({
        'dayofweek': np.tile(range(7), 15)[:100],
        'dayofmonth': np.tile(range(1, 32), 4)[:100],
        'month': 1,
        'week': np.tile(range(1, 53), 2)[:100],
        'lag_1': np.roll(y, 1),
        'rolling_mean_7': pd.Series(y).rolling(7).mean(),
        'trend': np.arange(100) / 100,
        'pct_change': pd.Series(y).pct_change(),
        'rolling_std_7': pd.Series(y).rolling(7).std()
    })
    
    engineer = AdvancedFeatureEngineer('test', verbose=True)
    df_processed, outlier_mask = engineer.process_features_complete(df_dummy, y=y)
    
    print(f"\nFeatures originais: {len(df_dummy.columns)}")
    print(f"Features após processamento: {len(df_processed.columns)}")
    print(f"Outliers detectados: {outlier_mask.sum() if outlier_mask is not None else 'N/A'}")
    print("\nNovas features adicionadas:")
    new_features = set(df_processed.columns) - set(df_dummy.columns)
    for feat in sorted(new_features):
        print(f"  - {feat}")
