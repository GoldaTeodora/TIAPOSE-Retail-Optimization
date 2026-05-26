"""
Pipeline para prever clientes, otimizar lucro e salvar plano ótimo.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from core.config import STORES, get_enriched_data_path, get_report_path, XGBOOST_FEATURES
from forecasting.ml_model import XGBoostForecaster
from optimization.methods import OptimizationMethods


def optimize_store_profit(store_name, forecast_horizon=7, objective='O1'):
    """
    Para uma loja, prevê clientes com XGBoost e otimiza o lucro semanal.
    Salva plano ótimo em reports/plan_{store}_O1.csv
    """
    # Carregar dados enriquecidos (features)
    enriched_path = get_enriched_data_path(store_name)
    df = pd.read_csv(enriched_path)
    # Corrigir nomes de colunas para compatibilidade
    col_map = {
        'DayOfMonth': 'DayOfMonth',
        'WeekOfYear': 'WeekOfYear',
        'Year': 'Year',
        'Is_Christmas': 'is_christmas',
        'Is_Easter_Sunday': 'is_easter_sunday',
        'Is_Known_Closed_Day': 'is_known_closed_day',
        'Is_Peak_Day': 'is_peak_day',
        'Lag_Customers_28': 'Lag_Customers_28',
        'Rolling_Mean_14': 'rolling_mean_14',
        'Rolling_Std_14': 'rolling_std_14',
    }
    # Ajustar nomes se necessário
    for k, v in col_map.items():
        if k not in df.columns and v in df.columns:
            df[k] = df[v]
    X = df[[f for f in XGBOOST_FEATURES if f in df.columns]].ffill().bfill().fillna(0)
    y = df['Num_Customers'] if 'Num_Customers' in df.columns else df['Num_Customers_Clean']
    
    # Treinar XGBoost
    split_idx = int(len(df) * 0.85)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    xgb = XGBoostForecaster(store_name)
    xgb.train(X_train, y_train)
    # Prever próximos 7 dias
    X_forecast = X_test.iloc[:forecast_horizon]
    forecast = xgb.predict(X_forecast, n_periods=forecast_horizon)
    
    # Otimizar plano de RH e promoção
    optimizer = OptimizationMethods(store_name, forecast, objective=objective)
    result = optimizer.optimize()
    plan = result['solution']
    
    # Salvar plano ótimo
    plan_df = pd.DataFrame({
        'Dia': np.arange(1, 8),
        'Previsao_Clientes': forecast,
        'Juniores': np.round(plan[:7]).astype(int),
        'Experts': np.round(plan[7:14]).astype(int),
        'Promocao': np.round(plan[14:21], 3)
    })
    out_path = get_report_path(f'plan_{store_name}_{objective}.csv')
    plan_df.to_csv(out_path, index=False)
    print(f"[OK] Plano ótimo salvo em: {out_path}")
    return plan_df, result

if __name__ == '__main__':
    for store in STORES:
        optimize_store_profit(store, forecast_horizon=7, objective='O1')
