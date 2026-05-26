"""
Otimização rolling window para todo o histórico de todas as lojas.
Gera planos ótimos semana a semana, salva CSV consolidado por loja.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from core.config import STORES, REPORTS_PATH, get_enriched_data_path, get_report_path, XGBOOST_FEATURES
from forecasting.ml_model import XGBoostForecaster
from optimization.methods import OptimizationMethods
from core.profit_calculator import calculate_weekly_profit

REPORTS_DIR = REPORTS_PATH
REPORTS_DIR.mkdir(exist_ok=True)

WINDOW = 7
TRAIN_MIN = 365  # mínimo 1 ano para treinar

def optimize_history_for_store(store_name, objective='O1'):
    print(f"\n[STORE] {store_name.upper()}")
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
    for k, v in col_map.items():
        if k not in df.columns and v in df.columns:
            df[k] = df[v]
    X = df[[f for f in XGBOOST_FEATURES if f in df.columns]].ffill().bfill().fillna(0)
    y = df['Num_Customers'] if 'Num_Customers' in df.columns else df['Num_Customers_Clean']
    dates = pd.to_datetime(df['Date'])

    results = []
    for start in range(TRAIN_MIN, len(df) - WINDOW + 1):
        X_train = X.iloc[:start]
        y_train = y[:start]
        X_test = X.iloc[start:start+WINDOW]
        y_test = y[start:start+WINDOW].reset_index(drop=True)
        week_dates = dates.iloc[start:start+WINDOW].tolist()

        # Treina modelo
        xgb = XGBoostForecaster(store_name)
        xgb.train(X_train, y_train)
        forecast = xgb.predict(X_test, n_periods=WINDOW)

        # Otimiza plano
        optimizer = OptimizationMethods(store_name, forecast, objective)
        result = optimizer.optimize()
        J = np.clip(result['solution'][0:7], 0, 20).astype(int)
        Xs = np.clip(result['solution'][7:14], 0, 20).astype(int)
        PR = np.clip(result['solution'][14:21], 0.0, 0.3)

        # Lucro real (com clientes reais)
        daily_plans = []
        for i in range(WINDOW):
            daily_plans.append({
                'num_customers': y_test.iloc[i],
                'J': J[i],
                'X': Xs[i],
                'PR': PR[i],
                'is_weekend': (i == 0) or (i == 6)
            })
        profit_info = calculate_weekly_profit(daily_plans, store_name)

        results.append({
            'start_date': week_dates[0],
            'end_date': week_dates[-1],
            'lucro_otimo': float(profit_info['weekly_profit']),
            'unidades': int(profit_info['total_units']),
            'hr_cost': float(profit_info['total_hr']),
            'J': [int(x) for x in J],
            'X': [int(x) for x in Xs],
            'PR': [float(x) for x in PR],
            'clientes_reais': [int(x) for x in y_test],
            'clientes_previstos': [float(x) for x in forecast]
        })

    # Salvar CSV
    out_path = REPORTS_DIR / f'rolling_optimum_{store_name}_{objective}.csv'
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"  [OK] {out_path} ({len(results)} semanas) [{objective}]")

if __name__ == '__main__':
    objectives = ['O1', 'O2', 'O3']
    for store in STORES:
        for obj in objectives:
            optimize_history_for_store(store, objective=obj)
