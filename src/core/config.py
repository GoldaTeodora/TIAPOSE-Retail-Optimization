"""
Configuração centralizada: parâmetros das lojas, custos, capacidades.
"""

import os
from pathlib import Path

# Diretórios
PROJECT_ROOT = Path(__file__).parent.parent.parent
SRC_PATH = PROJECT_ROOT / "src"
DATA_PATH = PROJECT_ROOT / "data"
MODELS_PATH = PROJECT_ROOT / "models"
REPORTS_PATH = PROJECT_ROOT / "reports"

# Garantir que existem
REPORTS_PATH.mkdir(exist_ok=True)
MODELS_PATH.mkdir(exist_ok=True)

# Parâmetros por loja (dari especificação do projeto)
STORE_PARAMS = {
    'baltimore': {
        'F_j': 1.00,      # Fator produtividade Junior
        'F_x': 1.15,      # Fator produtividade Expert
        'W_s': 700        # Custo fixo semanal
    },
    'lancaster': {
        'F_j': 1.05,
        'F_x': 1.20,
        'W_s': 730
    },
    'philadelphia': {
        'F_j': 1.10,
        'F_x': 1.15,
        'W_s': 760
    },
    'richmond': {
        'F_j': 1.15,
        'F_x': 1.25,
        'W_s': 800
    }
}

# Custos diários de RH
HR_COSTS = {
    'weekday': {'J': 60, 'X': 80},
    'weekend': {'J': 70, 'X': 95}
}

# Capacidade de atendimento
CUSTOMER_CAPACITY = {
    'J': 6,  # Clientes por junior
    'X': 7   # Clientes por expert
}

# Forecasting: Rolling Window
ROLLING_WINDOW = {
    'test_size': 7,           # 7 dias para testar
    'num_iterations': 15,     # 15 iterações
    'random_state': 42
}

# Optimization
OPTIMIZATION = {
    'methods': ['hill_climbing', 'simulated_annealing', 'genetic_algorithm', 'particle_swarm'],
    'objectives': ['O1', 'O2', 'O3'],
    'J_bounds': (0, 20),
    'X_bounds': (0, 20),
    'PR_bounds': (0.0, 0.3),
    'units_constraint_O2': 10000  # Max units para O2
}

# Features para XGBoost
XGBOOST_FEATURES = [
    'Pct_On_Sale',
    'Day_of_Week', 'Is_Weekend', 'Month', 'Quarter', 'Day_of_Year',
    'Year', 'WeekOfYear', 'DayOfMonth',
    'Is_Christmas', 'Is_Easter_Sunday', 'Is_Known_Closed_Day',
    'Is_Black_Friday', 'Is_Tourist_Event', 'Is_Holiday', 'Is_Memorial_Day',
    'Event_Nearby', 'Holiday_Nearby',
    'Is_Peak_Day',
    'Lag_Customers_1', 'Lag_Customers_7', 'Lag_Customers_14',
    'Rolling_Mean_7', 'Rolling_Std_7', 'Rolling_Mean_14', 'Rolling_Std_14', 'Rolling_Mean_30'
]

STORES = list(STORE_PARAMS.keys())

def get_data_path(store_name, raw=False):
    """Retorna caminho para arquivo de dados."""
    folder = 'raw' if raw else 'processed'
    suffix = '' if raw else '_clean'
    return DATA_PATH / folder / f'{store_name}{suffix}.csv'

def get_model_path(store_name, model_type):
    """Retorna caminho para arquivo de modelo.
    
    Args:
        store_name: 'baltimore', 'lancaster', etc.
        model_type: 'arima', 'ets', 'xgboost', 'features'
    
    Returns:
        Path object
    """
    store_path = MODELS_PATH / store_name
    store_path.mkdir(parents=True, exist_ok=True)
    if model_type == 'xgboost':
        return store_path / f'{model_type}.joblib'
    else:
        return store_path / f'{model_type}.pkl'

def get_report_path(filename):
    """Retorna caminho para arquivo de relatório."""
    return REPORTS_PATH / filename
