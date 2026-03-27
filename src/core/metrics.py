"""
Funções de cálculo de métricas para avaliação de forecasting.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def calculate_metrics(y_true, y_pred):
    """Calcula todas as métricas de forecasting.
    
    Args:
        y_true: array com valores reais
        y_pred: array com previsões
    
    Returns:
        Dict com métricas: MAE, RMSE, MAPE, NMAE, R²
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # MAPE (Mean Absolute Percentage Error)
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) if mask.any() else np.inf
    
    # NMAE (Normalized MAE) - normalizado pela amplitude
    y_range = np.max(y_true) - np.min(y_true)
    nmae = mae / y_range if y_range > 0 else 0
    
    # R² Score
    r2 = r2_score(y_true, y_pred)
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'NMAE': nmae,
        'R2': r2
    }

def aggregate_metrics(list_of_metrics):
    """Agrega métricas de múltiplas iterações.
    
    Args:
        list_of_metrics: lista de dicts com métricas
    
    Returns:
        Dict com mean e median de cada métrica
    """
    if not list_of_metrics:
        return {}
    
    # Converter para formato que permite agregar
    metric_names = list_of_metrics[0].keys()
    aggregated = {}
    
    for metric in metric_names:
        values = [m[metric] for m in list_of_metrics if not np.isnan(m[metric]) and not np.isinf(m[metric])]
        
        if values:
            aggregated[f'{metric}_mean'] = np.mean(values)
            aggregated[f'{metric}_median'] = np.median(values)
            aggregated[f'{metric}_std'] = np.std(values)
        else:
            aggregated[f'{metric}_mean'] = np.nan
            aggregated[f'{metric}_median'] = np.nan
            aggregated[f'{metric}_std'] = np.nan
    
    return aggregated

def print_metrics(metrics, label=""):
    """Imprime métricas de forma legível."""
    if label:
        print(f"\n{label}")
        print("=" * 60)
    
    for key, value in metrics.items():
        if isinstance(value, float):
            if 'MAPE' in key or 'NMAE' in key or 'R2' in key:
                print(f"  {key:20s}: {value:+.4f}")
            else:
                print(f"  {key:20s}: {value:.2f}")
        else:
            print(f"  {key:20s}: {value}")
