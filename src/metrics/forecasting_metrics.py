import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def calculate_all_metrics(y_true, y_pred):
    """Calcula todas as métricas principais de forecasting."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) if mask.any() else np.inf
    y_range = np.max(y_true) - np.min(y_true)
    nmae = mae / y_range if y_range > 0 else 0
    r2 = r2_score(y_true, y_pred)
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'NMAE': nmae,
        'R2': r2
    }
