import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    nmae = mae / np.mean(y_true)
    
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'NMAE': nmae}