import pandas as pd
import numpy as np
import pickle
import os
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from metrics import calculate_metrics

def train_pro_model(store_name):
    df = pd.read_csv(f'data/processed/{store_name}_clean.csv').dropna()
    
    # Lista de features expandida para o 20
    features = [
        'Day_of_Week', 'Is_Weekend', 'Month', 'Quarter', 'Day_of_Year',
        'Is_Black_Friday', 'Is_Tourist_Event', 'Is_Holiday', 'Is_Memorial_Day',
        'Event_Nearby', 'Holiday_Nearby',
        'Lag_Customers_7', 'Lag_Customers_14', 
        'Rolling_Mean_7', 'Rolling_Std_7', 'Rolling_Mean_30'
    ]
    
    test_days = 90
    train_df = df.iloc[:-test_days]
    test_df = df.iloc[-test_days:]
    
    param_grid = {
        'n_estimators': [500, 1000],
        'max_depth': [3, 6, 8],
        'learning_rate': [0.01, 0.05],
        'subsample': [0.8, 0.9],
        'colsample_bytree': [0.8, 0.9]
    }

    horizon_results = []
    print(f"\n>>> TREINANDO {store_name.upper()} COM TRATAMENTO DE EVENTOS...")

    for h in range(1, 8):
        y_train_raw = train_df['Num_Customers'].shift(-h).dropna()
        y_train_log = np.log1p(y_train_raw)
        X_train = train_df[features].iloc[:len(y_train_log)]
        
        y_test_raw = test_df['Num_Customers'].shift(-h).dropna()
        X_test = test_df[features].iloc[:len(y_test_raw)]

        xgb = XGBRegressor(random_state=42, objective='reg:squarederror')
        search = RandomizedSearchCV(xgb, param_grid, n_iter=8, cv=3, 
                                    scoring='neg_mean_absolute_error', n_jobs=-1)
        search.fit(X_train, y_train_log)
        
        best_model = search.best_estimator_
        preds_log = best_model.predict(X_test)
        preds_final = np.expm1(preds_log) 
        
        m = calculate_metrics(y_test_raw, preds_final)
        m['horizon'] = h
        horizon_results.append(m)
        
        os.makedirs(f'models/{store_name}', exist_ok=True)
        with open(f'models/{store_name}/xgb_h{h}.pkl', 'wb') as f:
            pickle.dump(best_model, f)

    res_df = pd.DataFrame(horizon_results)
    print(f"DONE: MAE Médio {res_df['MAE'].mean():.2f} | NMAE {res_df['NMAE'].mean():.4f}")
    return res_df

for s in ['baltimore', 'lancaster', 'philadelphia', 'richmond']:
    train_pro_model(s)