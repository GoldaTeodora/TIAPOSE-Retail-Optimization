import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from src.metrics import calculate_metrics


# FEATURES (já criadas no preprocessing)
FEATURES = [
    'Day_of_Week', 'Is_Weekend', 'Month', 'Quarter', 'Day_of_Year',
    'Is_Black_Friday', 'Is_Tourist_Event', 'Is_Holiday', 'Is_Memorial_Day',
    'Event_Nearby', 'Holiday_Nearby',
    'Lag_Customers_7', 'Lag_Customers_14',
    'Rolling_Mean_7', 'Rolling_Std_7', 'Rolling_Mean_30'
]


def backtest_xgboost(store_name, n_tests=150, horizon=7):
    df = pd.read_csv(f'data/processed/{store_name}_clean.csv').dropna()

    # garantir ordem temporal
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    results = []

    for i in range(n_tests):
        train_end = len(df) - horizon - i
        if train_end <= 50:
            break

        train_df = df.iloc[:train_end]
        test_df = df.iloc[train_end:train_end + horizon]

        try:
            X_train = train_df[FEATURES]
            y_train = train_df['Num_Customers']

            X_test = test_df[FEATURES]
            y_test = test_df['Num_Customers']

            model = XGBRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )

            model.fit(X_train, y_train)

            preds = model.predict(X_test)

            metrics = calculate_metrics(y_test, preds)
            results.append(metrics)

        except Exception:
            continue

    results_df = pd.DataFrame(results)

    print(f"\n===== XGBOOST: {store_name.upper()} =====")
    print(results_df.mean())

    # guardar resultados
    results_df.mean().to_csv(f'reports/xgb_{store_name}.csv')


def run():
    stores = ['baltimore', 'lancaster', 'philadelphia', 'richmond']

    for store in stores:
        backtest_xgboost(store)


if __name__ == "__main__":
    run()