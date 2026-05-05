import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from src.metrics import calculate_metrics


def backtest_ets(store_name, n_tests=500, horizon=7):
    df = pd.read_csv(f'data/processed/{store_name}_clean.csv')

    # garantir ordem temporal
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    results = []

    for i in range(n_tests):
        train_end = len(df) - horizon - i
        if train_end <= 30:
            break

        train = df['Num_Customers'].iloc[:train_end]
        test = df['Num_Customers'].iloc[train_end:train_end + horizon]

        try:
            model = ExponentialSmoothing(
                train,
                trend='add',
                seasonal='add',
                seasonal_periods=7
            ).fit()

            preds = model.forecast(horizon)

            metrics = calculate_metrics(test, preds)
            results.append(metrics)

        except Exception:
            continue

    results_df = pd.DataFrame(results)

    print(f"\n===== ETS: {store_name.upper()} =====")
    print(results_df.mean())

    # guardar resultados médios
    results_df.mean().to_csv(f'reports/ets_{store_name}.csv')


def run():
    stores = ['baltimore', 'lancaster', 'philadelphia', 'richmond']

    for store in stores:
        backtest_ets(store)


if __name__ == "__main__":
    run()