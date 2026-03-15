import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from metrics import calculate_metrics

def run_ets(store_name):
    df = pd.read_csv(f'data/processed/{store_name}_clean.csv')
    train = df['Num_Customers'].iloc[:-90]
    test = df['Num_Customers'].iloc[-90:]
    
    # Modelagem Triple Exponential Smoothing
    model = ExponentialSmoothing(train, seasonal='add', seasonal_periods=7).fit()
    preds = model.forecast(90)
    
    m = calculate_metrics(test, preds)
    print(f"ETS {store_name}: MAE {m['MAE']:.2f}")
    return m

for s in ['baltimore', 'lancaster', 'philadelphia', 'richmond']:
    run_ets(s)