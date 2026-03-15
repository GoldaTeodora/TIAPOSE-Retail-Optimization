import pandas as pd
from pmdarima import auto_arima
from metrics import calculate_metrics

def run_arima(store_name):
    df = pd.read_csv(f'data/processed/{store_name}_clean.csv')
    # Split 90 dias
    train = df['Num_Customers'].iloc[:-90]
    test = df['Num_Customers'].iloc[-90:]
    
    # O auto_arima testa várias combinações sozinho
    model = auto_arima(train, seasonal=True, m=7, suppress_warnings=True)
    preds = model.predict(n_periods=90)
    
    m = calculate_metrics(test, preds)
    print(f"ARIMA {store_name}: MAE {m['MAE']:.2f}")
    return m

# Corre para as 4 lojas
for s in ['baltimore', 'lancaster', 'philadelphia', 'richmond']:
    run_arima(s)