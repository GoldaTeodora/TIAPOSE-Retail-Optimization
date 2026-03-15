import pandas as pd
from metrics import calculate_metrics

def run_baselines(store_name):
    df = pd.read_csv(f'data/processed/{store_name}_clean.csv')
    
    # 6.1 Split: Últimos 90 dias para teste
    test_days = 90
    train = df.iloc[:-test_days]
    test = df.iloc[-test_days:]
    
    results = {}
    
    # A) Naive (Último valor disponível)
    y_pred_naive = [train['Num_Customers'].iloc[-1]] * test_days
    results['Naive'] = calculate_metrics(test['Num_Customers'], y_pred_naive)
    
    # B) Seasonal Naive (Mesmo dia da semana passada)
    y_pred_s_naive = df['Lag_Customers_7'].iloc[-test_days:]
    results['S_Naive'] = calculate_metrics(test['Num_Customers'], y_pred_s_naive)
    
    # C) Moving Average (Média dos últimos 7 dias)
    y_pred_ma = df['Rolling_Mean_7'].iloc[-test_days:]
    results['MA_7'] = calculate_metrics(test['Num_Customers'], y_pred_ma)
    
    print(f"\n--- BASELINES: {store_name.upper()} ---")
    for m, m_val in results.items():
        print(f"{m}: MAE={m_val['MAE']:.2f}, NMAE={m_val['NMAE']:.4f}")
        
    return results

# Executar para todas as lojas
for s in ['baltimore', 'lancaster', 'philadelphia', 'richmond']:
    run_baselines(s)