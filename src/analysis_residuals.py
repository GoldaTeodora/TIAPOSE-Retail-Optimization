import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
import os

warnings.filterwarnings('ignore')

def plot_residuals_pro(store_name, horizon=1):
    df = pd.read_csv(f'data/processed/{store_name}_clean.csv').dropna()
    test_df = df.iloc[-90:].copy()
    
    model_path = f'models/{store_name}/xgb_h{horizon}.pkl'
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    except FileNotFoundError:
        return

    features = [
        'Day_of_Week', 'Is_Weekend', 'Month', 'Quarter', 'Day_of_Year',
        'Is_Black_Friday', 'Is_Tourist_Event', 'Is_Holiday', 'Is_Memorial_Day',
        'Event_Nearby', 'Holiday_Nearby',
        'Lag_Customers_7', 'Lag_Customers_14', 
        'Rolling_Mean_7', 'Rolling_Std_7', 'Rolling_Mean_30'
    ]
    
    preds_log = model.predict(test_df[features])
    preds = np.expm1(preds_log)
    
    test_df['Prediction'] = preds
    test_df['Error'] = test_df['Num_Customers'] - test_df['Prediction']
    
    # Gráficos
    os.makedirs('reports/figures', exist_ok=True)
    plt.figure(figsize=(10, 5))
    sns.scatterplot(data=test_df, x='Prediction', y='Error')
    plt.axhline(0, color='red', linestyle='--')
    plt.title(f'Resíduos Finais - {store_name.upper()}')
    plt.savefig(f'reports/figures/residuals_PRO_{store_name}.png')
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.plot(test_df['Date'].astype(str), test_df['Num_Customers'], label='Real', alpha=0.6)
    plt.plot(test_df['Date'].astype(str), test_df['Prediction'], label='Previsto', linestyle='--')
    plt.xticks(rotation=45, ticks=test_df['Date'].astype(str).iloc[::10])
    plt.title(f'Comparação Temporal - {store_name.upper()}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'reports/figures/time_comparison_{store_name}.png')
    plt.close()
    
    print(f"\nDIAGNÓSTICO: {store_name.upper()}")
    top_errors = test_df.reindex(test_df.Error.abs().sort_values(ascending=False).index).head(5)
    print(f"{'Data':<12} | {'Real':<6} | {'Previsto':<10} | {'Erro':<8}")
    print("-" * 40)
    for _, row in top_errors.iterrows():
        date_str = str(row['Date']).split(' ')[0]
        print(f"{date_str:<12} | {int(row['Num_Customers']):<6} | {int(row['Prediction']):<10} | {int(row['Error']):<8}")

if __name__ == "__main__":
    for s in ['baltimore', 'lancaster', 'philadelphia', 'richmond']:
        plot_residuals_pro(s)
    print(f"\n[OK] Gráficos atualizados em 'reports/figures/'")