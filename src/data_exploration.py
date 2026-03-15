import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuração visual
sns.set_theme(style="whitegrid")

def run_eda(store_name):
    df = pd.read_csv(f'data/processed/{store_name}_clean.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Criar pasta para as figuras se não existir (conforme o roadmap)
    os.makedirs('reports/figures', exist_ok=True)
    
    # 1. Gráfico de Série Temporal (Geral)
    plt.figure(figsize=(15, 6))
    sns.lineplot(data=df, x='Date', y='Num_Customers')
    plt.title(f'Evolução Diária de Clientes - {store_name.capitalize()}')
    plt.savefig(f'reports/figures/{store_name}_timeseries.png')
    plt.close()

    # 2. Impacto do Dia da Semana (Sazonalidade)
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='Day_of_Week', y='Num_Customers')
    plt.title(f'Distribuição de Clientes por Dia da Semana - {store_name.capitalize()}')
    plt.xticks(range(7), ['Seg', 'Ter', 'Qua', 'Qui', 'Sex', 'Sáb', 'Dom'])
    plt.savefig(f'reports/figures/{store_name}_weekly_seasonality.png')
    plt.close()

    # 3. Análise de Correlação
    plt.figure(figsize=(8, 6))
    corr = df[['Num_Customers', 'Sales', 'Pct_On_Sale', 'Is_Weekend', 'Is_Black_Friday']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title(f'Matriz de Correlação - {store_name.capitalize()}')
    plt.savefig(f'reports/figures/{store_name}_correlation.png')
    plt.close()

    print(f"Análise concluída para {store_name}. Gráficos guardados em reports/figures/")

# Executar para todas as lojas
for store in ['baltimore', 'lancaster', 'philadelphia', 'richmond']:
    run_eda(store)