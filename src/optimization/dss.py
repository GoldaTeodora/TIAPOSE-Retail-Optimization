"""
Sistema de Suporte à Decisão (DSS) integrado.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from optimization.methods import OptimizationMethods
from forecasting.arima_model import ARIMAForecaster
from forecasting.ets_model import ETSForecaster
from forecasting.ml_model import XGBoostForecaster
from core.profit_calculator import calculate_daily_profit, calculate_weekly_profit
from core.config import get_data_path, get_report_path, get_model_path, XGBOOST_FEATURES, STORES, STORE_PARAMS


class DSSLoja:
    """DSS para uma loja específica."""
    
    def __init__(self, store_name):
        """Inicializa DSS para loja.
        
        Args:
            store_name: Nome da loja
        """
        self.store_name = store_name
        self.df = None
        self.load_data()
    
    def load_data(self):
        """Carrega dados históricos."""
        path = get_data_path(self.store_name, raw=False)
        self.df = pd.read_csv(path)
    
    def forecast_next_week(self):
        """Prevê clientes para os próximos 7 dias.
        
        Returns:
            Array com 7 previsões
        """
        y = self.df['Num_Customers'].values
        X = self.df[XGBOOST_FEATURES].copy()
        
        # Pegar últimos dados possíveis como se fosse "hoje"
        # Usar XGBoost como referência (melhor performance)
        
        try:
            xgb = XGBoostForecaster(self.store_name)
            xgb.train(X.iloc[:-7], y[:-7])
            
            # Prever os últimos 7 dias (são os "próximos")
            forecast = xgb.predict(X.iloc[-7:], n_periods=7)
            return forecast
        except Exception as e:
            print(f"  Aviso: Falha em XGBoost, usando ARIMA: {e}")
            
            try:
                arima = ARIMAForecaster(self.store_name)
                arima.train(None, y[:-7])
                forecast = arima.predict(n_periods=7)
                return forecast
            except Exception as e2:
                print(f"  Aviso: Falha em ARIMA, usando média: {e2}")
                return np.full(7, np.mean(y[-14:]))
    
    def get_next_week_info(self):
        """Informação sobre a próxima semana (datas, fins de semana, etc).
        
        Returns:
            DataFrame com info de cada dia
        """
        # Simular próxima semana começando segunda-feira
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        dates = [f"Day {i+1}" for i in range(7)]
        
        info = pd.DataFrame({
            'Day': days,
            'Date': dates,
            'IsWeekend': [False]*5 + [True]*2
        })
        
        return info
    
    def generate_plan(self, objective='O1'):
        """Gera plano otimizado para próxima semana.
        
        Args:
            objective: 'O1', 'O2', ou 'O3'
        
        Returns:
            Dict com 'plan_df', 'summary', 'filepath'
        """
        print(f"\n  Gerando plano para {self.store_name} - {objective}")
        
        # Prever clientes
        forecast = self.forecast_next_week()
        
        # Otimizar
        optimizer = OptimizationMethods(self.store_name, forecast, objective)
        result = optimizer.optimize()
        
        # Extrair solução
        J = result['solution'][0:7]
        X = result['solution'][7:14]
        PR = result['solution'][14:21]
        
        # Montar plano
        info = self.get_next_week_info()
        info['Clientes'] = forecast.astype(int)
        info['J'] = np.clip(J, 0, 20).astype(int)
        info['X'] = np.clip(X, 0, 20).astype(int)
        info['PR'] = np.clip(PR, 0.0, 0.3)
        
        # Calcular lucro por dia
        daily_profits = []
        daily_units = []
        daily_hr = []
        
        for i in range(7):
            profit, units, hr = calculate_daily_profit(
                int(forecast[i]), int(info.loc[i, 'J']), int(info.loc[i, 'X']),
                info.loc[i, 'PR'], bool(info.loc[i, 'IsWeekend']), self.store_name
            )
            daily_profits.append(profit)
            daily_units.append(units)
            daily_hr.append(hr)
        
        info['Lucro'] = daily_profits
        info['Unidades'] = daily_units
        info['HR_Cost'] = daily_hr
        
        # Salvar
        filepath = get_report_path(f'plan_{self.store_name}_{objective}.csv')
        info.to_csv(filepath, index=False)
        
        # Summary
        total_profit = sum(daily_profits) - STORE_PARAMS[self.store_name]['W_s']
        total_units = sum(daily_units)
        total_hr = sum(daily_hr)
        
        summary = f"""
  [OK] Plano {objective} para {self.store_name.upper()}:
    - Lucro semanal: ${total_profit:,.2f}
    - Unidades totais: {total_units:,}
    - Custo HR: ${total_hr:,.2f}
    - Arquivo: {filepath}
        """
        
        return {
            'plan_df': info,
            'summary': summary,
            'filepath': filepath,
            'total_profit': total_profit,
            'total_units': total_units,
            'total_hr': total_hr
        }


class DSS:
    """Sistema de Suporte à Decisão (interface principal)."""
    
    def __init__(self):
        """Inicializa DSS."""
        pass
    
    def run_interactive(self):
        """Executa modo interativo com menu."""
        while True:
            print("\n" + "="*70)
            print("SISTEMA DE SUPORTE À DECISÃO (DSS)")
            print("="*70)
            print("\nOpções:")
            print("  1 - Gerar plano para uma loja")
            print("  2 - Gerar plano para todas as lojas")
            print("  3 - Comparar objetivos (O1, O2, O3)")
            print("  4 - Sair")
            
            choice = input("\nEscolha (1-4): ").strip()
            
            if choice == '1':
                self._option_single_store()
            elif choice == '2':
                self._option_all_stores()
            elif choice == '3':
                self._option_compare_objectives()
            elif choice == '4':
                print("\nSaindo...")
                break
            else:
                print("\nOpção inválida!")
    
    def _option_single_store(self):
        """Opção 1: Plano para uma loja."""
        print("\nLojas disponíveis:")
        for i, store in enumerate(STORES, 1):
            print(f"  {i} - {store.capitalize()}")
        
        choice = input(f"\nEscolha (1-{len(STORES)}): ").strip()
        
        try:
            store_idx = int(choice) - 1
            store = STORES[store_idx]
        except (ValueError, IndexError):
            print("Opção inválida!")
            return
        
        print("\nObjetivos:")
        print("  1 - O1 (Maximizar lucro)")
        print("  2 - O2 (Lucro com restrição de unidades)")
        print("  3 - O3 (Lucro + minimizar HR)")
        
        obj_choice = input("\nEscolha (1-3): ").strip()
        objectives = {'1': 'O1', '2': 'O2', '3': 'O3'}
        
        if obj_choice not in objectives:
            print("Opção inválida!")
            return
        
        objective = objectives[obj_choice]
        
        dss = DSSLoja(store)
        result = dss.generate_plan(objective)
        print(result['summary'])
        print("\n" + result['plan_df'].to_string(index=False))
    
    def _option_all_stores(self):
        """Opção 2: Plano para todas as lojas."""
        print("\nGerando planos para todos os stores com objetivo O1...")
        
        all_results = {}
        for store in STORES:
            dss = DSSLoja(store)
            result = dss.generate_plan('O1')
            all_results[store] = result
            print(result['summary'])
        
        # Resumo consolidado
        print("\n" + "="*70)
        print("RESUMO CONSOLIDADO")
        print("="*70)
        total_profit = sum(r['total_profit'] for r in all_results.values())
        total_units = sum(r['total_units'] for r in all_results.values())
        
        print(f"\nLucro total semanal: ${total_profit:,.2f}")
        print(f"Unidades totais: {total_units:,}")
    
    def _option_compare_objectives(self):
        """Opção 3: Comparar objetivos para uma loja."""
        print("\nLojas disponíveis:")
        for i, store in enumerate(STORES, 1):
            print(f"  {i} - {store.capitalize()}")
        
        choice = input(f"\nEscolha (1-{len(STORES)}): ").strip()
        
        try:
            store_idx = int(choice) - 1
            store = STORES[store_idx]
        except (ValueError, IndexError):
            print("Opção inválida!")
            return
        
        print(f"\nComparando objetivos para {store.upper()}...")
        
        dss = DSSLoja(store)
        results = {}
        
        for objective in ['O1', 'O2', 'O3']:
            result = dss.generate_plan(objective)
            results[objective] = result
        
        # Tabela comparativa
        print("\n" + "="*70)
        print(f"COMPARAÇÃO DE OBJETIVOS - {store.upper()}")
        print("="*70)
        print(f"\n{'Objetivo':<15} {'Lucro':<20} {'Unidades':<15}")
        print("-"*50)
        for objective, result in results.items():
            print(f"{objective:<15} ${result['total_profit']:>18,.2f} {result['total_units']:>14,}")
