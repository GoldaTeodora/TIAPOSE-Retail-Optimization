"""
Pipeline para prever clientes, otimizar lucro e salvar plano ótimo.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.profit_calculator import calculate_daily_profit

import pandas as pd
import numpy as np
from forecasting.ml_model import XGBoostForecaster
from core.config import STORES, XGBOOST_FEATURES
from optimization.methods import OptimizationMethods
from optimization.methods_global import OptimizationMethodsGlobal


def compute_daily_metrics(store_name, forecast, J, X, PR):

    results = []

    for d in range(7):

        clientes = int(round(forecast[d]))

        j = int(J[d])
        x = int(X[d])
        pr = float(PR[d])

        is_weekend = (d == 0) or (d == 6)

        daily_profit, total_units, total_hr = calculate_daily_profit(
            num_customers=clientes,
            J=j,
            X=x,
            PR=pr,
            is_weekend=is_weekend,
            store_name=store_name
        )

        attended = min(7 * x + 6 * j, clientes)

        results.append({
            'Dia': d + 1,
            'Clientes': clientes,
            'Atendidos': attended,
            'Juniores': j,
            'Experts': x,
            'Promocao': round(pr, 2),
            'Unidades': total_units,
            'Custo_RH': total_hr,
            'Lucro': daily_profit
        })

    return pd.DataFrame(results)

def generate_forecasts(stores, forecast_horizon=7):
    forecasts = {}

    for store in stores:
        enriched_path = Path(f'data/enriched/{store}_features.csv')
        df = pd.read_csv(enriched_path)

        X = df[[f for f in XGBOOST_FEATURES if f in df.columns]].ffill().bfill().fillna(0)
        y = df['Num_Customers'] if 'Num_Customers' in df.columns else df['Num_Customers_Clean']

        split_idx = int(len(df) * 0.85)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]

        xgb = XGBoostForecaster(store)
        xgb.train(X_train, y.iloc[:split_idx])

        forecast = xgb.predict(X_test.iloc[:forecast_horizon], n_periods=forecast_horizon)

        forecasts[store] = np.maximum(
            np.round(np.array(forecast).flatten()[:7]),
            0
        ).astype(int)

    return forecasts



def optimize_store_profit(store_name, forecast, objective='O1', n_runs=20, use_real=False):
    """
    Para uma loja, prevê clientes com XGBoost e otimiza o lucro semanal.
    """

    # =============================
    # 1. PREVISÃO (XGBoost)
    # =============================
    

    if use_real:
        print(" Using REAL data (baseline)")

        # ainda precisas de y aqui
        enriched_path = Path(f'data/enriched/{store_name}_features.csv')
        df = pd.read_csv(enriched_path)
        y = df['Num_Customers'] if 'Num_Customers' in df.columns else df['Num_Customers_Clean']

        forecast = np.maximum(
            np.round(y.iloc[-7:].values),
            0
        ).astype(int)

    else:
        print("Using FORECAST data")
        #  usa forecast já calculado
        forecast = np.maximum(
            np.round(forecast).astype(int).flatten()[:7],
            0
        )

    # =============================
    # 2. PR SCENARIOS
    # =============================
    PR_scenarios = {
        'coarse': np.array([0, 0.1, 0.2, 0.3]),
        'fine': np.arange(0, 0.31, 0.05)
    }

    comparison_results = []
    best_result = None

    # =============================
    # 3. OTIMIZAÇÃO
    # =============================
    methods = ['random', 'hill_climbing', 'simulated_annealing', 'genetic', 'pso', 'de']

    for name, pr_values in PR_scenarios.items():
        print(f"\n=== PR SCENARIO: {name} ===")
        print("\n--- COMPARAÇÃO DE ALGORITMOS ---")

        scenario_results = []
        all_histories = []

        for method in methods:

            values = []
            histories = []

            for seed in range(n_runs):
                optimizer = OptimizationMethods(
                    store_name,
                    forecast,
                    objective=objective,
                    method=method,
                    seed=seed
                )
                optimizer.PR_values = pr_values

                result = optimizer.optimize()

                values.append(result['best_value'])

                #  garantir que history existe
                if 'history' in result:
                    histories.append(result['history'])

            mean = np.mean(values)
            std = np.std(values)

            print(f"{method}: {mean:.2f} (±{std:.2f})")

            comparison_results.append({
                'store': store_name,
                'objective': objective,
                'pr_type': name,
                'algorithm': method,
                'mean_profit': mean,
                'std_profit': std
            })

            scenario_results.append({
                'method': method,
                'mean': mean
            })

            # =============================
            # CONVERGÊNCIA (se existir)
            # =============================
            valid_histories = [
                h for h in histories
                if len(h) > 0
            ]

            if len(valid_histories) > 0:

                min_len = min(len(h) for h in valid_histories)

                valid_histories = [
                    h[:min_len]
                    for h in valid_histories
                ]

                mean_history = np.mean(valid_histories, axis=0)

                all_histories.append((method, mean_history))

        # =============================
        # GRÁFICO DE CONVERGÊNCIA
        # =============================
        if len(all_histories) > 0:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(8,5))

            for method_name, history in all_histories:
                plt.plot(history, label=method_name)

            plt.xlabel("Iterações")
            plt.ylabel("Melhor valor")
            plt.title(f"Convergência ({store_name} - {objective} - {name})")
            plt.legend()
            plt.grid()

            Path("reports").mkdir(exist_ok=True)
            plt.savefig(f"reports/convergence_{store_name}_{objective}_{name}.png")
            plt.close()

        # =============================
        # MELHOR MÉTODO
        # =============================
        best_method = max(scenario_results, key=lambda x: x['mean'])['method']

        best_run_value = -np.inf
        best_run_result = None

        for seed in range(n_runs):
            optimizer = OptimizationMethods(
                store_name,
                forecast,
                objective=objective,
                method=best_method,
                seed=seed
            )
            optimizer.PR_values = pr_values

            res = optimizer.optimize()

            if res['best_value'] > best_run_value:
                best_run_value = res['best_value']
                best_run_result = res

        result = best_run_result

        if best_result is None or result['best_value'] > best_result['best_value']:
            best_result = result.copy()
            best_result['method'] = best_method

    # =============================
    # 4. GUARDAR RESULTADOS
    # =============================
    Path('reports').mkdir(exist_ok=True)

    results_df = pd.DataFrame(comparison_results)

    summary = results_df.groupby('algorithm').agg(
        mean_profit=('mean_profit', 'mean'),
        std_profit=('mean_profit', 'std')
    ).reset_index()

    summary = summary.sort_values(by='mean_profit', ascending=False)

    summary_pr = results_df.groupby(['algorithm', 'pr_type'])['mean_profit'].mean().reset_index()

    results_df.to_csv(f'reports/comparison_{store_name}_{objective}_{n_runs}runs.csv', index=False)
    summary.to_csv(f'reports/summary_{store_name}_{objective}_{n_runs}runs.csv', index=False)
    summary_pr.to_csv(f'reports/summary_pr_{store_name}_{objective}_{n_runs}runs.csv', index=False)

    print("\n=== MÉDIA POR ALGORITMO ===")
    print(summary)

    # =============================
    # 5. PLANO FINAL
    # =============================
    if best_result is None:
        raise RuntimeError("Nenhuma solução encontrada.")

    solution = best_result['solution']

    J = np.round(solution[0:7]).astype(int)
    X = np.round(solution[7:14]).astype(int)
    PR = solution[14:21]

    plan_df = compute_daily_metrics(
        store_name,
        forecast,
        J,
        X,
        PR
)

    out_path = Path(f'reports/plan_{store_name}_{objective}_{n_runs}runs.csv')
    plan_df.to_csv(out_path, index=False)

    print(f"\n Melhor método local: {best_result.get('method', 'N/A')} ({best_result['best_value']:.2f})")
    print(f"[OK] Plano ótimo salvo em: {out_path}")

    return plan_df, best_result



def optimize_global_profit(
    stores,
    forecasts,
    objective='O2',
    n_runs=20
):

      
    # =============================
    # 2. MÉTODOS
    # =============================
    if objective == 'O3_NS':

        methods = ['nsga2']

    elif objective == 'O3_WEIGHTED':

        methods = [
            'genetic',
            'pso',
            'de'
        ]

    else:

        methods = [
            'random',
            'hill_climbing',
            'simulated_annealing',
            'genetic',
            'pso',
            'de'
        ]

    
    comparison_results = []
    scenario_results = []
    all_histories = []

    print("\n=== GLOBAL OPTIMIZATION ===")
    print("\n--- COMPARAÇÃO DE ALGORITMOS ---")

    # =============================
    # 3. COMPARAÇÃO MULTI-RUN
    # =============================
    for method in methods:

        values = []
        histories = []

        for seed in range(n_runs):
            optimizer = OptimizationMethodsGlobal(
                stores=stores,
                forecasts=forecasts,
                objective=objective,
                seed=seed
            )

            result = optimizer.optimize(method)
            values.append(result['best_value'])

            if 'history' in result:
                histories.append(result['history'])

        values = np.array(values)

        mean = np.mean(values)
        std = np.std(values)
        best = np.max(values)
        worst = np.min(values)

        print(f"{method}: {mean:.2f} (±{std:.2f}) | best={best:.2f}, worst={worst:.2f}")

        comparison_results.append({
            'objective': objective,
            'algorithm': method,
            'mean_profit': mean,
            'std_profit': std,
            'best_profit': best,
            'worst_profit': worst
        })

        scenario_results.append({
            'method': method,
            'mean': mean
        })

        

        valid_histories = [
            h for h in histories
            if len(h) > 0
        ]

        

        if len(valid_histories) > 0:

            min_len = min(len(h) for h in valid_histories)

            valid_histories = [
                h[:min_len]
                for h in valid_histories
            ]

            mean_history = np.mean(
                valid_histories,
                axis=0
            )

            all_histories.append(
                (method, mean_history)
            )


    # =============================
    # GRÁFICO DE CONVERGÊNCIA GLOBAL
    # =============================
    if len(all_histories) > 0:

        import matplotlib.pyplot as plt

        plt.figure(figsize=(8,5))

        for method_name, history in all_histories:

            plt.plot(
                history,
                label=method_name
            )

        plt.xlabel("Iterações")
        plt.ylabel("Melhor valor")
        plt.title(f"Convergência Global ({objective})")

        plt.legend()
        plt.grid()

        Path("reports").mkdir(exist_ok=True)

        plt.savefig(
            f"reports/global_convergence_{objective}.png"
        )

        plt.close()

    # =============================
    # 4. ESCOLHER MELHOR MÉTODO
    # =============================
    best_method = max(scenario_results, key=lambda x: x['mean'])['method']

    print(f"\n Melhor método global: {best_method}")

    # =============================
    # 5. MELHOR EXECUÇÃO FINAL
    # =============================
    best_value = -np.inf
    best_result = None
    best_optimizer = None

    for seed in range(n_runs):
        optimizer = OptimizationMethodsGlobal(
            stores=stores,
            forecasts=forecasts,
            objective=objective,
            seed=seed
        )

        result = optimizer.optimize(best_method)

        if result['best_value'] > best_value:
            best_value = result['best_value']
            best_result = result
            best_optimizer = optimizer

    if best_result is None:
        raise RuntimeError("Nenhuma solução global encontrada.")

    # =============================
    # 6. GUARDAR COMPARAÇÃO
    # =============================
    Path('reports').mkdir(exist_ok=True)

    results_df = pd.DataFrame(comparison_results)
    results_df = results_df.sort_values(by='mean_profit', ascending=False)

    results_df.to_csv(
        f'reports/global_comparison_{objective}_{n_runs}runs.csv',
        index=False
    )

    print("\n=== RESUMO GLOBAL ===")
    print(results_df)

    summary = results_df.groupby('algorithm').agg(
        mean_profit=('mean_profit', 'mean'),
        std_profit=('mean_profit', 'std')
    ).reset_index()
    summary = summary.sort_values(by='mean_profit', ascending=False)

    
    summary.to_csv(
        f'reports/global_summary_{objective}_{n_runs}runs.csv',
        index=False
    )

    # =============================
    # 7. GERAR PLANOS POR LOJA
    # =============================
    solution = best_optimizer._repair_solution(
        best_optimizer._fix_solution(
            best_result['solution']
    )
).copy()    

    J_all, X_all, PR_all = best_optimizer._split_solution(solution)

    for i, store in enumerate(stores):

        start = i * 7

        J = J_all[start:start+7].astype(int)

        X = X_all[start:start+7].astype(int)

        PR = PR_all[start:start+7]

        plan_df = compute_daily_metrics(
            store,
            forecasts[store],
            J,
            X,
            PR
        )

        plan_df.to_csv(
            f'reports/global_plan_{store}_{objective}_{n_runs}runs.csv',
            index=False
        )

    print(f"\n Melhor solução global ({objective}): {best_result['best_value']:.2f}")

    if objective == 'O3_NS' and 'pareto_solutions' in best_result:

        pareto_df = pd.DataFrame([
            {
                'profit': s['profit'],
                'hr': s['hr']
            }
            for s in best_result['pareto_solutions']
        ])

        pareto_df.to_csv(
            f'reports/pareto_front_{objective}_{n_runs}runs.csv',
            index=False
        )

        print("\nPareto front guardado.")

        import matplotlib.pyplot as plt

        plt.figure(figsize=(7,5))

        plt.scatter(
            pareto_df['hr'],
            pareto_df['profit']
        )

        plt.xlabel("HR Cost")
        plt.ylabel("Profit")
        plt.title("Pareto Front")

        plt.grid()

        plt.savefig(
            f'reports/pareto_front_{objective}_{n_runs}runs.png'
        )

        plt.close()

    return best_result


if __name__ == '__main__':

    np.random.seed(42)

    forecast_df = pd.read_csv(
        'reports/future_forecast_7_days.csv'
    )

    print("\n===== FORECASTS USADOS NA OTIMIZAÇÃO =====")

    for store in STORES:

        store_data = forecast_df[
            forecast_df['Loja'].str.lower() == store.lower()
    ]

        forecast_values = (
            store_data['Clientes_Previstos']
            .values[:7]
            .astype(int)
        )

        print(store)
        print(forecast_values)

    forecast_df['Loja'] = (
        forecast_df['Loja']
        .str.lower()
)

    forecasts = {}

    for store in STORES:

        store_data = forecast_df[
            forecast_df['Loja'] == store
        ]

        forecasts[store] = (
            store_data['Clientes_Previstos']
            .values[:7]
            .astype(int)
        )

    n_runs = 20

    print(f"\n============================")
    print(f" RUNS = {n_runs}")
    print(f"============================")

    # GLOBAL O2 only
    print(f"\n--- GLOBAL O2 ---")
    optimize_global_profit(
        STORES,
        forecasts,
        objective='O2',
        n_runs=n_runs
    )