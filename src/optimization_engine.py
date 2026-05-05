import pickle
import pandas as pd
import numpy as np
import os
from optimization.profit_calculator import calculate_profit


STORE_PARAMS = {
    'baltimore':    {'F_j': 1.00, 'F_x': 1.15, 'W_s': 700},
    'lancaster':    {'F_j': 1.05, 'F_x': 1.20, 'W_s': 730},
    'philadelphia': {'F_j': 1.10, 'F_x': 1.15, 'W_s': 760},
    'richmond':     {'F_j': 1.15, 'F_x': 1.25, 'W_s': 800}
}


def hill_climbing(num_customers, is_weekend, store_name, scenario="O1"):

    best_j, best_x, best_pr = 0, 1, 0.0

    best_profit, best_units = calculate_profit(
        num_customers, best_j, best_x, best_pr,
        is_weekend, STORE_PARAMS[store_name]
    )

    moves = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,0.05), (0,0,-0.05)]

    for _ in range(100):
        improved = False

        for dj, dx, dpr in moves:
            nj = max(0, min(20, best_j + dj))
            nx = max(1, min(20, best_x + dx))
            npr = max(0.0, min(0.3, round(best_pr + dpr, 2)))

            profit, units = calculate_profit(
                num_customers, nj, nx, npr,
                is_weekend, STORE_PARAMS[store_name]
            )

            # -----------------------------
            # CENÁRIOS
            # -----------------------------
            if scenario == "O2":
                # ⚠️ limite alto (pode não ativar — intencional)
                if units > 10000:
                    profit = -1e9

            elif scenario == "O3":
                penalty = 10 * (nj + nx)
                profit = profit - penalty

            # -----------------------------
            if profit > best_profit:
                best_profit = profit
                best_j, best_x, best_pr = nj, nx, npr
                improved = True

        if not improved:
            break

    return best_j, best_x, best_pr, best_profit


def generate_plan():

    features = [
        'Day_of_Week', 'Is_Weekend', 'Month', 'Quarter', 'Day_of_Year',
        'Is_Black_Friday', 'Is_Tourist_Event', 'Is_Holiday', 'Is_Memorial_Day',
        'Event_Nearby', 'Holiday_Nearby',
        'Lag_Customers_7', 'Lag_Customers_14',
        'Rolling_Mean_7', 'Rolling_Std_7', 'Rolling_Mean_30'
    ]

    scenarios = ["O1", "O2", "O3"]

    results = []

    for s in STORE_PARAMS.keys():
        df = pd.read_csv(f'data/processed/{s}_clean.csv').dropna().tail(7)

        for scenario in scenarios:
            for h in range(1, 8):

                with open(f'models/{s}/xgb_h{h}.pkl', 'rb') as f:
                    model = pickle.load(f)

                row = df.iloc[h-1]

                pred_log = model.predict(pd.DataFrame([row[features]]))[0]
                num_c = np.expm1(pred_log)

                j, x, pr, profit = hill_climbing(
                    num_c,
                    row['Is_Weekend'] == 1,
                    s,
                    scenario
                )

                results.append([
                    s,
                    row['Date'],
                    scenario,
                    int(num_c),
                    j,
                    x,
                    pr,
                    round(profit, 2)
                ])

    return pd.DataFrame(results, columns=[
        'Loja', 'Data', 'Cenario',
        'Pred_Clientes', 'J', 'X', 'PR', 'Lucro'
    ])


if __name__ == "__main__":

    final_plan = generate_plan()

    print("\n--- PLANO OTIMIZADO (TODOS OS CENÁRIOS) ---")
    print(final_plan.to_string(index=False))

    os.makedirs('reports', exist_ok=True)
    final_plan.to_csv('reports/weekly_plan_scenarios.csv', index=False)