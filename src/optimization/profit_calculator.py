import numpy as np

def calculate_profit(num_customers, J, X, PR, is_weekend, store_params):
    # 1. CUSTOS DE RH (Regra 7.4 - Página 11) 
    cost_j = 70 if is_weekend else 60
    cost_x = 95 if is_weekend else 80
    
    # 2. CAPACIDADE (Regra 7.1 - Página 11) [cite: 40]
    cap_x = 7 * X
    cap_j = 6 * J
    A = min(cap_x + cap_j, num_customers)
    
    # 3. VENDAS E RECEITA (Regra 7.2 - Página 12) 
    # "first all X workers are used, then if needed, the remainder J workers are used" [cite: 42]
    total_revenue = 0
    rem_customers = A
    
    # Atendimento por Experts
    served_x = min(rem_customers, cap_x)
    if served_x > 0:
        # U = round(F * 10 / ln(2 - PR)) [cite: 46]
        u_x = round((store_params['F_x'] * 10) / np.log(2 - PR))
        # P = round(U * (1 - PR) * 1.07) [cite: 47]
        p_x = round(u_x * (1 - PR) * 1.07)
        total_revenue += served_x * p_x
        rem_customers -= served_x
        
    # Atendimento por Juniores
    served_j = min(rem_customers, cap_j)
    if served_j > 0:
        u_j = round((store_params['F_j'] * 10) / np.log(2 - PR))
        p_j = round(u_j * (1 - PR) * 1.07)
        total_revenue += served_j * p_j

    # 4. LUCRO FINAL (Página 12) [cite: 47]
    hr_costs = (J * cost_j) + (X * cost_x)
    fixed_costs = store_params['W_s'] # W_s é subtraído do lucro semanal [cite: 47]
    
    # Retorno diário antes do custo fixo
    daily_return = total_revenue - hr_costs
    return daily_return, fixed_costs