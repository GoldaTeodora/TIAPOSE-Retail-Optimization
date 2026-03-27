"""
Cálculo de lucro diário e métricas para otimização.

Fórmula do projeto:
  - Clientes atendidos = min(7*X + 6*J, num_clientes)
  - Unidades/cliente = round(F * 10 / ln(2 - PR))
  - Lucro/cliente = round(unidades * (1 - PR) * 1.07)
  - Lucro diário = Lucro_total - (J*cost_j + X*cost_x)
  - Lucro semanal = sum(lucro diário) - W_s (custo fixo semanal)
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from core.config import STORE_PARAMS, HR_COSTS, CUSTOMER_CAPACITY


def calculate_daily_profit(num_customers, J, X, PR, is_weekend, store_name):
    """Calcula lucro de um dia específico.
    
    Args:
        num_customers: Número de clientes do dia
        J: Número de juniores
        X: Número de experts
        PR: Taxa de promoção (0.0 a 0.3)
        is_weekend: Boolean, True se fim de semana
        store_name: Nome da loja ('baltimore', 'lancaster', etc.)
    
    Returns:
        Tuple: (daily_profit, total_units, total_hr_cost)
    
    Raises:
        ValueError: Se parâmetros inválidos
    """
    # Validação
    if not 0 <= PR <= 0.3:
        raise ValueError(f"PR deve estar entre 0 e 0.3, recebido {PR}")
    if J < 0 or X < 0:
        raise ValueError(f"J e X devem ser positivos")
    if num_customers < 0:
        raise ValueError(f"num_customers deve ser positivo")
    
    store = STORE_PARAMS[store_name]
    hr = HR_COSTS['weekend' if is_weekend else 'weekday']
    
    # Clientes atendidos (prioridade: Experts, depois Juniores)
    expert_capacity = X * CUSTOMER_CAPACITY['X']
    junior_capacity = J * CUSTOMER_CAPACITY['J']
    attended = min(expert_capacity + junior_capacity, num_customers)
    
    # Distribuir clientes entre experts e juniores
    if attended <= expert_capacity:
        x_customers = attended
        j_customers = 0
    else:
        x_customers = expert_capacity
        j_customers = attended - expert_capacity
    
    # Unidades por cliente (fórmula do projeto)
    # PR=0 -> unidades = F*10 / ln(2) ≈ F*14.4
    # PR=0.3 -> unidades = F*10 / ln(1.7) ≈ F*21.4
    denominator = np.log(2.0 - PR) if (2.0 - PR) > 0 else 1e-6
    units_per_customer = np.round(store['F_x'] * 10 / denominator)
    
    # Lucro por cliente: units * (1 - PR) * 1.07
    profit_per_customer = np.round(units_per_customer * (1 - PR) * 1.07)
    
    # Lucro total do dia (só do que foi atendido)
    total_profit_revenue = (x_customers + j_customers) * profit_per_customer
    
    # Custo de RH
    j_cost = J * hr['J']
    x_cost = X * hr['X']
    total_hr_cost = j_cost + x_cost
    
    # Lucro líquido do dia
    daily_profit = total_profit_revenue - total_hr_cost
    
    # Unidades totais
    total_units = (x_customers + j_customers) * int(units_per_customer)
    
    return float(daily_profit), int(total_units), float(total_hr_cost)


def calculate_weekly_profit(daily_plans, store_name):
    """Calcula lucro de uma semana (7 dias).
    
    Args:
        daily_plans: Lista de 7 dicts com chaves:
            'num_customers', 'J', 'X', 'PR', 'is_weekend'
        store_name: Nome da loja
    
    Returns:
        Dict com 'weekly_profit', 'total_units', 'total_hr', 'daily_breakdown'
    """
    store = STORE_PARAMS[store_name]
    
    daily_profits = []
    total_units = 0
    total_hr = 0
    
    for day_plan in daily_plans:
        daily_profit, day_units, day_hr = calculate_daily_profit(
            day_plan['num_customers'],
            day_plan['J'],
            day_plan['X'],
            day_plan['PR'],
            day_plan['is_weekend'],
            store_name
        )
        daily_profits.append(daily_profit)
        total_units += day_units
        total_hr += day_hr
    
    # Lucro semanal = sum(lucros diários) - W_s (custo fixo semanal)
    weekly_profit = sum(daily_profits) - store['W_s']
    
    return {
        'weekly_profit': weekly_profit,
        'total_units': total_units,
        'total_hr': total_hr,
        'daily_breakdown': daily_profits
    }


def evaluate_solution(J_array, X_array, PR_array, customers_forecast, store_name, objective='O1'):
    """Avalia uma solução (plano de 7 dias) para otimização.
    
    Args:
        J_array: Array de 7 valores (juniores por dia)
        X_array: Array de 7 valores (experts por dia)
        PR_array: Array de 7 valores (promoção por dia)
        customers_forecast: Previsão de clientes (7 dias)
        store_name: Nome da loja
        objective: 'O1' (maximizar lucro), 'O2' (O1 com constraint), 'O3' (multi-objetivo)
    
    Returns:
        Float: valor para otimização (quanto maior, melhor a solução)
    """
    daily_plans = []
    for i in range(7):
        daily_plans.append({
            'num_customers': customers_forecast[i],
            'J': int(np.clip(J_array[i], 0, 20)),
            'X': int(np.clip(X_array[i], 0, 20)),
            'PR': np.clip(PR_array[i], 0.0, 0.3),
            'is_weekend': i >= 5  # Dias 5 e 6 são sábado e domingo
        })
    
    weekly = calculate_weekly_profit(daily_plans, store_name)
    
    if objective == 'O1':
        # Maximizar lucro sem restrições
        return weekly['weekly_profit']
    
    elif objective == 'O2':
        # Maximizar lucro com constraint de unidades <= 10000
        if weekly['total_units'] <= 10000:
            return weekly['weekly_profit']
        else:
            # Penalidade forte para violação
            return -1e10
    
    elif objective == 'O3':
        # Multi-objetivo: lucro + minimizar HR
        # Normalizar: lucro é tipicamente 1000-5000, HR é 3000-10000
        # Peso: 70% lucro, 30% HR minimizado
        normalized_profit = weekly['weekly_profit'] / 5000  # Normalizar a ~1
        normalized_hr = weekly['total_hr'] / 10000  # Normalizar a ~1
        return 0.7 * normalized_profit - 0.3 * normalized_hr
    
    else:
        raise ValueError(f"Objetivo desconhecido: {objective}")
