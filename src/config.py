import math

# Constantes das Lojas
STORE_PARAMS = {
    'Baltimore':    {'F_j': 1.00, 'F_x': 1.15, 'W_s': 700},
    'Lancaster':    {'F_j': 1.05, 'F_x': 1.20, 'W_s': 730},
    'Philadelphia': {'F_j': 1.10, 'F_x': 1.15, 'W_s': 760},
    'Richmond':     {'F_j': 1.15, 'F_x': 1.25, 'W_s': 800}
}

# Custos de RH (Diários)
HR_COSTS = {
    'weekday': {'J': 60, 'X': 80},
    'weekend': {'J': 70, 'X': 95}
}

# Capacidade de Atendimento
CAPACITY = {'J': 6, 'X': 7}

def standard_round(x):
    """Implementa arredondamento padrão (0.5 sobe)."""
    return math.floor(x + 0.5)