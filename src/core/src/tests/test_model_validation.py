import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from core.profit_calculator import calculate_daily_profit
store = "richmond"

clientes = 100
J = 5
X = 5

print("\n=== TESTE PROMOÇÃO ===")

for PR in [0.0, 0.1, 0.2, 0.3]:
    lucro, unidades, hr = calculate_daily_profit(
        num_customers=clientes,
        J=J,
        X=X,
        PR=PR,
        is_weekend=False,
        store_name=store
    )
    print(f"PR={PR} -> Lucro={lucro:.2f}, Unidades={unidades}")

    print("\n=== TESTE STAFF ===")

clientes = 150
PR = 0.1

for J, X in [(2,2), (5,5), (10,10)]:
    lucro, unidades, hr = calculate_daily_profit(
        num_customers=clientes,
        J=J,
        X=X,
        PR=PR,
        is_weekend=False,
        store_name=store
    )
    print(f"J={J}, X={X} -> Lucro={lucro:.2f}, HR={hr}")

    print("\n=== TESTE EXTREMO ===")

lucro, unidades, hr = calculate_daily_profit(
    num_customers=100,
    J=0,
    X=0,
    PR=0.1,
    is_weekend=False,
    store_name=store
)

print(f"Sem staff -> Lucro={lucro:.2f}")

