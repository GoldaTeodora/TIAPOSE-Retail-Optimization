from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from core.config import REPORTS_PATH, get_data_path

STORES = ["baltimore", "lancaster", "philadelphia", "richmond"]
CUSTOMER_FORECAST_FILE = REPORTS_PATH / "future_forecast_7_days.csv"
OUTPUT_FILE = REPORTS_PATH / "future_sales_forecast_7_days.csv"


def calculate_sales_ratio(store_name: str) -> float:
    df = pd.read_csv(get_data_path(store_name, raw=False)).dropna()
    total_customers = df["Num_Customers"].sum()
    total_sales = df["Sales"].sum()
    if total_customers == 0:
        return 0.0
    return total_sales / total_customers


def generate_sales_forecast():
    if not CUSTOMER_FORECAST_FILE.exists():
        raise FileNotFoundError(
            "Ficheiro reports/future_forecast_7_days.csv não encontrado. "
            "Execute primeiro: python generate_future_forecast.py"
        )

    df_forecast = pd.read_csv(CUSTOMER_FORECAST_FILE)
    results = []

    for store in STORES:
        df_store = df_forecast[df_forecast["Loja"] == store].copy()
        if df_store.empty:
            continue

        sales_ratio = calculate_sales_ratio(store)
        df_store["Sales_Ratio"] = sales_ratio
        df_store["Sales_Previstas"] = (
            df_store["Clientes_Previstos"] * sales_ratio
        ).round(0).astype(int)
        results.append(df_store)

    final_df = pd.concat(results, ignore_index=True)
    REPORTS_PATH.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Previsões complementares de vendas guardadas em: {OUTPUT_FILE}")
    print(final_df)


if __name__ == "__main__":
    generate_sales_forecast()
