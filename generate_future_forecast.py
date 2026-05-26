from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from core.config import REPORTS_PATH, XGBOOST_FEATURES, get_data_path, get_model_path

STORES = ["baltimore", "lancaster", "philadelphia", "richmond"]
TARGET = "Num_Customers"


def load_xgb_model(store):
    model_path = get_model_path(store, "xgboost")
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")

    loaded = joblib.load(model_path)
    if isinstance(loaded, dict):
        return loaded.get("model", loaded)
    if isinstance(loaded, tuple):
        return loaded[0]
    return loaded


def build_future_row(history_df, future_date, history_values):
    last_row = history_df.iloc[-1]
    row = {}

    row["Day_of_Week"] = future_date.dayofweek + 1
    row["Is_Weekend"] = int(future_date.dayofweek >= 5)
    row["Month"] = future_date.month
    row["Quarter"] = future_date.quarter
    row["Day_of_Year"] = future_date.dayofyear
    row["Year"] = future_date.year
    row["WeekOfYear"] = int(future_date.isocalendar().week)
    row["DayOfMonth"] = future_date.day

    for feature in [
        "Num_Employees", "Pct_On_Sale", "Is_Christmas", "Is_Easter_Sunday",
        "Is_Known_Closed_Day", "Is_Black_Friday", "Is_Tourist_Event",
        "Is_Holiday", "Is_Memorial_Day", "Event_Nearby", "Holiday_Nearby",
        "Is_Peak_Day",
    ]:
        row[feature] = float(last_row.get(feature, 0)) if feature in ["Num_Employees", "Pct_On_Sale"] else 0

    def lag(offset):
        if len(history_values) >= offset:
            return float(history_values[-offset])
        return float(history_values[0])

    row["Lag_Customers_1"] = lag(1)
    row["Lag_Customers_7"] = lag(7)
    row["Lag_Customers_14"] = lag(14)
    row["Lag_Customers_28"] = lag(28)

    def rolling_stats(window):
        values = history_values[-window:] if len(history_values) >= window else history_values[:]
        return float(np.mean(values)), float(np.std(values))

    row["Rolling_Mean_7"], row["Rolling_Std_7"] = rolling_stats(7)
    row["Rolling_Mean_14"], row["Rolling_Std_14"] = rolling_stats(14)
    row["Rolling_Mean_30"] = float(np.mean(history_values[-30:] if len(history_values) >= 30 else history_values))

    future_df = pd.DataFrame([row])
    for feature in XGBOOST_FEATURES:
        if feature not in future_df.columns:
            future_df[feature] = 0
    return future_df[XGBOOST_FEATURES]


def generate_forecast_for_store(store):
    df = pd.read_csv(get_data_path(store, raw=False)).dropna()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    last_date = df["Date"].max()
    model = load_xgb_model(store)

    history_values = df[TARGET].astype(float).tolist()
    rows = []

    for horizon in range(1, 8):
        future_date = last_date + pd.Timedelta(days=horizon)
        X_future = build_future_row(df, future_date, history_values)
        pred = float(model.predict(X_future)[0])
        pred = max(pred, 0)

        rows.append({
            "Loja": store,
            "Data": future_date.date(),
            "Horizonte": horizon,
            "Clientes_Previstos": round(pred),
        })
        history_values.append(pred)

    return pd.DataFrame(rows)


def run():
    REPORTS_PATH.mkdir(parents=True, exist_ok=True)
    all_forecasts = []
    for store in STORES:
        print(f"Gerando previsão futura para {store}...")
        all_forecasts.append(generate_forecast_for_store(store))

    final_df = pd.concat(all_forecasts, ignore_index=True)
    output_path = REPORTS_PATH / "future_forecast_7_days.csv"
    final_df.to_csv(output_path, index=False)
    print(f"\nPrevisões futuras guardadas em: {output_path}")
    print(final_df)


if __name__ == "__main__":
    run()
