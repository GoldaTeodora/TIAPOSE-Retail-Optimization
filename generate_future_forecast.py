import os
import pickle
import pandas as pd
import numpy as np
from pathlib import Path


STORES = ["baltimore", "lancaster", "philadelphia", "richmond"]

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")

TARGET = "Num_Customers"

BASE_FEATURES = [
    "Day_of_Week", "Is_Weekend", "Month", "Quarter", "Day_of_Year",
    "Is_Black_Friday", "Is_Tourist_Event", "Is_Holiday", "Is_Memorial_Day",
    "Event_Nearby", "Holiday_Nearby"
]

LAG_FEATURES = BASE_FEATURES + [
    "Lag_Customers_1", "Lag_Customers_2", "Lag_Customers_3",
    "Lag_Customers_4", "Lag_Customers_5", "Lag_Customers_6",
    "Lag_Customers_7"
]


def create_future_calendar_features(date):
    return {
        "Day_of_Week": date.dayofweek,
        "Is_Weekend": int(date.dayofweek >= 5),
        "Month": date.month,
        "Quarter": date.quarter,
        "Day_of_Year": date.dayofyear,

        # valores futuros desconhecidos: assumir 0 por defeito
        "Is_Black_Friday": 0,
        "Is_Tourist_Event": 0,
        "Is_Holiday": 0,
        "Is_Memorial_Day": 0,
        "Event_Nearby": 0,
        "Holiday_Nearby": 0
    }


def build_future_row(df, future_date):
    row = create_future_calendar_features(future_date)

    history = df[TARGET].dropna().values

    for lag in range(1, 8):
        row[f"Lag_Customers_{lag}"] = history[-lag]

    return pd.DataFrame([row])[LAG_FEATURES]


def generate_forecast_for_store(store):
    df = pd.read_csv(DATA_DIR / f"{store}_clean.csv").dropna()

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    last_date = df["Date"].max()

    rows = []

    for h in range(1, 8):
        future_date = last_date + pd.Timedelta(days=h)

        model_path = MODEL_DIR / store / f"xgb_h{h}.pkl"

        if not model_path.exists():
            raise FileNotFoundError(f"Modelo não encontrado: {model_path}")

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        X_future = build_future_row(df, future_date)

        pred_log = model.predict(X_future)[0]
        pred = np.expm1(pred_log)

        rows.append({
            "Loja": store,
            "Data": future_date.date(),
            "Horizonte": h,
            "Clientes_Previstos": round(max(pred, 0))
        })

    return pd.DataFrame(rows)


def run():
    os.makedirs(REPORT_DIR, exist_ok=True)

    all_forecasts = []

    for store in STORES:
        print(f"Gerando previsão futura para {store}...")
        forecast_df = generate_forecast_for_store(store)
        all_forecasts.append(forecast_df)

    final_df = pd.concat(all_forecasts, ignore_index=True)

    output_path = REPORT_DIR / "future_forecast_7_days.csv"
    final_df.to_csv(output_path, index=False)

    print(f"\nPrevisões futuras guardadas em: {output_path}")
    print(final_df)


if __name__ == "__main__":
    run()