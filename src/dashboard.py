import ast
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core.config import REPORTS_PATH, get_data_path

st.set_page_config(page_title="Intelligent Retail DSS", layout="wide")

STORES = ["baltimore", "lancaster", "philadelphia", "richmond"]
SCENARIOS = ["O1", "O2", "O3"]

st.title("🧠 Intelligent Decision Support System for USA Stores")
st.caption("Forecast futuro, validação, otimização e comparação num único painel.")


def format_store(store_name: str) -> str:
    return store_name.capitalize()


def scenario_description(scenario: str) -> str:
    descriptions = {
        "O1": "Maximização direta do lucro.",
        "O2": "Lucro com restrição adicional de unidades.",
        "O3": "Lucro com penalização de recursos humanos.",
    }
    return descriptions.get(scenario, "")


def parse_sequence(value):
    if pd.isna(value):
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, (list, tuple)):
                return list(parsed)
            return [parsed]
        except Exception:
            return [value]
    return [value]


def safe_sum(value):
    values = parse_sequence(value)
    numeric_values = []
    for item in values:
        try:
            numeric_values.append(float(item))
        except Exception:
            continue
    if numeric_values:
        return float(sum(numeric_values))
    try:
        return float(value)
    except Exception:
        return 0.0


def get_report_file(filename: str) -> Path:
    return REPORTS_PATH / filename


@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_data(show_spinner=False)
def load_store_history(store_name: str) -> pd.DataFrame | None:
    path = get_data_path(store_name, raw=False)
    df = load_csv(path)
    if df is None:
        return None
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
    return df


@st.cache_data(show_spinner=False)
def load_future_forecast() -> pd.DataFrame | None:
    df = load_csv(get_report_file("future_forecast_7_days.csv"))
    if df is None:
        return None
    if "Data" in df.columns:
        df["Data"] = pd.to_datetime(df["Data"])
    if "Loja" in df.columns:
        df["Loja"] = df["Loja"].str.lower()
    return df


@st.cache_data(show_spinner=False)
def load_future_sales_forecast() -> pd.DataFrame | None:
    df = load_csv(get_report_file("future_sales_forecast_7_days.csv"))
    if df is None:
        return None
    if "Data" in df.columns:
        df["Data"] = pd.to_datetime(df["Data"])
    if "Loja" in df.columns:
        df["Loja"] = df["Loja"].str.lower()
    return df


@st.cache_data(show_spinner=False)
def load_forecast_validation() -> pd.DataFrame | None:
    df = load_csv(get_report_file("forecast_validation_all_stores.csv"))
    if df is None:
        return None
    if "Dia" in df.columns:
        df["Dia"] = pd.to_datetime(df["Dia"])
    if "Loja" in df.columns:
        df["Loja"] = df["Loja"].str.lower()
    return df


@st.cache_data(show_spinner=False)
def load_rolling_result(store_name: str, objective: str | None = None) -> pd.DataFrame | None:
    filename = f"rolling_optimum_{store_name}.csv" if objective is None else f"rolling_optimum_{store_name}_{objective}.csv"
    df = load_csv(get_report_file(filename))
    if df is None:
        return None
    for date_col in ["start_date", "end_date"]:
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col])
    if "Loja" in df.columns:
        df["Loja"] = df["Loja"].str.lower()
    return df


store_options = STORES
selected_store = st.sidebar.selectbox("Loja", store_options, format_func=format_store)
selected_compare_store = st.sidebar.selectbox(
    "Comparar com",
    store_options,
    index=1 if selected_store != store_options[1] else 0,
    format_func=format_store,
    key="compare_store",
)

# Data loading
history_df = load_store_history(selected_store)
future_df = load_future_forecast()
sales_future_df = load_future_sales_forecast()
validation_df = load_forecast_validation()

future_tabs = st.tabs([
    "Exploração",
    "Forecast Futuro",
    "Forecast de Vendas",
    "Validação",
    "Otimização",
    "Comparação e Simulação",
])

with future_tabs[0]:
    st.header("Exploração dos Dados")
    st.markdown(
        "Esta secção resume o comportamento histórico da loja selecionada, com foco em padrões temporais, sazonalidade e relações entre variáveis."
    )

    if history_df is None:
        st.warning("Ficheiro processado da loja não encontrado.")
    else:
        if "Date" in history_df.columns:
            history_df = history_df.sort_values("Date")

        col1, col2, col3 = st.columns(3)
        col1.metric("Média de Clientes", f"{history_df['Num_Customers'].mean():.0f}")
        col2.metric("Pico de Clientes", f"{history_df['Num_Customers'].max():.0f}")
        if "Sales" in history_df.columns:
            col3.metric("Vendas Totais", f"{history_df['Sales'].sum():,.0f}")
        else:
            col3.metric("Vendas Totais", "N/A")

        fig_time = px.line(
            history_df,
            x="Date",
            y="Num_Customers",
            title=f"Evolução Diária de Clientes — {format_store(selected_store)}",
            markers=True,
        )
        fig_time.update_layout(template="plotly_white", xaxis_title="Data", yaxis_title="Clientes")
        st.plotly_chart(fig_time, use_container_width=True)

        if "Day_of_Week" in history_df.columns:
            day_labels = {1: "Seg", 2: "Ter", 3: "Qua", 4: "Qui", 5: "Sex", 6: "Sáb", 7: "Dom"}
            history_df = history_df.copy()
            history_df["Dia_Label"] = history_df["Day_of_Week"].map(day_labels)
            fig_week = px.box(
                history_df,
                x="Dia_Label",
                y="Num_Customers",
                title=f"Distribuição por Dia da Semana — {format_store(selected_store)}",
            )
            fig_week.update_layout(template="plotly_white", xaxis_title="Dia", yaxis_title="Clientes")
            st.plotly_chart(fig_week, use_container_width=True)

        corr_cols = [col for col in ["Num_Customers", "Sales", "Pct_On_Sale", "Is_Weekend", "Is_Black_Friday"] if col in history_df.columns]
        if len(corr_cols) >= 2:
            corr = history_df[corr_cols].corr()
            fig_corr = px.imshow(corr, text_auto=".2f", aspect="auto", title=f"Matriz de Correlação — {format_store(selected_store)}")
            fig_corr.update_layout(template="plotly_white")
            st.plotly_chart(fig_corr, use_container_width=True)

        if "Sales" in history_df.columns:
            fig_sales = px.scatter(
                history_df,
                x="Num_Customers",
                y="Sales",
                trendline="ols",
                title=f"Clientes vs Vendas — {format_store(selected_store)}",
            )
            fig_sales.update_layout(template="plotly_white", xaxis_title="Clientes", yaxis_title="Vendas")
            st.plotly_chart(fig_sales, use_container_width=True)

with future_tabs[1]:
    st.header("Forecast Futuro")
    st.markdown("Previsão de clientes para os próximos 7 dias por loja.")

    if future_df is None:
        st.warning("Gere primeiro `future_forecast_7_days.csv` com `python generate_future_forecast.py`.")
    else:
        future_store = future_df[future_df["Loja"] == selected_store].copy()
        if future_store.empty:
            st.warning(f"Sem previsões futuras para {format_store(selected_store)}.")
        else:
            future_store = future_store.sort_values("Data")
            col1, col2, col3 = st.columns(3)
            col1.metric("Clientes Médios", f"{future_store['Clientes_Previstos'].mean():.0f}")
            col2.metric("Pico Previsto", f"{future_store['Clientes_Previstos'].max():.0f}")
            col3.metric("Total 7 Dias", f"{future_store['Clientes_Previstos'].sum():.0f}")

            if history_df is not None:
                recent_history = history_df.tail(14).copy()
                fig_future = go.Figure()
                fig_future.add_trace(go.Scatter(
                    x=recent_history["Date"],
                    y=recent_history["Num_Customers"],
                    mode="lines+markers",
                    name="Histórico recente",
                ))
                fig_future.add_trace(go.Scatter(
                    x=future_store["Data"],
                    y=future_store["Clientes_Previstos"],
                    mode="lines+markers",
                    name="Forecast futuro",
                    line=dict(dash="dash"),
                ))
                fig_future.update_layout(template="plotly_white", xaxis_title="Data", yaxis_title="Clientes")
                st.plotly_chart(fig_future, use_container_width=True)

            st.dataframe(future_store[["Loja", "Data", "Horizonte", "Clientes_Previstos"]], use_container_width=True)

            by_store = (
                future_df.groupby("Loja", as_index=False)
                .agg(Clientes_Medios=("Clientes_Previstos", "mean"), Pico_Previsto=("Clientes_Previstos", "max"), Total_Semanal=("Clientes_Previstos", "sum"))
            )
            col1, col2 = st.columns(2)
            with col1:
                fig_total = px.bar(by_store, x="Loja", y="Total_Semanal", text_auto=".0f", title="Total de Clientes Previsto por Loja")
                st.plotly_chart(fig_total, use_container_width=True)
            with col2:
                fig_peak = px.bar(by_store, x="Loja", y="Pico_Previsto", text_auto=".0f", title="Pico de Procura Previsto por Loja")
                st.plotly_chart(fig_peak, use_container_width=True)

with future_tabs[2]:
    st.header("Forecast de Vendas")
    st.markdown("Converte a previsão de clientes em vendas previstas por loja.")

    if sales_future_df is None:
        st.warning("Gere primeiro `future_sales_forecast_7_days.csv` com `python generate_future_sales_forecast.py`.")
    else:
        sales_store = sales_future_df[sales_future_df["Loja"] == selected_store].copy()
        if sales_store.empty:
            st.warning(f"Sem previsões de vendas para {format_store(selected_store)}.")
        else:
            sales_store = sales_store.sort_values("Data")
            col1, col2, col3 = st.columns(3)
            col1.metric("Vendas Médias", f"{sales_store['Sales_Previstas'].mean():,.0f}")
            col2.metric("Pico de Vendas", f"{sales_store['Sales_Previstas'].max():,.0f}")
            col3.metric("Total 7 Dias", f"{sales_store['Sales_Previstas'].sum():,.0f}")

            fig_sales = go.Figure()
            fig_sales.add_trace(go.Scatter(
                x=sales_store["Data"],
                y=sales_store["Sales_Previstas"],
                mode="lines+markers",
                name="Vendas previstas",
            ))
            fig_sales.update_layout(template="plotly_white", xaxis_title="Data", yaxis_title="Vendas")
            st.plotly_chart(fig_sales, use_container_width=True)

            st.dataframe(sales_store[["Loja", "Data", "Horizonte", "Clientes_Previstos", "Sales_Ratio", "Sales_Previstas"]], use_container_width=True)

with future_tabs[3]:
    st.header("Validação do Modelo")
    st.markdown("Comparação entre valores reais e previsões geradas para os últimos 7 dias.")

    if validation_df is None:
        st.warning("Gere primeiro `forecast_validation_all_stores.csv` com a validação da main.")
    else:
        validation_store = validation_df[validation_df["Loja"] == selected_store].copy()
        if validation_store.empty:
            st.warning(f"Sem dados de validação para {format_store(selected_store)}.")
        else:
            validation_store = validation_store.sort_values("Dia")
            validation_store["Erro Absoluto"] = (validation_store["Clientes Reais"] - validation_store["Clientes Previstos"]).abs()

            col1, col2, col3 = st.columns(3)
            col1.metric("MAE", f"{validation_store['Erro Absoluto'].mean():.2f}")
            col2.metric("Erro Máximo", f"{validation_store['Erro Absoluto'].max():.2f}")
            col3.metric("Dias Avaliados", f"{len(validation_store)}")

            fig_val = go.Figure()
            fig_val.add_trace(go.Scatter(
                x=validation_store["Dia"],
                y=validation_store["Clientes Reais"],
                mode="lines+markers",
                name="Clientes Reais",
            ))
            fig_val.add_trace(go.Scatter(
                x=validation_store["Dia"],
                y=validation_store["Clientes Previstos"],
                mode="lines+markers",
                name="Clientes Previstos",
            ))
            fig_val.update_layout(template="plotly_white", xaxis_title="Data", yaxis_title="Clientes")
            st.plotly_chart(fig_val, use_container_width=True)

            fig_err = px.bar(validation_store, x="Dia", y="Erro Absoluto", title=f"Erro Absoluto Diário — {format_store(selected_store)}")
            fig_err.update_layout(template="plotly_white", xaxis_title="Data", yaxis_title="Erro")
            st.plotly_chart(fig_err, use_container_width=True)

            st.dataframe(validation_store, use_container_width=True)

            ranking_rows = []
            for store in STORES:
                df_store = validation_df[validation_df["Loja"] == store].copy()
                if df_store.empty:
                    continue
                df_store["Erro Absoluto"] = (df_store["Clientes Reais"] - df_store["Clientes Previstos"]).abs()
                y_true = df_store["Clientes Reais"].astype(float)
                y_pred = df_store["Clientes Previstos"].astype(float)
                rmse = ((y_true - y_pred) ** 2).mean() ** 0.5
                mae = df_store["Erro Absoluto"].mean()
                ranking_rows.append({
                    "Loja": store,
                    "RMSE": rmse,
                    "MAE": mae,
                    "R2": 1.0 - (((y_true - y_pred) ** 2).sum() / ((y_true - y_true.mean()) ** 2).sum()) if ((y_true - y_true.mean()) ** 2).sum() > 0 else 0.0,
                })

            if ranking_rows:
                ranking_df = pd.DataFrame(ranking_rows).sort_values("MAE")
                st.subheader("Ranking Global de Validação")
                st.dataframe(ranking_df, use_container_width=True)

with future_tabs[4]:
    st.header("Otimização")
    st.markdown("Planos ótimos semanais por objetivo, com comparação entre cenários.")

    objective_tabs = st.multiselect("Objetivos", SCENARIOS, default=SCENARIOS)
    objective_results = {}
    for objective in objective_tabs:
        objective_results[objective] = load_rolling_result(selected_store, objective)

    available_results = {obj: df for obj, df in objective_results.items() if df is not None}
    if not available_results:
        st.warning(f"Sem relatórios de otimização para {format_store(selected_store)}.")
    else:
        col1, col2, col3 = st.columns(3)
        for obj, df in available_results.items():
            if "lucro_otimo" in df.columns:
                total_profit = df["lucro_otimo"].sum()
                total_units = df["unidades"].sum() if "unidades" in df.columns else 0
                total_hr = df["hr_cost"].sum() if "hr_cost" in df.columns else 0
                if obj == "O1":
                    col1.metric(f"{obj} Lucro", f"${total_profit:,.2f}")
                elif obj == "O2":
                    col2.metric(f"{obj} Lucro", f"${total_profit:,.2f}")
                else:
                    col3.metric(f"{obj} Lucro", f"${total_profit:,.2f}")

        fig_profit = go.Figure()
        for obj, df in available_results.items():
            if "start_date" in df.columns and "lucro_otimo" in df.columns:
                fig_profit.add_trace(go.Scatter(x=df["start_date"], y=df["lucro_otimo"], mode="lines+markers", name=obj))
        fig_profit.update_layout(template="plotly_white", xaxis_title="Data", yaxis_title="Lucro ótimo")
        st.plotly_chart(fig_profit, use_container_width=True)

        fig_units = go.Figure()
        for obj, df in available_results.items():
            if "start_date" in df.columns and "unidades" in df.columns:
                fig_units.add_trace(go.Scatter(x=df["start_date"], y=df["unidades"], mode="lines+markers", name=obj))
        fig_units.update_layout(template="plotly_white", xaxis_title="Data", yaxis_title="Unidades")
        st.plotly_chart(fig_units, use_container_width=True)

        fig_hr = go.Figure()
        for obj, df in available_results.items():
            if "start_date" in df.columns and "hr_cost" in df.columns:
                fig_hr.add_trace(go.Scatter(x=df["start_date"], y=df["hr_cost"], mode="lines+markers", name=obj))
        fig_hr.update_layout(template="plotly_white", xaxis_title="Data", yaxis_title="Custo HR")
        st.plotly_chart(fig_hr, use_container_width=True)

        st.subheader("Tabela Detalhada")
        for obj, df in available_results.items():
            st.markdown(f"**Objetivo {obj}:** {scenario_description(obj)}")
            st.dataframe(df, use_container_width=True, height=260)

with future_tabs[5]:
    st.header("Comparação e Simulação")
    st.markdown("Compara cenários entre lojas e permite simular pesos e restrições de decisão.")

    compare_results = {}
    for store in STORES:
        for objective in SCENARIOS:
            df = load_rolling_result(store, objective)
            if df is not None:
                compare_results[(store, objective)] = df

    if not compare_results:
        st.warning("Sem relatórios de comparação disponíveis.")
    else:
        comparison_rows = []
        for (store, objective), df in compare_results.items():
            comparison_rows.append({
                "Loja": store,
                "Objetivo": objective,
                "Lucro Total": df["lucro_otimo"].sum() if "lucro_otimo" in df.columns else 0,
                "Unidades Totais": df["unidades"].sum() if "unidades" in df.columns else 0,
                "Custo HR Total": df["hr_cost"].sum() if "hr_cost" in df.columns else 0,
            })

        comparison_df = pd.DataFrame(comparison_rows)
        st.dataframe(comparison_df.sort_values(["Loja", "Objetivo"]), use_container_width=True)

        store_df = comparison_df[comparison_df["Loja"] == selected_store].copy()
        if not store_df.empty:
            fig_store_compare = px.bar(
                store_df,
                x="Objetivo",
                y="Lucro Total",
                text_auto=".2f",
                title=f"Lucro por Objetivo — {format_store(selected_store)}",
            )
            fig_store_compare.update_layout(template="plotly_white")
            st.plotly_chart(fig_store_compare, use_container_width=True)

        compare_store_df = comparison_df[comparison_df["Loja"].isin([selected_store, selected_compare_store])].copy()
        if not compare_store_df.empty:
            fig_compare = px.bar(
                compare_store_df,
                x="Objetivo",
                y="Lucro Total",
                color="Loja",
                barmode="group",
                text_auto=".2f",
                title=f"Comparação de Lucro — {format_store(selected_store)} vs {format_store(selected_compare_store)}",
            )
            fig_compare.update_layout(template="plotly_white")
            st.plotly_chart(fig_compare, use_container_width=True)

        st.subheader("Simulação de Decisão")
        sim_store = st.selectbox("Loja para simulação", STORES, format_func=format_store, key="sim_store")
        sim_objective = st.selectbox("Objetivo", SCENARIOS, index=0, key="sim_obj")
        max_units = st.slider("Máximo de unidades vendidas", min_value=5000, max_value=20000, value=10000, step=500)
        max_hr = st.slider("Máximo de custo HR", min_value=0, max_value=3000, value=1200, step=50)
        weight_profit = st.slider("Peso do lucro", min_value=0.0, max_value=1.0, value=0.7, step=0.05)
        weight_hr = 1.0 - weight_profit
        st.markdown(f"Peso do custo HR: **{weight_hr:.2f}**")

        sim_df = load_rolling_result(sim_store, sim_objective)
        if sim_df is None or sim_df.empty:
            st.warning(f"Sem dados para simulação em {format_store(sim_store)} - {sim_objective}.")
        else:
            sim_df = sim_df.copy()
            if "lucro_otimo" in sim_df.columns:
                last_row = sim_df.iloc[-1]
                lucro = float(last_row.get("lucro_otimo", 0))
                unidades = float(last_row.get("unidades", 0))
                hr_cost = float(last_row.get("hr_cost", 0))

                if unidades > max_units and unidades > 0:
                    lucro *= max_units / unidades
                    unidades = float(max_units)
                if hr_cost > max_hr and hr_cost > 0:
                    lucro *= max_hr / hr_cost
                    hr_cost = float(max_hr)

                score = weight_profit * lucro - weight_hr * hr_cost

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Lucro ajustado", f"${lucro:,.2f}")
                col2.metric("Unidades ajustadas", f"{unidades:,.0f}")
                col3.metric("HR ajustado", f"${hr_cost:,.2f}")
                col4.metric("Score simulado", f"{score:,.2f}")

                st.success(f"Melhor cenário simulado para os parâmetros escolhidos: **{sim_objective}**")

st.markdown("---")
st.caption("Desenvolvido com Streamlit, Plotly e GitHub Copilot")
