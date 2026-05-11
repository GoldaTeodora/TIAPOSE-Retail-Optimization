import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="Intelligent Retail DSS", layout="wide")

# 🔥 FIX ÚNICO (caminho absoluto correto)
REPORT_DIR = Path(__file__).resolve().parent.parent / "reports"

STORES = ["baltimore", "lancaster", "philadelphia", "richmond"]
SCENARIOS = ["O1", "O2", "O3"]


# =========================
# HELPERS
# =========================
@st.cache_data
def load_csv(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_data
def load_future_forecast() -> pd.DataFrame | None:
    path = REPORT_DIR / "future_forecast_7_days.csv"
    df = load_csv(path)
    if df is not None:
        df["Data"] = pd.to_datetime(df["Data"])
        df["Loja"] = df["Loja"].str.lower()
        df["Dia_Semana"] = df["Data"].dt.day_name()
    return df


@st.cache_data
def load_forecast_validation() -> pd.DataFrame | None:
    path = REPORT_DIR / "forecast_validation_all_stores.csv"
    df = load_csv(path)
    if df is not None and "Dia" in df.columns:
        df["Dia"] = pd.to_datetime(df["Dia"])
        df["Loja"] = df["Loja"].str.lower()
    return df


@st.cache_data
def load_weekly_plan() -> pd.DataFrame | None:
    # tenta primeiro ficheiro esperado
    path = REPORT_DIR / "weekly_plan_scenarios.csv"
    if path.exists():
        df = pd.read_csv(path)
    else:
        # 🔥 fallback para os teus ficheiros plan_*
        files = list(REPORT_DIR.glob("plan_*_20runs.csv"))

        if not files:
            return None

        dfs = []
        for f in files:
            df_temp = pd.read_csv(f)

            # 🔥 extrair loja e cenário do nome
            name = f.stem  # ex: plan_baltimore_O1_20runs
            parts = name.split("_")

            if len(parts) >= 4:
                loja = parts[1]
                cenario = parts[2]

                df_temp["Loja"] = loja
                df_temp["Cenario"] = cenario

            dfs.append(df_temp)

        df = pd.concat(dfs, ignore_index=True)

    if "Loja" in df.columns:
        df["Loja"] = df["Loja"].str.lower()

    if "Data" in df.columns:
        df["Data"] = pd.to_datetime(df["Data"], errors="coerce")

    return df


@st.cache_data
def load_summary_optimization() -> pd.DataFrame | None:
    path = REPORT_DIR / "summary_optimization.csv"

    if path.exists():
        df = pd.read_csv(path)
    else:
        # 🔥 fallback para summary_*
        files = list(REPORT_DIR.glob("summary_*_20runs.csv"))

        if not files:
            return None

        dfs = []
        for f in files:
            df_temp = pd.read_csv(f)

            # extrair loja e cenário
            name = f.stem  # summary_baltimore_O1_20runs
            parts = name.split("_")

            if len(parts) >= 4:
                loja = parts[1]
                cenario = parts[2]

                df_temp["Loja"] = loja
                df_temp["Cenario"] = cenario

            dfs.append(df_temp)

        df = pd.concat(dfs, ignore_index=True)

    if "Loja" in df.columns:
        df["Loja"] = df["Loja"].str.lower()

    return df


def format_store(s: str) -> str:
    return s.capitalize()


def scenario_description(scenario: str) -> str:
    descriptions = {
        "O1": "Maximização direta do lucro.",
        "O2": "Lucro sujeito a restrições adicionais.",
        "O3": "Lucro com penalização associada aos recursos."
    }
    return descriptions.get(scenario, "")


# =========================
# SIDEBAR
# =========================
st.title("🧠 Sistema Inteligente de Apoio à Decisão")

st.sidebar.header("Filtros Globais")
selected_store = st.sidebar.selectbox(
    "Loja",
    STORES,
    format_func=format_store
)

selected_compare_store = st.sidebar.selectbox(
    "Comparar com outra loja",
    STORES,
    index=1 if selected_store != STORES[1] else 0,
    format_func=format_store
)


# =========================
# LOAD DATA
# =========================
df_future = load_future_forecast()
df_forecast = load_forecast_validation()
df_weekly = load_weekly_plan()
df_summary = load_summary_optimization()


tab1, tab2, tab3, tab4 = st.tabs([
    "Previsão de Clientes",
    "Otimização de Planos",
    "Comparação de Resultados",
    "Simulação e Decisão"
])


# =========================
# TAB 1 — PREVISÃO
# =========================
with tab1:
    st.header("Previsão de Clientes")
    st.info(
        "Esta área apresenta a previsão futura dos próximos 7 dias e a validação do modelo com valores reais."
    )

    st.subheader("🔮 Previsão para os Próximos 7 Dias")

    if df_future is None:
        st.warning("Ficheiro reports/future_forecast_7_days.csv não encontrado.")
    else:
        df_future_store = df_future[df_future["Loja"] == selected_store].copy()

        if df_future_store.empty:
            st.warning(f"Sem previsões futuras para {format_store(selected_store)}.")
        else:
            df_future_store = df_future_store.sort_values("Data")

            avg_clients = df_future_store["Clientes_Previstos"].mean()
            peak_clients = df_future_store["Clientes_Previstos"].max()
            total_clients = df_future_store["Clientes_Previstos"].sum()
            peak_row = df_future_store.loc[df_future_store["Clientes_Previstos"].idxmax()]

            col1, col2, col3 = st.columns(3)
            col1.metric("Clientes Médios (7 dias)", f"{avg_clients:.0f}")
            col2.metric("Pico Previsto", f"{peak_clients:.0f}")
            col3.metric("Total Semanal Previsto", f"{total_clients:.0f}")

            st.success(
                f"Maior procura prevista em **{peak_row['Data'].date()}** "
                f"({peak_row['Dia_Semana']}) com **{int(peak_row['Clientes_Previstos'])} clientes**."
            )

            if peak_clients > avg_clients * 1.15:
                st.warning(
                    "⚠️ Procura acima da média semanal prevista. Recomenda-se atenção ao planeamento de equipa."
                )
            else:
                st.info("A procura prevista mantém-se relativamente estável ao longo da semana.")

            fig_future = go.Figure()
            fig_future.add_trace(go.Scatter(
                x=df_future_store["Data"],
                y=df_future_store["Clientes_Previstos"],
                mode="lines+markers",
                name="Clientes Previstos"
            ))
            fig_future.update_layout(
                title=f"Previsão de Clientes — Próximos 7 Dias ({format_store(selected_store)})",
                xaxis_title="Data",
                yaxis_title="Número de Clientes",
                template="plotly_white"
            )
            st.plotly_chart(fig_future, width="stretch")

            st.subheader("Tabela de Previsão Futura")
            st.dataframe(
                df_future_store[[
                    "Loja", "Data", "Dia_Semana", "Horizonte", "Clientes_Previstos"
                ]],
                use_container_width=True
            )

    st.subheader("🏬 Comparação da Previsão entre Lojas")

    if df_future is None:
        st.warning("Não foi possível comparar lojas porque o ficheiro de previsão futura não foi encontrado.")
    else:
        resumo_lojas = (
            df_future
            .groupby("Loja", as_index=False)
            .agg(
                Clientes_Medios=("Clientes_Previstos", "mean"),
                Pico_Previsto=("Clientes_Previstos", "max"),
                Total_Semanal=("Clientes_Previstos", "sum")
            )
        )

        loja_maior_procura = resumo_lojas.loc[
            resumo_lojas["Total_Semanal"].idxmax()
        ]

        st.info(
            f"A loja com maior procura semanal prevista é "
            f"**{format_store(loja_maior_procura['Loja'])}**, "
            f"com **{int(loja_maior_procura['Total_Semanal'])} clientes** previstos."
        )

        col1, col2 = st.columns(2)

        with col1:
            fig_total = px.bar(
                resumo_lojas,
                x="Loja",
                y="Total_Semanal",
                text_auto=".0f",
                title="Total de Clientes Previsto por Loja"
            )
            st.plotly_chart(fig_total, width="stretch")

        with col2:
            fig_pico = px.bar(
                resumo_lojas,
                x="Loja",
                y="Pico_Previsto",
                text_auto=".0f",
                title="Pico de Procura Previsto por Loja"
            )
            st.plotly_chart(fig_pico, width="stretch")

        fig_linhas = px.line(
            df_future,
            x="Data",
            y="Clientes_Previstos",
            color="Loja",
            markers=True,
            title="Evolução da Previsão dos Próximos 7 Dias por Loja"
        )
        st.plotly_chart(fig_linhas, width="stretch")

        st.subheader("Resumo da Previsão por Loja")
        st.dataframe(resumo_lojas, use_container_width=True)

    st.divider()

    st.subheader("📊 Validação do Modelo — Últimos 7 Dias")
    st.caption("Comparação entre valores reais e previstos para avaliar a qualidade do modelo.")

    if df_forecast is None:
        st.warning("Ficheiro reports/forecast_validation_all_stores.csv não encontrado.")
    else:
        df_store = df_forecast[df_forecast["Loja"] == selected_store].copy()

        if df_store.empty:
            st.warning(f"Sem dados de validação para {format_store(selected_store)}.")
        else:
            df_store = df_store.sort_values("Dia")
            df_store["Erro Absoluto"] = (
                df_store["Clientes Reais"] - df_store["Clientes Previstos"]
            ).abs()

            col1, col2, col3 = st.columns(3)
            col1.metric("MAE", f"{df_store['Erro Absoluto'].mean():.2f}")
            col2.metric("Erro Máximo", f"{df_store['Erro Absoluto'].max():.2f}")
            col3.metric("Dias Avaliados", f"{len(df_store)}")

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_store["Dia"],
                y=df_store["Clientes Reais"],
                mode="lines+markers",
                name="Clientes Reais"
            ))
            fig.add_trace(go.Scatter(
                x=df_store["Dia"],
                y=df_store["Clientes Previstos"],
                mode="lines+markers",
                name="Clientes Previstos"
            ))
            fig.update_layout(
                title=f"Clientes Reais vs Previstos — {format_store(selected_store)}",
                xaxis_title="Data",
                yaxis_title="Número de Clientes",
                template="plotly_white"
            )
            st.plotly_chart(fig, width="stretch")

            fig_err = px.bar(
                df_store,
                x="Dia",
                y="Erro Absoluto",
                title=f"Erro Absoluto Diário — {format_store(selected_store)}"
            )
            st.plotly_chart(fig_err, width="stretch")

            st.subheader("Tabela de Validação")
            st.dataframe(df_store, use_container_width=True)


# =========================
# RESTO DO CÓDIGO (INALTERADO)
# =========================

st.markdown("---")
st.markdown("Desenvolvido em Streamlit para suporte à decisão em lojas de retalho.")