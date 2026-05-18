import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="Intelligent Retail DSS", layout="wide")

REPORT_DIR = Path("reports")
STORES = ["baltimore", "lancaster", "philadelphia", "richmond"]
SCENARIOS = ["O1", "O2", "O3_WEIGHTED", "O3_NS"]


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
def load_future_sales_forecast() -> pd.DataFrame | None:
    path = REPORT_DIR / "future_sales_forecast_7_days.csv"

    df = load_csv(path)

    if df is not None:
        if "Data" in df.columns:
            df["Data"] = pd.to_datetime(df["Data"])

        if "Loja" in df.columns:
            df["Loja"] = df["Loja"].str.lower()

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
def load_global_summary(objective: str) -> pd.DataFrame | None:
    path = REPORT_DIR / f"global_summary_{objective}_20runs.csv"
    return load_csv(path)


@st.cache_data
def load_global_comparison(objective: str) -> pd.DataFrame | None:
    path = REPORT_DIR / f"global_comparison_{objective}_20runs.csv"
    return load_csv(path)


@st.cache_data
def load_global_plan(store: str, objective: str) -> pd.DataFrame | None:
    path = REPORT_DIR / f"global_plan_{store}_{objective}_20runs.csv"

    df = load_csv(path)

    if df is not None and "Data" in df.columns:
        df["Data"] = pd.to_datetime(df["Data"])

    return df


@st.cache_data
def load_local_plan(store: str) -> pd.DataFrame | None:
    path = REPORT_DIR / f"plan_{store}_O1_20runs.csv"

    df = load_csv(path)

    if df is not None and "Data" in df.columns:
        df["Data"] = pd.to_datetime(df["Data"])

    return df


@st.cache_data
def load_pareto_front() -> pd.DataFrame | None:
    path = REPORT_DIR / "pareto_front_O3_NS_20runs.csv"
    return load_csv(path)


def format_store(s: str) -> str:
    return s.capitalize()


def scenario_description(scenario: str) -> str:
    descriptions = {
        "O1": "Maximização direta do lucro.",
        "O2": "Lucro sujeito a restrições adicionais.",
        "O3_WEIGHTED": "Função multiobjetivo ponderada.",
        "O3_NS": "Otimização multiobjetivo com NSGA-II."
    }
    return descriptions.get(scenario, "")


# =========================
# SIDEBAR
# =========================
st.title(" Sistema Inteligente de Apoio à Decisão")

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

selected_objective = st.sidebar.selectbox(
    "Objetivo de Otimização",
    SCENARIOS
)


# =========================
# LOAD DATA
# =========================
df_future = load_future_forecast()
df_forecast = load_forecast_validation()
df_sales_forecast = load_future_sales_forecast()

if selected_objective == "O1":
    df_plan = load_local_plan(selected_store)
else:
    df_plan = load_global_plan(selected_store, selected_objective)

df_summary = load_global_summary(selected_objective)
df_comparison = load_global_comparison(selected_objective)
df_pareto = load_pareto_front()


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

    # =========================
    # PREVISÃO FUTURA
    # =========================
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

    # =========================
    # COMPARAÇÃO ENTRE LOJAS — PREVISÃO
    # =========================
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

    # =========================
    # PREVISÃO DE VENDAS
    # =========================
    st.subheader("💰 Previsão de Vendas")

    if df_sales_forecast is None:
        st.warning("Ficheiro reports/future_sales_forecast_7_days.csv não encontrado.")
    else:
        df_sales_store = df_sales_forecast[
            df_sales_forecast["Loja"] == selected_store
        ].copy()

        if df_sales_store.empty:
            st.warning(f"Sem previsões de vendas para {format_store(selected_store)}.")
        else:
            df_sales_store = df_sales_store.sort_values("Data")

            avg_sales = df_sales_store["Sales_Previstas"].mean()
            peak_sales = df_sales_store["Sales_Previstas"].max()
            total_sales = df_sales_store["Sales_Previstas"].sum()
            peak_sales_row = df_sales_store.loc[df_sales_store["Sales_Previstas"].idxmax()]

            col1, col2, col3 = st.columns(3)
            col1.metric("Vendas Médias Previstas", f"{avg_sales:,.0f}")
            col2.metric("Pico de Vendas Previsto", f"{peak_sales:,.0f}")
            col3.metric("Total Semanal Previsto", f"{total_sales:,.0f}")

            st.success(
                f"Maior volume de vendas previsto em **{peak_sales_row['Data'].date()}**, "
                f"com **{int(peak_sales_row['Sales_Previstas']):,} vendas previstas**."
            )

            fig_sales = go.Figure()
            fig_sales.add_trace(go.Scatter(
                x=df_sales_store["Data"],
                y=df_sales_store["Sales_Previstas"],
                mode="lines+markers",
                name="Vendas Previstas"
            ))
            fig_sales.update_layout(
                title=f"Previsão de Vendas — Próximos 7 Dias ({format_store(selected_store)})",
                xaxis_title="Data",
                yaxis_title="Sales Previstas",
                template="plotly_white"
            )
            st.plotly_chart(fig_sales, width="stretch")

            st.subheader("Tabela de Previsão de Vendas")
            st.dataframe(
                df_sales_store[[
                    "Loja", "Data", "Horizonte", "Clientes_Previstos",
                    "Sales_Ratio", "Sales_Previstas"
                ]],
                use_container_width=True
            )
    st.divider()

    # =========================
    # VALIDAÇÃO
    # =========================
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
# TAB 2 — OTIMIZAÇÃO
# =========================
with tab2:
    st.header("Otimização de Planos Semanais")
    st.info("Visualização dos resultados da otimização por cenário, com filtro temporal.")

    if df_plan is None:
        st.warning("Ficheiros de otimização não encontrados.")
    else:
        df_store = df_plan.copy()

        if df_store.empty:
            st.warning(f"Sem dados de otimização para {format_store(selected_store)}.")
        else:
            df_store = df_store.sort_values("Dia")

            available_days = sorted(df_store["Dia"].astype(int).unique().tolist())

            selected_days = st.multiselect(
                "Seleciona dias para análise",
                available_days,
                default=available_days
            )

            if selected_days:
                df_filtered = df_store[df_store["Dia"].isin(selected_days)].copy()
            else:
                df_filtered = df_store.copy()

            st.subheader(f"Resumo do Objetivo — {format_store(selected_store)}")

            lucro_total = df_filtered["Lucro"].sum()

            st.metric(
                "Lucro Total",
                f"{lucro_total:.2f}"
            )

            st.write(df_filtered.columns)

            resumo_cenario = pd.DataFrame([{
                "Objetivo": selected_objective,
                "Juniores": df_filtered["Juniores"].mean(),
                "Experts": df_filtered["Experts"].mean(),
                "Promocao": df_filtered["Promocao"].mean(),
                "Lucro": df_filtered["Lucro"].sum()
            }])

            col1, col2 = st.columns(2)

            with col1:
                fig_hr = go.Figure()
                fig_hr.add_trace(go.Bar(
                    x=resumo_cenario["Objetivo"],
                    y=resumo_cenario["Juniores"],
                    name="Juniores"
                ))
                fig_hr.add_trace(go.Bar(
                    x=resumo_cenario["Objetivo"],
                    y=resumo_cenario["Experts"],
                    name="Experts"
                ))
                fig_hr.update_layout(
                    barmode="group",
                    title="Recursos Humanos Médios por Cenário",
                    template="plotly_white"
                )
                st.plotly_chart(fig_hr, width="stretch")

            with col2:
                fig_pr = px.bar(
                    resumo_cenario,
                    x="Objetivo",
                    y="Promocao",
                    text_auto=".2f",
                    title="Promoção Média (PR) por Cenário"
                )
                st.plotly_chart(fig_pr, width="stretch")

            

            st.subheader("Comparação Lado a Lado dos Cenários")
            compare_table = resumo_cenario.copy()
            compare_table["Descrição"] = compare_table["Objetivo"].map(scenario_description)
            compare_table = compare_table[[
                "Objetivo",
                "Descrição",
                "Lucro",
                "Juniores",
                "Experts",
                "Promocao"
            ]]

            st.dataframe(compare_table, use_container_width=True)
            # =========================
            # EVOLUÇÃO DO LUCRO
            # =========================
            fig_lucro = px.line(
                df_filtered,
                x="Dia",
                y="Lucro",
                markers=True,
                title=f"Evolução do Lucro Diário — {format_store(selected_store)}"
            )

            st.plotly_chart(fig_lucro, width="stretch")

            # =========================
            # TABELA FINAL
            # =========================

            st.subheader("Tabela Detalhada da Otimização")
            st.dataframe(
                df_filtered.style.format({
                    "Lucro": "{:.2f}",
                    "Promocao": "{:.2f}"
                    
                }),
                use_container_width=True
            )


# =========================
# TAB 3 — COMPARAÇÃO
# =========================
with tab3:
    st.header("Comparação de Resultados")
    st.info("Comparação entre lojas e cenários, com base nos resumos finais.")

    if df_summary is None:
        st.warning("Ficheiro reports/summary_optimization.csv não encontrado.")
    else:
        st.subheader("Resumo da Loja Selecionada")
        df_store = df_summary.copy()

        if not df_store.empty:
            st.dataframe(df_store, use_container_width=True)

            fig_store = px.bar(
                df_store,
                x="algorithm",
                y="mean_profit",
                text_auto=".2f",
                title=f"Lucro por Cenário — {format_store(selected_store)}"
            )
            st.plotly_chart(fig_store, width="stretch")
        else:
            st.warning(f"Sem resumo disponível para {format_store(selected_store)}.")

        st.subheader("Comparação de Algoritmos")

        if df_comparison is not None and not df_comparison.empty:

                st.dataframe(df_comparison, use_container_width=True)

                fig_compare = px.bar(
                    df_comparison,
                    x="algorithm",
                    y="mean_profit",
                    color="algorithm",
                    text_auto=".2f",
                    title="Comparação de Algoritmos"
                )

                st.plotly_chart(fig_compare, width="stretch")

        else:
                st.warning("Sem dados de comparação disponíveis.")


# =========================
# TAB 4 — SIMULAÇÃO
# =========================
with tab4:
    st.header("Simulação e Suporte à Decisão")
    st.info("Exploração interativa dos cenários com base nos resultados já calculados.")

    if df_plan is None:
        st.warning("Ficheiros de otimização não encontrados.")
    else:
        df_store = df_plan.copy()

        if df_store.empty:
            st.warning(f"Sem dados de simulação para {format_store(selected_store)}.")
        else:
            peso_lucro = st.slider("Peso do Lucro", 0.0, 1.0, 0.70, 0.05)
            peso_rh = 1.0 - peso_lucro
            st.markdown(f"**Peso dos Recursos Humanos:** {peso_rh:.2f}")

            max_rh = st.slider("Máximo total de RH permitido (J + X)", 0, 500, 200, 10)
            max_pr = st.slider("Promoção máxima permitida (PR)", 0.0, 0.30, 0.30, 0.01)

            resumo = []

            df_store["RH_Total_Dia"] = (
                df_store["Juniores"] +
                df_store["Experts"]
            )

            df_valid = df_store[
                (df_store["RH_Total_Dia"] <= max_rh) &
                (df_store["Promocao"] <= max_pr)
            ].copy()

            if df_valid.empty:

                resumo.append((
                    selected_objective,
                    0.0,
                    0.0,
                    0.0,
                    "Sem solução válida"
                ))

            else:

                lucro_total = df_valid["Lucro"].sum()
                rh_total = df_valid["RH_Total_Dia"].sum()
                pr_medio = df_valid["Promocao"].mean()

                score = peso_lucro * lucro_total - peso_rh * rh_total

                resumo.append((
                    selected_objective,
                    lucro_total,
                    rh_total,
                    score,
                    f"PR médio={pr_medio:.2f}"
                ))
            df_sim = pd.DataFrame(
                resumo,
                columns=["Objetivo", "Lucro", "RH_Total", "Score", "Observação"]
            )

            st.dataframe(df_sim, use_container_width=True)

            valid_rows = df_sim[df_sim["Observação"] != "Sem solução válida"]
            if not valid_rows.empty:
                melhor = valid_rows.sort_values("Score", ascending=False).iloc[0]

                st.success(
                    f"Melhor cenário para {format_store(selected_store)}: "
                    f"**{melhor['Objetivo']}** "
                    f"(Score = {melhor['Score']:.2f})"
                )

                st.markdown("### Explicação da decisão")
                st.markdown(
                    f"- O cenário **{melhor['Objetivo']}** apresenta o melhor equilíbrio entre lucro e recursos humanos.\n"
                    f"- Lucro total considerado: **{melhor['Lucro']:.2f}**\n"
                    f"- RH total considerado: **{melhor['RH_Total']:.2f}**\n"
                    f"- Observação: **{melhor['Observação']}**"
                )

                fig_score = px.bar(
                    valid_rows,
                    x="Objetivo",
                    y="Score",
                    text_auto=".2f",
                    title=f"Score por Cenário — {format_store(selected_store)}"
                )
                st.plotly_chart(fig_score, width="stretch")
            else:
                st.warning("Nenhum cenário cumpre as restrições definidas.")


st.markdown("---")
st.markdown("Desenvolvido em Streamlit para suporte à decisão em lojas de retalho.")
