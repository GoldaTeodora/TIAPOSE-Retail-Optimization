import sys
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

st.set_page_config(page_title="Intelligent Retail DSS", layout="wide")

REPORT_DIR = Path("reports")
STORES = ["baltimore", "lancaster", "philadelphia", "richmond"]
SCENARIOS = ["O1", "O2", "O3_WEIGHTED", "O3_NS"]

DAY_NAMES = {
    1: "Domingo", 2: "Segunda", 3: "Terça",
    4: "Quarta", 5: "Quinta", 6: "Sexta", 7: "Sábado"
}
WEEKEND_DAYS = {1, 7}
STORE_WS = {
    "baltimore": 700, "lancaster": 730,
    "philadelphia": 760, "richmond": 800
}


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
def load_global_summary(store: str, objective: str) -> pd.DataFrame | None:
    if objective == "O1":
        path = REPORT_DIR / f"summary_{store}_O1_20runs.csv"
    else:
        path = REPORT_DIR / f"global_summary_{objective}_20runs.csv"
    return load_csv(path)


@st.cache_data
def load_global_comparison(store: str, objective: str) -> pd.DataFrame | None:
    if objective == "O1":
        path = REPORT_DIR / f"summary_{store}_O1_20runs.csv"
    else:
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


def load_store_plan(store: str, objective: str) -> pd.DataFrame | None:
    if objective == "O1":
        return load_local_plan(store)
    return load_global_plan(store, objective)


# =========================
# SIDEBAR
# =========================
st.title("Sistema Inteligente de Apoio à Decisão")

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

df_summary = load_global_summary(selected_store, selected_objective)
df_comparison = load_global_comparison(selected_store, selected_objective)
df_pareto = load_pareto_front()


tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Previsão e Forecast",
    "Gestão e Decisão",
    "Análise de Lojas",
    "Análise Técnica",
    "Simulação"
])


# =========================
# TAB 1 — PREVISÃO
# =========================
with tab1:
    st.header("Previsão de Clientes")
    st.info(
        "Esta área apresenta a previsão futura dos próximos 7 dias e a validação do modelo com valores reais."
    )

    st.subheader("Previsão para os Próximos 7 Dias")

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
                    "Procura acima da média semanal prevista. Recomenda-se atenção ao planeamento de equipa."
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

    st.subheader("Comparação da Previsão entre Lojas")

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

        loja_maior_procura = resumo_lojas.loc[resumo_lojas["Total_Semanal"].idxmax()]

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

    st.subheader("Previsão de Vendas")

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

    st.subheader("Validação do Modelo — Últimos 7 Dias")
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
    st.header(f"Plano Semanal Recomendado — {format_store(selected_store)}")
    st.caption(f"Objetivo de otimização: **{selected_objective}** | {scenario_description(selected_objective)}")

    if df_plan is None:
        st.warning("Ficheiros de otimização não encontrados.")
    else:
        df_store = df_plan.copy()

        if df_store.empty:
            st.warning(f"Sem dados de otimização para {format_store(selected_store)}.")
        else:
            df_store = df_store.sort_values("Dia")
            df_store["Dia_Nome"] = df_store["Dia"].map(DAY_NAMES)
            df_store["Aberto"] = df_store["Atendidos"] > 0
            df_store["Fim_Semana"] = df_store["Dia"].isin(WEEKEND_DAYS)
            df_store["Label"] = df_store.apply(
                lambda r: f"Dia {int(r['Dia'])} ({r['Dia_Nome']})", axis=1
            )

            ws = STORE_WS.get(selected_store, 0)
            lucro_operacional = df_store["Lucro"].sum()
            lucro_semanal = lucro_operacional - ws
            dias_abertos = df_store["Aberto"].sum()
            total_rh = df_store["Juniores"].sum() + df_store["Experts"].sum()
            total_clientes = df_store["Clientes"].sum()
            total_atendidos = df_store["Atendidos"].sum()
            taxa_atendimento = (total_atendidos / total_clientes * 100) if total_clientes > 0 else 0

            melhor_row = df_store.loc[df_store["Lucro"].idxmax()]
            melhor_label = melhor_row["Label"]

            st.subheader("Resumo Executivo")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Lucro Semanal Líquido", f"${lucro_semanal:,.0f}",
                        help=f"Lucro operacional ${lucro_operacional:,.0f} − custo fixo semanal ${ws}")
            col2.metric("Dias em Operação", f"{dias_abertos}/7")
            col3.metric("Taxa de Atendimento", f"{taxa_atendimento:.1f}%",
                        help="Percentagem de clientes atendidos sobre o total previsto")
            col4.metric("RH Total Semanal", f"{total_rh:.0f} pessoas")

            st.success(
                f"**Melhor dia da semana:** {melhor_label} — "
                f"Lucro de **${melhor_row['Lucro']:,.0f}** com {int(melhor_row['Experts'])} experts "
                f"e {int(melhor_row['Juniores'])} juniores."
            )

            dias_fechados = df_store[~df_store["Aberto"]]

            if not dias_fechados.empty and selected_objective != "O1":
                st.subheader("Decisao do Optimizador — Dias Sem Operacao")

                dias_abertos_df = df_store[df_store["Aberto"]]
                melhor_lucro_dia_util = dias_abertos_df[
                    ~dias_abertos_df["Fim_Semana"]
                ]["Lucro"].max() if not dias_abertos_df[~dias_abertos_df["Fim_Semana"]].empty else 0

                for _, row in dias_fechados.iterrows():
                    dia_label = row["Label"]
                    is_weekend = row["Fim_Semana"]
                    clientes = int(row["Clientes"])

                    if is_weekend:
                        razao = (
                            f"**{dia_label}** — {clientes} clientes previstos, mas a loja nao abre. "
                            f"Motivo: fim de semana com custo de Expert $95/dia vs $80 num dia util. "
                            f"Para os mesmos clientes atendidos, o lucro seria ~33% menor do que num dia util. "
                            f"O optimizador priorizou dias com maior margem de lucro."
                        )
                    else:
                        razao = (
                            f"**{dia_label}** — {clientes} clientes previstos, mas a loja nao abre. "
                            f"Motivo: a restricao de 10,000 unidades globais forcou a concentracao "
                            f"de recursos nos dias de maior retorno por unidade."
                        )

                    st.warning(razao)

                dias_abertos_nomes = dias_abertos_df["Dia_Nome"].tolist()
                st.info(
                    f"**Recomendacao:** Operar nos dias {', '.join(dias_abertos_nomes)}. "
                    f"Sao os dias com maior margem de lucro dentro da restricao de unidades."
                )

            st.subheader("Plano Diario Detalhado")

            tabela = df_store[[
                "Label", "Aberto", "Juniores", "Experts",
                "Promocao", "Clientes", "Atendidos",
                "Unidades", "Custo_RH", "Lucro"
            ]].copy()

            tabela["Vendas"] = (tabela["Lucro"] + tabela["Custo_RH"]).where(
                df_store["Atendidos"] > 0, 0
            )

            tabela.columns = [
                "Dia", "Aberto", "Juniores", "Experts",
                "Promocao", "Clientes", "Atendidos",
                "Unidades", "Custo RH", "Lucro", "Vendas"
            ]

            tabela["Aberto_bool"] = tabela["Aberto"]
            tabela["Aberto"] = tabela["Aberto"].map({True: "Sim", False: "Nao"})
            tabela["Promocao"] = tabela.apply(
                lambda r: (f"{r['Promocao']*100:.1f}%" if r["Aberto_bool"] else "—"),
                axis=1
            )

            total_row = pd.DataFrame([{
                "Dia": "TOTAL SEMANA",
                "Aberto": "",
                "Juniores": int(tabela["Juniores"].sum()),
                "Experts": int(tabela["Experts"].sum()),
                "Promocao": "",
                "Clientes": int(tabela["Clientes"].sum()),
                "Atendidos": int(tabela["Atendidos"].sum()),
                "Unidades": int(tabela["Unidades"].sum()),
                "Custo RH": tabela["Custo RH"].sum(),
                "Lucro": tabela["Lucro"].sum(),
                "Vendas": tabela["Vendas"].sum(),
                "Aberto_bool": None
            }])

            tabela_final = pd.concat([tabela, total_row], ignore_index=True)

            row_styles = []
            for _, r in tabela_final.iterrows():
                if r["Dia"] == "TOTAL SEMANA":
                    row_styles.append("background-color: #4a4a4a; color: white; font-weight: bold")
                elif r["Aberto_bool"] == True:
                    row_styles.append("background-color: #1a6b35; color: white")
                elif r["Aberto_bool"] == False:
                    row_styles.append("background-color: #7a1f1f; color: white")
                else:
                    row_styles.append("")

            tabela_show = tabela_final[[
                "Dia", "Aberto", "Juniores", "Experts", "Promocao",
                "Clientes", "Atendidos", "Unidades", "Vendas", "Custo RH", "Lucro"
            ]].copy()
            tabela_show.rename(columns={
                "Vendas": "Vendas ($)",
                "Custo RH": "Custo RH ($)",
                "Lucro": "Lucro ($)"
            }, inplace=True)

            for col in ["Vendas ($)", "Custo RH ($)", "Lucro ($)"]:
                tabela_show[col] = tabela_show[col].apply(
                    lambda v: f"${v:,.0f}" if pd.notna(v) else ""
                )

            def apply_row_style(row):
                return [row_styles[row.name]] * len(row)

            styled = tabela_show.style.apply(apply_row_style, axis=1)

            st.dataframe(styled, use_container_width=True, hide_index=True)
            st.caption("Verde = dia em operacao | Vermelho = dia sem operacao | Cinzento = total semanal")

            st.divider()

            st.subheader("Planeamento de Recursos Humanos")

            fig_workers = go.Figure()
            fig_workers.add_trace(go.Bar(
                x=df_store["Label"],
                y=df_store["Juniores"],
                name="Juniores",
                marker_color="skyblue"
            ))
            fig_workers.add_trace(go.Bar(
                x=df_store["Label"],
                y=df_store["Experts"],
                name="Experts",
                marker_color="royalblue"
            ))
            fig_workers.update_layout(
                barmode="stack",
                title="Equipa por Dia da Semana",
                xaxis_title="Dia",
                yaxis_title="Numero de Trabalhadores",
                template="plotly_white"
            )
            st.plotly_chart(fig_workers, use_container_width=True)

            st.subheader("Procura Prevista vs Clientes Atendidos")

            fig_clients = go.Figure()
            fig_clients.add_trace(go.Bar(
                x=df_store["Label"],
                y=df_store["Clientes"],
                name="Clientes Previstos",
                marker_color="lightgray"
            ))
            fig_clients.add_trace(go.Bar(
                x=df_store["Label"],
                y=df_store["Atendidos"],
                name="Clientes Atendidos",
                marker_color="steelblue"
            ))
            fig_clients.update_layout(
                barmode="overlay",
                title="Clientes Previstos vs Atendidos por Dia",
                xaxis_title="Dia",
                yaxis_title="Clientes",
                template="plotly_white",
                hovermode="x unified"
            )
            st.plotly_chart(fig_clients, use_container_width=True)

            st.subheader("Lucro Diario")

            colors = [
                "green" if aberto else "lightcoral"
                for aberto in df_store["Aberto"]
            ]
            fig_lucro = go.Figure()
            fig_lucro.add_trace(go.Bar(
                x=df_store["Label"],
                y=df_store["Lucro"],
                marker_color=colors,
                name="Lucro Diario",
                text=df_store["Lucro"].apply(lambda v: f"${v:,.0f}"),
                textposition="outside"
            ))
            fig_lucro.update_layout(
                title=f"Lucro Diario — {format_store(selected_store)} ({selected_objective})",
                xaxis_title="Dia",
                yaxis_title="Lucro ($)",
                template="plotly_white",
                showlegend=False
            )
            st.plotly_chart(fig_lucro, use_container_width=True)
            st.caption("Verde = dia em operacao | Vermelho = dia sem operacao")


# =========================
# TAB 3 — ANÁLISE DE LOJAS
# =========================
with tab3:
    st.header("Análise Comparativa de Lojas")
    st.info(
        f"Comparação de métricas operacionais e financeiras entre as 4 lojas "
        f"para o objetivo **{selected_objective}**."
    )

    all_store_plans = {}
    for store in STORES:
        df_temp = load_store_plan(store, selected_objective)
        if df_temp is not None and not df_temp.empty:
            all_store_plans[store] = df_temp

    if not all_store_plans:
        st.warning("Dados de lojas não encontrados para este objetivo.")
    else:
        store_metrics = []
        for store, df_s in all_store_plans.items():
            ws = STORE_WS.get(store, 0)
            lucro_op = df_s["Lucro"].sum()
            lucro_liq = lucro_op - ws
            dias_ab = int((df_s["Atendidos"] > 0).sum())
            total_atend = int(df_s["Atendidos"].sum())
            total_cl = int(df_s["Clientes"].sum())
            taxa = total_atend / total_cl * 100 if total_cl > 0 else 0
            total_j = int(df_s["Juniores"].sum())
            total_x = int(df_s["Experts"].sum())
            total_un = int(df_s["Unidades"].sum())
            custo_rh = df_s["Custo_RH"].sum()

            store_metrics.append({
                "loja_key": store,
                "Loja": format_store(store),
                "Custo Fixo Semanal ($)": ws,
                "Lucro Operacional ($)": lucro_op,
                "Lucro Líquido ($)": lucro_liq,
                "Dias Abertos": dias_ab,
                "Clientes Previstos": total_cl,
                "Clientes Atendidos": total_atend,
                "Taxa Atendimento (%)": round(taxa, 1),
                "Juniores": total_j,
                "Experts": total_x,
                "Total RH": total_j + total_x,
                "Custo RH ($)": custo_rh,
                "Unidades Vendidas": total_un
            })

        df_metrics = pd.DataFrame(store_metrics)

        best_store_name = df_metrics.loc[df_metrics["Lucro Líquido ($)"].idxmax(), "Loja"]
        st.success(f"Loja com maior lucro líquido semanal: **{best_store_name}**")

        # Metric cards
        cols_cards = st.columns(len(store_metrics))
        for i, row in df_metrics.iterrows():
            with cols_cards[i]:
                is_best = row["Lucro Líquido ($)"] == df_metrics["Lucro Líquido ($)"].max()
                label = f"**{row['Loja']}**" + (" ⭐" if is_best else "")
                st.markdown(label)
                st.metric("Lucro Líquido", f"${row['Lucro Líquido ($)']:,.0f}")
                st.metric("Atendimento", f"{row['Taxa Atendimento (%)']:.1f}%")
                st.metric("Dias Abertos", f"{row['Dias Abertos']}/7")
                st.metric("Total RH", f"{row['Total RH']}")

        st.divider()

        # ========== RH DISTRIBUTION ==========
        st.subheader("Distribuição de Recursos Humanos por Loja")
        st.caption("Total de trabalhadores mobilizados ao longo da semana, por categoria.")

        fig_rh = go.Figure()
        fig_rh.add_trace(go.Bar(
            x=df_metrics["Loja"],
            y=df_metrics["Juniores"],
            name="Juniores",
            marker_color="skyblue",
            text=df_metrics["Juniores"],
            textposition="inside"
        ))
        fig_rh.add_trace(go.Bar(
            x=df_metrics["Loja"],
            y=df_metrics["Experts"],
            name="Experts",
            marker_color="royalblue",
            text=df_metrics["Experts"],
            textposition="inside"
        ))
        fig_rh.update_layout(
            barmode="stack",
            title="Total de RH por Loja (semana completa)",
            yaxis_title="Número de Trabalhadores",
            template="plotly_white",
            legend=dict(orientation="h", y=-0.2)
        )
        st.plotly_chart(fig_rh, use_container_width=True)

        col1_rh, col2_rh = st.columns(2)
        with col1_rh:
            fig_rh_ratio = px.pie(
                df_metrics,
                names="Loja",
                values="Total RH",
                title="Distribuição do RH Total entre Lojas",
                template="plotly_white"
            )
            st.plotly_chart(fig_rh_ratio, use_container_width=True)
        with col2_rh:
            fig_custo = px.bar(
                df_metrics,
                x="Loja",
                y="Custo RH ($)",
                text=df_metrics["Custo RH ($)"].apply(lambda v: f"${v:,.0f}"),
                title="Custo Total de RH por Loja ($)",
                template="plotly_white",
                color="Custo RH ($)",
                color_continuous_scale="Reds"
            )
            fig_custo.update_traces(textposition="outside")
            fig_custo.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig_custo, use_container_width=True)

        st.divider()

        # ========== CLIENTS VS ATTENDED ==========
        st.subheader("Clientes Previstos vs Atendidos por Loja")

        col1_cl, col2_cl = st.columns(2)
        with col1_cl:
            fig_clients = go.Figure()
            fig_clients.add_trace(go.Bar(
                x=df_metrics["Loja"],
                y=df_metrics["Clientes Previstos"],
                name="Previstos",
                marker_color="lightsteelblue"
            ))
            fig_clients.add_trace(go.Bar(
                x=df_metrics["Loja"],
                y=df_metrics["Clientes Atendidos"],
                name="Atendidos",
                marker_color="steelblue"
            ))
            fig_clients.update_layout(
                barmode="group",
                title="Clientes Previstos vs Atendidos",
                yaxis_title="Número de Clientes",
                template="plotly_white",
                legend=dict(orientation="h", y=-0.2)
            )
            st.plotly_chart(fig_clients, use_container_width=True)

        with col2_cl:
            fig_taxa = px.bar(
                df_metrics,
                x="Loja",
                y="Taxa Atendimento (%)",
                text=df_metrics["Taxa Atendimento (%)"].apply(lambda v: f"{v:.1f}%"),
                title="Taxa de Atendimento por Loja",
                template="plotly_white",
                color="Taxa Atendimento (%)",
                color_continuous_scale="RdYlGn",
                range_color=[0, 100]
            )
            fig_taxa.update_traces(textposition="outside")
            fig_taxa.update_layout(coloraxis_showscale=False, yaxis_range=[0, 110])
            st.plotly_chart(fig_taxa, use_container_width=True)

        st.divider()

        # ========== UNITS VS PROFIT ==========
        st.subheader("Unidades Vendidas e Lucro por Loja")

        col1_up, col2_up = st.columns(2)
        with col1_up:
            fig_units = go.Figure()
            fig_units.add_trace(go.Bar(
                x=df_metrics["Loja"],
                y=df_metrics["Unidades Vendidas"],
                marker_color="darkorange",
                text=df_metrics["Unidades Vendidas"],
                textposition="outside",
                name="Unidades"
            ))
            fig_units.update_layout(
                title="Unidades Vendidas por Loja",
                yaxis_title="Unidades",
                template="plotly_white",
                showlegend=False
            )
            st.plotly_chart(fig_units, use_container_width=True)

        with col2_up:
            colors_profit = [
                "#1a6b35" if v == df_metrics["Lucro Líquido ($)"].max() else "#3498db"
                for v in df_metrics["Lucro Líquido ($)"]
            ]
            fig_profit_bar = go.Figure()
            fig_profit_bar.add_trace(go.Bar(
                x=df_metrics["Loja"],
                y=df_metrics["Lucro Líquido ($)"],
                marker_color=colors_profit,
                text=df_metrics["Lucro Líquido ($)"].apply(lambda v: f"${v:,.0f}"),
                textposition="outside"
            ))
            fig_profit_bar.update_layout(
                title="Lucro Líquido Semanal por Loja ($)",
                yaxis_title="Lucro ($)",
                template="plotly_white",
                showlegend=False
            )
            st.plotly_chart(fig_profit_bar, use_container_width=True)

        fig_scatter = px.scatter(
            df_metrics,
            x="Unidades Vendidas",
            y="Lucro Líquido ($)",
            text="Loja",
            size="Clientes Atendidos",
            color="Loja",
            title="Unidades Vendidas vs Lucro Líquido (tamanho = clientes atendidos)",
            template="plotly_white",
            labels={
                "Unidades Vendidas": "Unidades",
                "Lucro Líquido ($)": "Lucro Líquido ($)"
            }
        )
        fig_scatter.update_traces(textposition="top center")
        st.plotly_chart(fig_scatter, use_container_width=True)

        st.divider()

        # ========== PROMOTIONS ==========
        st.subheader("Análise de Promoções")
        st.info(
            "O otimizador não aplica promoção nos dias em operação. "
            "Matematicamente, promoção (PR > 0%) aumenta as unidades por cliente "
            "mas reduz a receita por unidade, resultando sempre num lucro menor. "
            "Qualquer valor de PR nos dados refere-se a dias encerrados (sem impacto no resultado)."
        )

        st.divider()

        # ========== SUMMARY TABLE ==========
        st.subheader("Tabela Resumo Completa")

        df_show = df_metrics[[
            "Loja", "Dias Abertos", "Clientes Previstos", "Clientes Atendidos",
            "Taxa Atendimento (%)", "Juniores", "Experts", "Unidades Vendidas",
            "Custo Fixo Semanal ($)", "Lucro Operacional ($)", "Lucro Líquido ($)"
        ]].copy()

        for col in ["Custo Fixo Semanal ($)", "Lucro Operacional ($)", "Lucro Líquido ($)"]:
            df_show[col] = df_show[col].apply(lambda v: f"${v:,.0f}")

        best_idx = df_metrics["Lucro Líquido ($)"].idxmax()

        def highlight_best_store(row):
            if row.name == best_idx:
                return ["background-color: #1a6b35; color: white; font-weight: bold"] * len(row)
            return [""] * len(row)

        styled_metrics = df_show.style.apply(highlight_best_store, axis=1)
        st.dataframe(styled_metrics, use_container_width=True, hide_index=True)
        st.caption("Verde = loja com maior lucro líquido")


# =========================
# TAB 4 — ANÁLISE TÉCNICA
# =========================
with tab4:
    st.header("Análise Técnica — Desempenho dos Algoritmos")

    is_global = selected_objective != "O1"
    if is_global:
        st.info(
            f"Objetivo **{selected_objective}** — otimização global para todas as 4 lojas em conjunto. "
            "Os resultados refletem o lucro total combinado das 4 lojas."
        )
    else:
        st.info(
            f"Objetivo **O1** — otimização local independente para **{format_store(selected_store)}**."
        )

    if df_comparison is None:
        if selected_objective == "O1":
            st.warning(f"Ficheiro summary_{selected_store}_O1_20runs.csv não encontrado.")
        else:
            st.warning(f"Ficheiro global_comparison_{selected_objective}_20runs.csv não encontrado.")
    else:
        df_comp = df_comparison.copy()

        best_row = df_comp.loc[df_comp["mean_profit"].idxmax()]
        best_extra = ""
        if "best_profit" in df_comp.columns and pd.notna(best_row.get("best_profit")):
            best_extra = f" | Melhor run individual: **${best_row['best_profit']:,.0f}**"
        std_val = best_row["std_profit"] if pd.notna(best_row["std_profit"]) else 0
        st.success(
            f"Melhor algoritmo: **{best_row['algorithm'].upper().replace('_', ' ')}** — "
            f"Lucro médio de **${best_row['mean_profit']:,.2f}** (±${std_val:.2f}){best_extra}"
        )

        st.subheader("Lucro Médio por Algoritmo (20 execuções)")

        df_sorted = df_comp.sort_values("mean_profit", ascending=False).copy()
        bar_colors = [
            "#2ecc71" if row["algorithm"] == best_row["algorithm"] else "#3498db"
            for _, row in df_sorted.iterrows()
        ]
        error_vals = df_sorted["std_profit"].fillna(0).tolist()

        fig_compare = go.Figure()
        fig_compare.add_trace(go.Bar(
            x=df_sorted["algorithm"].str.upper().str.replace("_", " "),
            y=df_sorted["mean_profit"],
            error_y=dict(type="data", array=error_vals, visible=True),
            marker_color=bar_colors,
            text=df_sorted["mean_profit"].apply(lambda v: f"${v:,.0f}"),
            textposition="outside"
        ))
        fig_compare.update_layout(
            xaxis_title="Algoritmo",
            yaxis_title="Lucro Médio ($)",
            template="plotly_white",
            showlegend=False
        )
        st.plotly_chart(fig_compare, use_container_width=True)
        st.caption("Verde = melhor algoritmo | Barras de erro = desvio padrão entre as 20 execuções")

        st.subheader("Tabela Comparativa")

        tbl = df_sorted.reset_index(drop=True).copy()
        tbl.insert(0, "Pos.", [f"#{i+1}" for i in range(len(tbl))])
        tbl["Algoritmo"] = tbl["algorithm"].str.upper().str.replace("_", " ")
        tbl["Lucro Médio ($)"] = tbl["mean_profit"].apply(lambda v: f"${v:,.2f}")
        tbl["Desvio Padrão ($)"] = tbl["std_profit"].apply(
            lambda v: f"${v:,.2f}" if pd.notna(v) and v > 0 else "—"
        )
        show_cols = ["Pos.", "Algoritmo", "Lucro Médio ($)", "Desvio Padrão ($)"]
        if "best_profit" in tbl.columns:
            tbl["Melhor Run ($)"] = tbl["best_profit"].apply(
                lambda v: f"${v:,.0f}" if pd.notna(v) else "—"
            )
            tbl["Pior Run ($)"] = tbl["worst_profit"].apply(
                lambda v: f"${v:,.0f}" if pd.notna(v) else "—"
            )
            show_cols += ["Melhor Run ($)", "Pior Run ($)"]

        def highlight_best_row(row):
            if row["Pos."] == "#1":
                return ["background-color: #1a6b35; color: white; font-weight: bold"] * len(row)
            return [""] * len(row)

        styled_tbl = tbl[show_cols].style.apply(highlight_best_row, axis=1)
        st.dataframe(styled_tbl, use_container_width=True, hide_index=True)

        if selected_objective == "O1":
            st.divider()
            st.subheader("Comparação entre Lojas — O1")
            st.caption("Melhor algoritmo por loja com o respetivo lucro médio.")

            stores_profit = []
            for store in STORES:
                df_temp = load_csv(REPORT_DIR / f"summary_{store}_O1_20runs.csv")
                if df_temp is not None and not df_temp.empty:
                    best_algo_row = df_temp.loc[df_temp["mean_profit"].idxmax()]
                    stores_profit.append({
                        "Loja": format_store(store),
                        "Melhor Algoritmo": best_algo_row["algorithm"].upper(),
                        "mean_profit": best_algo_row["mean_profit"],
                        "std_profit": best_algo_row["std_profit"]
                    })

            if stores_profit:
                df_store_comp = pd.DataFrame(stores_profit)
                store_errors = df_store_comp["std_profit"].fillna(0).tolist()

                fig_stores = px.bar(
                    df_store_comp,
                    x="Loja",
                    y="mean_profit",
                    color="Melhor Algoritmo",
                    text=df_store_comp.apply(
                        lambda r: f"${r['mean_profit']:,.0f}<br>{r['Melhor Algoritmo']}", axis=1
                    ),
                    error_y=store_errors,
                    labels={"mean_profit": "Lucro Médio ($)"},
                    template="plotly_white"
                )
                fig_stores.update_traces(textposition="outside")
                fig_stores.update_layout(
                    title="Melhor Lucro Médio por Loja (melhor algoritmo de cada loja)",
                    xaxis_title="Loja",
                    yaxis_title="Lucro Médio ($)",
                    template="plotly_white"
                )
                st.plotly_chart(fig_stores, use_container_width=True)

                df_store_comp["Lucro Médio ($)"] = df_store_comp["mean_profit"].apply(lambda v: f"${v:,.2f}")
                df_store_comp["Desvio Padrão ($)"] = df_store_comp["std_profit"].apply(lambda v: f"${v:,.2f}" if pd.notna(v) else "—")
                st.dataframe(
                    df_store_comp[["Loja", "Melhor Algoritmo", "Lucro Médio ($)", "Desvio Padrão ($)"]],
                    use_container_width=True, hide_index=True
                )

        if selected_objective == "O3_NS":
            st.divider()
            st.subheader("Pareto Front — NSGA-II")
            st.caption(
                "Cada ponto é uma solução não-dominada. O gestor escolhe o equilíbrio entre "
                "maximizar lucro e minimizar recursos humanos."
            )
            if df_pareto is not None and not df_pareto.empty:
                fig_pareto = px.scatter(
                    df_pareto,
                    x="hr",
                    y="profit",
                    title="Pareto Front — Lucro vs Recursos Humanos",
                    labels={"hr": "Total de RH", "profit": "Lucro ($)"},
                    template="plotly_white"
                )
                fig_pareto.update_traces(marker=dict(size=8, color="royalblue"))
                st.plotly_chart(fig_pareto, use_container_width=True)
            else:
                st.warning("Ficheiro do Pareto Front não encontrado.")

        # =========================
        # ANÁLISE DE CONVERGÊNCIA
        # =========================
        st.divider()
        st.subheader("Análise de Convergência")
        st.caption(
            "Evolução da melhor solução encontrada ao longo das iterações. "
            "Quanto mais rápido a curva estabiliza, mais eficiente é o algoritmo."
        )

        if selected_objective == "O1":
            p_coarse = REPORT_DIR / f"convergence_{selected_store}_O1_coarse.png"
            p_fine   = REPORT_DIR / f"convergence_{selected_store}_O1_fine.png"

            vista = st.radio(
                "Vista:",
                ["Visão Geral (coarse)", "Detalhe das primeiras iterações (fine)"],
                horizontal=True
            )

            if vista.startswith("Visão"):
                if p_coarse.exists():
                    st.image(str(p_coarse), use_container_width=True,
                             caption=f"Visão Geral — {format_store(selected_store)} | Todas as iterações")
                else:
                    st.warning("Gráfico de convergência (coarse) não encontrado.")
            else:
                if p_fine.exists():
                    st.image(str(p_fine), use_container_width=True,
                             caption=f"Detalhe — {format_store(selected_store)} | Primeiras iterações (zoom)")
                else:
                    st.warning("Gráfico de convergência (fine) não encontrado.")

            with st.expander("Como interpretar este gráfico?"):
                st.markdown(
                    "- Cada linha representa um algoritmo de otimização.\n"
                    "- O eixo Y mostra o melhor lucro encontrado até àquela iteração.\n"
                    "- Uma curva que sobe rápido e estabiliza cedo indica boa convergência.\n"
                    "- **PSO** e **Genetic** tendem a convergir para valores mais altos.\n"
                    "- **Hill Climbing** e **Simulated Annealing** convergem depressa mas para soluções menos boas.\n"
                    "- **Random** continua a melhorar lentamente (sem memória de iterações anteriores)."
                )

        else:
            conv_key = f"conv_{selected_objective}"

            col_btn, col_info = st.columns([1, 3])
            with col_btn:
                run_conv = st.button(
                    "Calcular Convergência",
                    key=f"btn_conv_{selected_objective}",
                    help="Executa os algoritmos com iterações reduzidas e mostra a curva interativa"
                )
            with col_info:
                st.caption(
                    "Executa uma versão rápida do otimizador (iterações reduzidas) "
                    "e mostra a convergência como gráfico interativo com hover."
                )

            if run_conv:
                if df_future is None:
                    st.error("Ficheiro de previsão future_forecast_7_days.csv não encontrado.")
                else:
                    conv_forecasts = {}
                    for store in STORES:
                        df_s = df_future[df_future["Loja"] == store].sort_values("Data")
                        if not df_s.empty:
                            conv_forecasts[store] = (
                                df_s["Clientes_Previstos"].round().astype(int).tolist()
                            )

                    if len(conv_forecasts) < len(STORES):
                        st.warning("Previsões incompletas — alguns algoritmos podem não correr.")

                    with st.spinner("A calcular convergência (pode demorar ~30 segundos)..."):
                        try:
                            from optimization.methods_global import OptimizationMethodsGlobal

                            opt = OptimizationMethodsGlobal(
                                stores=STORES,
                                forecasts=conv_forecasts,
                                objective=selected_objective,
                                seed=42
                            )

                            histories = {}
                            if selected_objective == "O3_NS":
                                result = opt.nsga2(population_size=20, generations=30)
                                histories["NSGA-II"] = result["history"]
                            else:
                                histories["Random"] = opt.random_search(n_iter=60)["history"]
                                histories["Hill Climbing"] = opt.hill_climbing(max_iter=100)["history"]
                                histories["Simulated Annealing"] = opt.simulated_annealing(
                                    max_iter=100, temp_init=100
                                )["history"]
                                histories["Genetic"] = opt.genetic_algorithm(
                                    population_size=15, generations=30
                                )["history"]
                                histories["PSO"] = opt.particle_swarm(
                                    n_particles=15, max_iter=30
                                )["history"]
                                histories["DE"] = opt.differential_evolution(
                                    pop_size=15, generations=30
                                )["history"]

                            st.session_state[conv_key] = histories
                            st.success("Convergência calculada com sucesso.")

                        except Exception as e:
                            st.error(f"Erro ao calcular convergência: {e}")

            if conv_key in st.session_state:
                histories = st.session_state[conv_key]
                fig_conv = go.Figure()
                for algo, history in histories.items():
                    fig_conv.add_trace(go.Scatter(
                        x=list(range(len(history))),
                        y=history,
                        mode="lines",
                        name=algo,
                        hovertemplate=f"<b>{algo}</b><br>Iteração: %{{x}}<br>Valor: %{{y:,.2f}}<extra></extra>"
                    ))
                fig_conv.update_layout(
                    title=f"Convergência Interativa — {selected_objective}",
                    xaxis_title="Iteração",
                    yaxis_title="Melhor Valor Encontrado",
                    template="plotly_white",
                    hovermode="x unified",
                    legend=dict(orientation="h", y=-0.2)
                )
                st.plotly_chart(fig_conv, use_container_width=True)

                with st.expander("Como interpretar este gráfico?"):
                    st.markdown(
                        "- Cada linha representa um algoritmo a evoluir ao longo das iterações.\n"
                        "- O eixo Y mostra o melhor valor (lucro total) encontrado até àquele momento.\n"
                        "- Uma curva que sobe rapidamente e estabiliza indica convergência eficiente.\n"
                        "- Passe o cursor sobre o gráfico para comparar todos os algoritmos na mesma iteração.\n"
                        "- Os resultados aqui usam **iterações reduzidas** para velocidade — "
                        "os resultados finais do relatório usaram mais iterações."
                    )
            else:
                p_static = REPORT_DIR / f"global_convergence_{selected_objective}.png"
                if p_static.exists():
                    st.caption("Gráfico estático pré-calculado. Clique em **Calcular Convergência** para versão interativa.")
                    st.image(str(p_static), use_container_width=True,
                             caption=f"Convergência Global — {selected_objective}")
                else:
                    st.info("Clique em **Calcular Convergência** para gerar o gráfico interativo.")


# =========================
# TAB 5 — SIMULAÇÃO
# =========================
with tab5:
    st.header("Simulação e Suporte à Decisão do Gestor")
    st.info(
        "Configure as preferências e restrições operacionais. "
        "O sistema analisa todos os cenários disponíveis e recomenda o melhor objetivo de otimização."
    )

    # ========== SECTION 1: PARÂMETROS ==========
    st.subheader("Parâmetros da Simulação")

    col_p1, col_p2 = st.columns(2)

    with col_p1:
        st.markdown("**Prioridades do Gestor**")
        peso_lucro = st.slider(
            "Peso do Lucro na Decisão",
            0.0, 1.0, 0.70, 0.05,
            help="Quanto mais alto, mais o sistema prioriza maximizar o lucro vs. minimizar RH"
        )
        peso_rh = round(1.0 - peso_lucro, 2)

        col_w1, col_w2 = st.columns(2)
        col_w1.metric("Peso Lucro", f"{peso_lucro:.0%}")
        col_w2.metric("Peso RH", f"{peso_rh:.0%}")

    with col_p2:
        st.markdown("**Restrições Operacionais**")
        max_rh_dia = st.slider(
            "Máximo de RH por dia (Juniores + Experts)",
            0, 200, 100, 5,
            help="Número máximo total de trabalhadores permitido num único dia"
        )
        max_pr = st.slider(
            "Promoção máxima permitida",
            0.0, 0.30, 0.30, 0.05,
            format="%.2f",
            help="Valor máximo de PR aceite nos dias de operação"
        )

    st.divider()

    # ========== SECTION 2: CARREGAR TODOS OS CENÁRIOS ==========
    all_plans_sim = {}
    for obj in SCENARIOS:
        df_temp = load_store_plan(selected_store, obj)
        if df_temp is not None and not df_temp.empty:
            all_plans_sim[obj] = df_temp

    if not all_plans_sim:
        st.warning("Nenhum ficheiro de plano encontrado para esta loja.")
    else:
        ws_sim = STORE_WS.get(selected_store, 0)

        sim_results = []
        for obj, df_obj in all_plans_sim.items():
            df_obj = df_obj.copy()
            df_obj["RH_Total_Dia"] = df_obj["Juniores"] + df_obj["Experts"]

            df_valid = df_obj[
                (df_obj["RH_Total_Dia"] <= max_rh_dia) &
                (df_obj["Promocao"] <= max_pr)
            ]

            if df_valid.empty:
                sim_results.append({
                    "Objetivo": obj,
                    "Lucro Operacional ($)": 0.0,
                    "Lucro Líquido ($)": -ws_sim,
                    "RH Total": 0,
                    "Dias Abertos": 0,
                    "Taxa Atendimento (%)": 0.0,
                    "Unidades": 0,
                    "Score": -ws_sim * peso_lucro,
                    "Válido": False,
                    "Observação": "Sem solução válida com estas restrições"
                })
            else:
                lucro_op = df_valid["Lucro"].sum()
                lucro_liq = lucro_op - ws_sim
                rh_total = int(df_valid["RH_Total_Dia"].sum())
                dias_ab = int((df_valid["Atendidos"] > 0).sum())
                atend = df_valid["Atendidos"].sum()
                clientes = df_obj["Clientes"].sum()
                taxa = atend / clientes * 100 if clientes > 0 else 0
                unidades = int(df_valid["Unidades"].sum())
                score = peso_lucro * lucro_liq - peso_rh * rh_total

                sim_results.append({
                    "Objetivo": obj,
                    "Lucro Operacional ($)": round(lucro_op, 2),
                    "Lucro Líquido ($)": round(lucro_liq, 2),
                    "RH Total": rh_total,
                    "Dias Abertos": dias_ab,
                    "Taxa Atendimento (%)": round(taxa, 1),
                    "Unidades": unidades,
                    "Score": round(score, 2),
                    "Válido": True,
                    "Observação": f"PR médio = {df_valid['Promocao'].mean():.2f}"
                })

        df_sim = pd.DataFrame(sim_results)
        df_valid_sim = df_sim[df_sim["Válido"]]

        # ========== SECTION 3: CARDS POR CENÁRIO ==========
        st.subheader("Comparação de Cenários")

        best_score = df_valid_sim["Score"].max() if not df_valid_sim.empty else None
        best_obj = df_valid_sim.loc[df_valid_sim["Score"].idxmax(), "Objetivo"] if not df_valid_sim.empty else None

        if best_obj:
            st.success(
                f"Cenário recomendado: **{best_obj}** "
                f"— Score = **{best_score:,.2f}** "
                f"com peso lucro={peso_lucro:.0%} e peso RH={peso_rh:.0%}"
            )

        cards_cols = st.columns(len(sim_results))
        for i, row in df_sim.iterrows():
            with cards_cols[i]:
                is_best = row["Objetivo"] == best_obj and row["Válido"]
                header_md = f"**{row['Objetivo']}**" + (" ⭐" if is_best else "")
                if is_best:
                    st.success(header_md)
                elif not row["Válido"]:
                    st.error(f"**{row['Objetivo']}**\nSem solução")
                else:
                    st.info(header_md)

                if row["Válido"]:
                    st.metric("Lucro Líquido", f"${row['Lucro Líquido ($)']:,.0f}")
                    st.metric("Score", f"{row['Score']:,.0f}")
                    st.metric("Dias Abertos", f"{row['Dias Abertos']}/7")
                    st.metric("Atendimento", f"{row['Taxa Atendimento (%)']:.1f}%")
                    st.metric("RH Total", f"{row['RH Total']}")

        st.divider()

        # ========== SECTION 4: GRÁFICOS COMPARATIVOS ==========
        st.subheader("Análise Visual dos Cenários")

        tab_charts_a, tab_charts_b, tab_charts_c = st.tabs([
            "Lucro e Score", "Trade-off RH vs Lucro", "Detalhes por Dia"
        ])

        with tab_charts_a:
            col_la, col_lb = st.columns(2)
            with col_la:
                df_valid_plot = df_sim[df_sim["Válido"]].copy()
                bar_colors_sim = [
                    "#1a6b35" if obj == best_obj else "#3498db"
                    for obj in df_valid_plot["Objetivo"]
                ]
                fig_lucro_sim = go.Figure()
                fig_lucro_sim.add_trace(go.Bar(
                    x=df_valid_plot["Objetivo"],
                    y=df_valid_plot["Lucro Líquido ($)"],
                    marker_color=bar_colors_sim,
                    text=df_valid_plot["Lucro Líquido ($)"].apply(lambda v: f"${v:,.0f}"),
                    textposition="outside",
                    name="Lucro Líquido"
                ))
                fig_lucro_sim.update_layout(
                    title="Lucro Líquido Semanal por Objetivo",
                    yaxis_title="Lucro ($)",
                    template="plotly_white",
                    showlegend=False
                )
                st.plotly_chart(fig_lucro_sim, use_container_width=True)

            with col_lb:
                bar_colors_score = [
                    "#1a6b35" if obj == best_obj else "#e67e22"
                    for obj in df_valid_plot["Objetivo"]
                ]
                fig_score = go.Figure()
                fig_score.add_trace(go.Bar(
                    x=df_valid_plot["Objetivo"],
                    y=df_valid_plot["Score"],
                    marker_color=bar_colors_score,
                    text=df_valid_plot["Score"].apply(lambda v: f"{v:,.0f}"),
                    textposition="outside",
                    name="Score"
                ))
                fig_score.update_layout(
                    title=f"Score Ponderado (lucro={peso_lucro:.0%}, RH={peso_rh:.0%})",
                    yaxis_title="Score",
                    template="plotly_white",
                    showlegend=False
                )
                st.plotly_chart(fig_score, use_container_width=True)

        with tab_charts_b:
            df_tradeoff = df_sim[df_sim["Válido"]].copy()
            df_tradeoff["Tamanho"] = df_tradeoff["Score"].clip(lower=1)
            fig_tradeoff = px.scatter(
                df_tradeoff,
                x="RH Total",
                y="Lucro Líquido ($)",
                color="Objetivo",
                size="Tamanho",
                size_max=40,
                text="Objetivo",
                hover_data=["Dias Abertos", "Taxa Atendimento (%)", "Score"],
                title="Trade-off: Recursos Humanos vs Lucro Líquido",
                template="plotly_white",
                labels={
                    "RH Total": "Total de RH (semana)",
                    "Lucro Líquido ($)": "Lucro Líquido ($)"
                }
            )
            fig_tradeoff.update_traces(textposition="top center")
            st.plotly_chart(fig_tradeoff, use_container_width=True)
            st.caption("Tamanho da bolha = Score. Ideal: alto lucro + baixo RH (canto superior esquerdo).")

        with tab_charts_c:
            obj_detail = st.selectbox(
                "Selecione o objetivo para ver o detalhe diário:",
                [r["Objetivo"] for r in sim_results if r["Válido"]],
                key="sim_detail_obj"
            )

            if obj_detail:
                df_det = load_store_plan(selected_store, obj_detail)
                if df_det is not None and not df_det.empty:
                    df_det = df_det.sort_values("Dia").copy()
                    df_det["Label"] = df_det["Dia"].map(DAY_NAMES).apply(
                        lambda d: f"Dia {df_det.index[df_det['Dia'].map(DAY_NAMES) == d].tolist()[0]+1 if False else ''}{d}"
                    )
                    df_det["Label"] = [
                        f"{DAY_NAMES.get(int(r['Dia']), str(int(r['Dia'])))}"
                        for _, r in df_det.iterrows()
                    ]
                    df_det["Aberto"] = df_det["Atendidos"] > 0

                    colors_det = ["#1a6b35" if ab else "#7a1f1f" for ab in df_det["Aberto"]]

                    fig_det = go.Figure()
                    fig_det.add_trace(go.Bar(
                        x=df_det["Label"],
                        y=df_det["Lucro"],
                        marker_color=colors_det,
                        text=df_det["Lucro"].apply(lambda v: f"${v:,.0f}"),
                        textposition="outside",
                        name="Lucro Diário"
                    ))
                    fig_det.update_layout(
                        title=f"Lucro Diário — {format_store(selected_store)} | {obj_detail}",
                        yaxis_title="Lucro ($)",
                        template="plotly_white",
                        showlegend=False
                    )
                    st.plotly_chart(fig_det, use_container_width=True)

                    col_d1, col_d2 = st.columns(2)
                    with col_d1:
                        fig_rh_det = go.Figure()
                        fig_rh_det.add_trace(go.Bar(
                            x=df_det["Label"],
                            y=df_det["Juniores"],
                            name="Juniores",
                            marker_color="skyblue"
                        ))
                        fig_rh_det.add_trace(go.Bar(
                            x=df_det["Label"],
                            y=df_det["Experts"],
                            name="Experts",
                            marker_color="royalblue"
                        ))
                        fig_rh_det.update_layout(
                            barmode="stack",
                            title="RH por Dia",
                            template="plotly_white"
                        )
                        st.plotly_chart(fig_rh_det, use_container_width=True)

                    with col_d2:
                        fig_cl_det = go.Figure()
                        fig_cl_det.add_trace(go.Bar(
                            x=df_det["Label"],
                            y=df_det["Clientes"],
                            name="Previstos",
                            marker_color="lightsteelblue"
                        ))
                        fig_cl_det.add_trace(go.Bar(
                            x=df_det["Label"],
                            y=df_det["Atendidos"],
                            name="Atendidos",
                            marker_color="steelblue"
                        ))
                        fig_cl_det.update_layout(
                            barmode="group",
                            title="Clientes por Dia",
                            template="plotly_white"
                        )
                        st.plotly_chart(fig_cl_det, use_container_width=True)

        st.divider()

        # ========== SECTION 5: ANÁLISE DE SENSIBILIDADE ==========
        st.subheader("Análise de Sensibilidade")
        st.caption("Como o lucro líquido do melhor cenário varia conforme as restrições mudam.")

        with st.expander("Mostrar análise de sensibilidade ao limite de RH"):
            rh_range = list(range(0, 210, 10))
            sensitivity_rows = []

            if best_obj and best_obj in all_plans_sim:
                df_best_plan = all_plans_sim[best_obj].copy()
                df_best_plan["RH_Total_Dia"] = df_best_plan["Juniores"] + df_best_plan["Experts"]

                for rh_val in rh_range:
                    df_filt = df_best_plan[
                        (df_best_plan["RH_Total_Dia"] <= rh_val) &
                        (df_best_plan["Promocao"] <= max_pr)
                    ]
                    lucro_s = df_filt["Lucro"].sum() - ws_sim if not df_filt.empty else -ws_sim
                    sensitivity_rows.append({"Max RH/dia": rh_val, "Lucro Líquido ($)": lucro_s})

                df_sens = pd.DataFrame(sensitivity_rows)
                fig_sens = px.line(
                    df_sens,
                    x="Max RH/dia",
                    y="Lucro Líquido ($)",
                    markers=True,
                    title=f"Sensibilidade do Lucro ao Limite de RH — {best_obj}",
                    template="plotly_white"
                )
                fig_sens.add_vline(
                    x=max_rh_dia,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Atual: {max_rh_dia}",
                    annotation_position="top right"
                )
                st.plotly_chart(fig_sens, use_container_width=True)

        st.divider()

        # ========== SECTION 6: RECOMENDAÇÃO E EXPORT ==========
        st.subheader("Recomendação Final e Exportação")

        if best_obj:
            best_row_sim = df_sim[df_sim["Objetivo"] == best_obj].iloc[0]

            col_rec1, col_rec2 = st.columns([2, 1])
            with col_rec1:
                st.markdown(f"""
**Cenário recomendado: {best_obj}**

| Métrica | Valor |
|---|---|
| Lucro Operacional | ${best_row_sim['Lucro Operacional ($)']:,.0f} |
| Custo Fixo Semanal | ${ws_sim:,.0f} |
| **Lucro Líquido** | **${best_row_sim['Lucro Líquido ($)']:,.0f}** |
| Dias em Operação | {best_row_sim['Dias Abertos']}/7 |
| Taxa de Atendimento | {best_row_sim['Taxa Atendimento (%)']:.1f}% |
| Total RH (semana) | {best_row_sim['RH Total']} |
| Score Ponderado | {best_row_sim['Score']:,.2f} |
""")
            with col_rec2:
                st.markdown("**Exportar Resultados**")

                csv_export = df_sim[[
                    "Objetivo", "Lucro Operacional ($)", "Lucro Líquido ($)",
                    "RH Total", "Dias Abertos", "Taxa Atendimento (%)",
                    "Unidades", "Score", "Observação"
                ]].copy()

                st.download_button(
                    label="Descarregar Comparação (CSV)",
                    data=csv_export.to_csv(index=False).encode("utf-8"),
                    file_name=f"simulacao_{selected_store}_{selected_objective}.csv",
                    mime="text/csv"
                )

                if best_obj in all_plans_sim:
                    df_export_plan = all_plans_sim[best_obj].copy()
                    st.download_button(
                        label=f"Descarregar Plano {best_obj} (CSV)",
                        data=df_export_plan.to_csv(index=False).encode("utf-8"),
                        file_name=f"plano_{selected_store}_{best_obj}.csv",
                        mime="text/csv",
                        key="btn_download_plan"
                    )
        else:
            st.warning(
                "Nenhum cenário válido encontrado com as restrições atuais. "
                "Tente aumentar o limite de RH por dia ou a promoção máxima."
            )

st.markdown("---")
st.markdown("Desenvolvido em Streamlit para suporte à decisão em lojas de retalho.")
