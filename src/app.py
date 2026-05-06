import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="Intelligent Retail DSS", layout="wide")

st.title("🧠 Intelligent Decision Support System for USA Stores")

# Caminho absoluto para reports (FIX)
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"

# Sidebar - Global filters
def sidebar():
    st.sidebar.header("Filtros Globais")
    store = st.sidebar.selectbox(
        "Loja",
        ["baltimore", "lancaster", "philadelphia", "richmond"],
        format_func=lambda x: x.capitalize(),
        key="sidebar_store"
    )
    week = st.sidebar.number_input("Semana alvo (nº)", min_value=1, value=1)
    return store, week

store, week = sidebar()

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Previsão de Clientes",
    "Otimização de Planos",
    "Comparação de Métodos",
    "Simulação e Decisão"
])

# =========================================================
# TAB 1
# =========================================================
with tab1:
    st.header("Previsão de Clientes por Loja e Método")
    st.info("Aqui o gestor pode analisar previsões, comparar métodos e ver a qualidade das previsões.")

    st.warning("⚠️ Ainda não existem ficheiros rolling_optimum → tab desativada")

# =========================================================
# TAB 2
# =========================================================
with tab2:
    st.header("Otimização de Planos Semanais")
    st.info("Visualize e compare planos otimizados para cada objetivo (O1, O2, O3).")

    lojas = ["baltimore", "lancaster", "philadelphia", "richmond"]
    objetivos = ["O1", "O2", "O3"]

    col1, col2 = st.columns(2)

    with col1:
       loja_sel = st.selectbox(
            "Loja",
            lojas,
            format_func=lambda x: x.capitalize(),
            key="opt_loja"
)

    with col2:
        objetivos_sel = st.multiselect(
            "Objetivos",
            objetivos,
            default=objetivos,
            key="opt_obj"
)

    st.subheader(f"Lucro por dia - {loja_sel.capitalize()}")

    fig = go.Figure()

    for obj in objetivos_sel:
        fname = REPORTS_DIR / f"plan_{loja_sel}_{obj}_20runs.csv"

        if fname.exists():
            df = pd.read_csv(fname)

            # 🔥 DETETAR COLUNA DE LUCRO
            colunas_possiveis = ["Profit", "Lucro", "profit", "lucro"]
            col_lucro = None

            for col in colunas_possiveis:
                if col in df.columns:
                    col_lucro = col
                    break

            if col_lucro is None:
                col_lucro = df.columns[-1]  # fallback

            fig.add_trace(go.Scatter(
                x=df["Dia"],
                y=df[col_lucro],
                mode="lines+markers",
                name=obj
            ))

    fig.update_layout(
        xaxis_title="Dia",
        yaxis_title="Lucro",
        template="plotly_dark"
    )

    st.plotly_chart(fig, use_container_width=True)

    # Tabelas
    st.subheader("Tabela detalhada")

    for obj in objetivos_sel:
        fname = REPORTS_DIR / f"plan_{loja_sel}_{obj}_20runs.csv"

        if fname.exists():
            df = pd.read_csv(fname)
            st.markdown(f"**Objetivo {obj}:**")
            st.dataframe(df, use_container_width=True)
        else:
            st.warning(f"Sem dados para {obj}")

# =========================================================
# TAB 3
# =========================================================
with tab3:
    st.header("Comparação de Métodos")

    lojas = ["baltimore", "lancaster", "philadelphia", "richmond"]
    resultados = []

    for loja in lojas:
        for obj in ["O1", "O2", "O3"]:
            fname = REPORTS_DIR / f"summary_{loja}_{obj}_20runs.csv"

            if fname.exists():
                df = pd.read_csv(fname)

                if "mean_profit" in df.columns:
                    best = df.loc[df["mean_profit"].idxmax()]
                    resultados.append((loja, obj, best["algorithm"], best["mean_profit"]))

    if resultados:
        st.table(pd.DataFrame(resultados, columns=["Loja", "Objetivo", "Melhor Algoritmo", "Lucro Médio"]))
    else:
        st.warning("Sem dados de comparação.")

# =========================================================
# TAB 4
# =========================================================
with tab4:
    st.header("Simulação e Decisão")

    lojas = ["baltimore", "lancaster", "philadelphia", "richmond"]

    loja_sim = st.selectbox(
        "Loja",
        lojas,
        format_func=lambda x: x.capitalize(),
        key="sim_loja"
)
    max_unidades = st.slider("Máximo unidades", 5000, 20000, 10000)
    peso_lucro = st.slider("Peso lucro", 0.0, 1.0, 0.7)

    peso_hr = 1 - peso_lucro

    resultados = []

    for obj in ["O1", "O2", "O3"]:
        fname = REPORTS_DIR / f"plan_{loja_sim}_{obj}_20runs.csv"

        if fname.exists():
            df = pd.read_csv(fname)

            # 🔥 DETETAR COLUNA DE LUCRO
            colunas_possiveis = ["Profit", "Lucro", "profit", "lucro"]
            col_lucro = None

            for col in colunas_possiveis:
                if col in df.columns:
                    col_lucro = col
                    break

            if col_lucro is None:
                col_lucro = df.columns[-1]

            lucro = df[col_lucro].sum()
            unidades = df["Clientes"].sum() if "Clientes" in df.columns else 0
            hr = df["Experts"].sum() if "Experts" in df.columns else 0

            if unidades > max_unidades:
                lucro *= max_unidades / unidades

            score = peso_lucro * lucro - peso_hr * hr

            resultados.append((obj, lucro, unidades, hr, score))

    if resultados:
        st.table(pd.DataFrame(resultados, columns=["Objetivo", "Lucro", "Unidades", "HR", "Score"]))

        melhor = max(resultados, key=lambda x: x[4])
        st.success(f"Melhor objetivo: {melhor[0]}")
    else:
        st.warning("Sem dados disponíveis.")

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.markdown("Desenvolvido por GitHub Copilot · GPT-4.1 · Streamlit · Plotly")