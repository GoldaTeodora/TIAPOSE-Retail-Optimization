import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="Intelligent Retail DSS", layout="wide")

st.title("🧠 Intelligent Decision Support System for USA Stores")

# Sidebar - Global filters
def sidebar():
    st.sidebar.header("Filtros Globais")
    store = st.sidebar.selectbox("Loja", ["baltimore", "lancaster", "philadelphia", "richmond"], format_func=lambda x: x.capitalize())
    week = st.sidebar.number_input("Semana alvo (nº)", min_value=1, value=1)
    return store, week

store, week = sidebar()

# Tabs for navigation
tab1, tab2, tab3, tab4 = st.tabs([
    "Previsão de Clientes",
    "Otimização de Planos",
    "Comparação de Métodos",
    "Simulação e Decisão"
])


with tab1:
    st.header("Previsão de Clientes por Loja e Método")
    st.info("Aqui o gestor pode analisar previsões, comparar métodos e ver a qualidade das previsões.")

    # Carregar dados de rolling_optimum_[loja].csv (valores reais) e rolling_optimum_[loja]_O1/O2/O3.csv (previsões)
    lojas = ["baltimore", "lancaster", "philadelphia", "richmond"]
    report_dir = Path("reports")
    loja_prev = st.selectbox("Loja", lojas, format_func=lambda x: x.capitalize(), key="prev_loja")
    df_real = None
    df_preds = {}
    # Real
    real_path = report_dir / f"rolling_optimum_{loja_prev}.csv"
    if real_path.exists():
        df_real = pd.read_csv(real_path, parse_dates=["start_date", "end_date"])
    # Previsões por objetivo
    for obj in ["O1", "O2", "O3"]:
        pred_path = report_dir / f"rolling_optimum_{loja_prev}_{obj}.csv"
        if pred_path.exists():
            df_preds[obj] = pd.read_csv(pred_path, parse_dates=["start_date", "end_date"])
        else:
            df_preds[obj] = None

    # Gráfico previsão vs real (clientes_previstos vs clientes_reais)
    st.subheader(f"Previsão vs Real de Clientes - {loja_prev.capitalize()}")
    fig = go.Figure()
    if df_real is not None:
        fig.add_trace(go.Scatter(x=df_real["start_date"], y=df_real["clientes_reais"].apply(lambda x: sum(eval(str(x))) if pd.notnull(x) else None),
                                 mode="lines+markers", name="Real (soma semanal)"))
    for obj, df in df_preds.items():
        if df is not None:
            fig.add_trace(go.Scatter(x=df["start_date"], y=df["clientes_previstos"].apply(lambda x: sum(eval(str(x))) if pd.notnull(x) else None),
                                     mode="lines+markers", name=f"Previsto {obj}"))
    fig.update_layout(xaxis_title="Data", yaxis_title="Clientes", template="plotly_white")
    st.plotly_chart(fig, width='stretch')

    # Métricas de qualidade (RMSE, MAE, R2) para cada objetivo
    import numpy as np
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    st.subheader("Métricas de Qualidade da Previsão (soma semanal)")
    if df_real is not None:
        y_true = df_real["clientes_reais"].apply(lambda x: sum(eval(str(x))) if pd.notnull(x) else None).values
        for obj, df in df_preds.items():
            if df is not None:
                y_pred = df["clientes_previstos"].apply(lambda x: sum(eval(str(x))) if pd.notnull(x) else None).values
                mask = (~pd.isnull(y_true)) & (~pd.isnull(y_pred))
                if mask.sum() > 0:
                    # Corrigir RMSE para compatibilidade com versões antigas do scikit-learn
                    rmse = np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))
                    mae = mean_absolute_error(y_true[mask], y_pred[mask])
                    r2 = r2_score(y_true[mask], y_pred[mask])
                    st.markdown(f"**{obj}:** RMSE = {rmse:.2f} | MAE = {mae:.2f} | R² = {r2:.2f}")
                else:
                    st.markdown(f"**{obj}:** Sem dados suficientes para calcular métricas.")
            else:
                st.markdown(f"**{obj}:** Sem previsões disponíveis.")


with tab2:
    st.header("Otimização de Planos Semanais")
    st.info("Visualize e compare planos otimizados para cada objetivo (O1, O2, O3). Analise lucro, unidades, HR, etc.")

    # Mapear lojas e caminhos dos arquivos
    lojas = ["baltimore", "lancaster", "philadelphia", "richmond"]
    objetivos = ["O1", "O2", "O3"]
    report_dir = Path("reports")
    dados = {}
    for loja in lojas:
        dados[loja] = {}
        for obj in objetivos:
            fname = report_dir / f"rolling_optimum_{loja}_{obj}.csv"
            if fname.exists():
                df = pd.read_csv(fname, parse_dates=["start_date", "end_date"])
                dados[loja][obj] = df
            else:
                dados[loja][obj] = None

    # Seleção de loja e objetivos
    col1, col2 = st.columns(2)
    with col1:
        loja_sel = st.selectbox("Loja", lojas, format_func=lambda x: x.capitalize(), key="opt_loja")
    with col2:
        objetivos_sel = st.multiselect("Objetivos", objetivos, default=objetivos, key="opt_obj")

    # Mostrar gráficos comparativos
    st.subheader(f"Lucro ótimo semanal - {loja_sel.capitalize()}")
    fig_lucro = go.Figure()
    for obj in objetivos_sel:
        df = dados[loja_sel][obj]
        if df is not None:
            fig_lucro.add_trace(go.Scatter(x=df["start_date"], y=df["lucro_otimo"], mode="lines+markers", name=obj))
    fig_lucro.update_layout(xaxis_title="Data", yaxis_title="Lucro ótimo", template="plotly_white")
    st.plotly_chart(fig_lucro, width='stretch')

    st.subheader(f"Unidades vendidas por semana - {loja_sel.capitalize()}")
    fig_unid = go.Figure()
    for obj in objetivos_sel:
        df = dados[loja_sel][obj]
        if df is not None:
            fig_unid.add_trace(go.Scatter(x=df["start_date"], y=df["unidades"], mode="lines+markers", name=obj))
    fig_unid.update_layout(xaxis_title="Data", yaxis_title="Unidades", template="plotly_white")
    st.plotly_chart(fig_unid, width='stretch')

    st.subheader(f"Custo de Recursos Humanos por semana - {loja_sel.capitalize()}")
    fig_hr = go.Figure()
    for obj in objetivos_sel:
        df = dados[loja_sel][obj]
        if df is not None:
            fig_hr.add_trace(go.Scatter(x=df["start_date"], y=df["hr_cost"], mode="lines+markers", name=obj))
    fig_hr.update_layout(xaxis_title="Data", yaxis_title="Custo HR", template="plotly_white")
    st.plotly_chart(fig_hr, width='stretch')

    # Tabela detalhada
    st.subheader(f"Tabela detalhada ({loja_sel.capitalize()})")
    for obj in objetivos_sel:
        df = dados[loja_sel][obj]
        if df is not None:
            st.markdown(f"**Objetivo {obj}:**")
            st.dataframe(df, use_container_width=True, height=300)
        else:
            st.warning(f"Sem dados para {loja_sel.capitalize()} - {obj}")


with tab3:
    st.header("Comparação de Métodos e Resultados")
    st.info("Compare métodos de previsão e otimização, veja gráficos de desempenho e escolha o melhor.")

    # Ranking de métodos de previsão por loja
    lojas = ["baltimore", "lancaster", "philadelphia", "richmond"]
    report_dir = Path("reports")
    metricas = {"RMSE": [], "MAE": [], "R2": []}
    for loja in lojas:
        real_path = report_dir / f"rolling_optimum_{loja}.csv"
        if real_path.exists():
            df_real = pd.read_csv(real_path, parse_dates=["start_date", "end_date"])
            y_true = df_real["clientes_reais"].apply(lambda x: sum(eval(str(x))) if pd.notnull(x) else None).values
            for obj in ["O1", "O2", "O3"]:
                pred_path = report_dir / f"rolling_optimum_{loja}_{obj}.csv"
                if pred_path.exists():
                    df_pred = pd.read_csv(pred_path, parse_dates=["start_date", "end_date"])
                    y_pred = df_pred["clientes_previstos"].apply(lambda x: sum(eval(str(x))) if pd.notnull(x) else None).values
                    mask = (~pd.isnull(y_true)) & (~pd.isnull(y_pred))
                    if mask.sum() > 0:
                        import numpy as np
                        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                        rmse = np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))
                        mae = mean_absolute_error(y_true[mask], y_pred[mask])
                        r2 = r2_score(y_true[mask], y_pred[mask])
                        metricas["RMSE"].append((loja, obj, rmse))
                        metricas["MAE"].append((loja, obj, mae))
                        metricas["R2"].append((loja, obj, r2))

    st.subheader("Ranking de Métodos de Previsão por Loja e Objetivo")
    for metrica, valores in metricas.items():
        if valores:
            st.markdown(f"**{metrica}:**")
            if metrica == "R2":
                valores_ordenados = sorted(valores, key=lambda x: -x[2])
            else:
                valores_ordenados = sorted(valores, key=lambda x: x[2])
            st.table(pd.DataFrame(valores_ordenados, columns=["Loja", "Objetivo", metrica]))
        else:
            st.warning(f"Sem dados para {metrica}")

    st.subheader("Resumo de Desempenho dos Planos Otimizados")
    # Resumo de lucro, unidades, HR por loja e objetivo
    resumo = []
    for loja in lojas:
        for obj in ["O1", "O2", "O3"]:
            fname = report_dir / f"rolling_optimum_{loja}_{obj}.csv"
            if fname.exists():
                df = pd.read_csv(fname, parse_dates=["start_date", "end_date"])
                lucro = df["lucro_otimo"].sum()
                unidades = df["unidades"].sum()
                hr = df["hr_cost"].sum()
                resumo.append((loja, obj, lucro, unidades, hr))
    if resumo:
        st.table(pd.DataFrame(resumo, columns=["Loja", "Objetivo", "Lucro Total", "Unidades Totais", "Custo HR Total"]))
    else:
        st.warning("Sem dados de planos otimizados.")


with tab4:
    st.header("Simulação e Suporte à Decisão")
    st.info("Simule decisões, ajuste restrições e veja o impacto nos resultados. Exporte relatórios e gráficos.")

    # Simulação simples: ajuste de restrições e pesos para O3
    lojas = ["baltimore", "lancaster", "philadelphia", "richmond"]
    loja_sim = st.selectbox("Loja para simulação", lojas, format_func=lambda x: x.capitalize(), key="sim_loja")
    max_unidades = st.slider("Máximo de unidades vendidas (restrição O2)", min_value=5000, max_value=20000, value=10000, step=500)
    peso_lucro = st.slider("Peso do lucro (O3)", min_value=0.0, max_value=1.0, value=0.7, step=0.05)
    peso_hr = 1.0 - peso_lucro
    st.markdown(f"Peso do custo HR: **{peso_hr:.2f}** (O3)")

    # Carregar dados O1, O2, O3
    report_dir = Path("reports")
    dfs = {}
    for obj in ["O1", "O2", "O3"]:
        fname = report_dir / f"rolling_optimum_{loja_sim}_{obj}.csv"
        if fname.exists():
            dfs[obj] = pd.read_csv(fname, parse_dates=["start_date", "end_date"])
        else:
            dfs[obj] = None

    # Simulação: mostrar impacto dos pesos e restrições (apenas para a semana mais recente)
    st.subheader("Impacto das Decisões nos Resultados (Semana mais recente)")
    resultados = []
    for obj, df in dfs.items():
        if df is not None and len(df) > 0:
            row = df.iloc[-1]  # última semana disponível
            unidades = row["unidades"]
            lucro = row["lucro_otimo"]
            hr = row["hr_cost"]
            if unidades > max_unidades:
                lucro = lucro * (max_unidades / unidades)  # penalização simples
                unidades = max_unidades
            score = peso_lucro * lucro - peso_hr * hr
            resultados.append((obj, lucro, unidades, hr, score))
    if resultados:
        st.table(pd.DataFrame(resultados, columns=["Objetivo", "Lucro", "Unidades", "Custo HR", "Score Simulado"]))
        melhor = max(resultados, key=lambda x: x[4])
        st.success(f"Melhor objetivo para os parâmetros escolhidos: **{melhor[0]}** (Score: {melhor[4]:.2f})")
    else:
        st.warning("Sem dados para simulação nesta loja.")

st.markdown("---")
st.markdown("Desenvolvido por GitHub Copilot · GPT-4.1 · Streamlit · Plotly")
