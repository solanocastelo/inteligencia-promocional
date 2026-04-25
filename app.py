import io
import math
import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analytics import alertas_estoque, calcular_abc, prever_proximos_meses, simular_metas, sugerir_encarte
from charts import LAYOUT_BASE, area_temporal, barras_horizontais, gauge_meta, pareto_abc, yoy_multilinhas
from loader import MESES_ORDEM, carregar_pasta, carregar_uploads, carregar_google_drive

st.set_page_config(page_title="Inteligência Promocional Casa Freitas",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    div[data-testid="stMetric"] {
        background: rgba(66,133,244,0.06);
        border: 1px solid rgba(66,133,244,0.15);
        border-radius: 12px; padding: 16px 20px;
    }
    div[data-testid="stMetric"] label { font-size:0.82rem !important; text-transform:uppercase; opacity:0.75; }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] { font-size:1.55rem !important; font-weight:700; }
    button[data-baseweb="tab"] { font-size:0.9rem !important; padding:10px 22px !important; }
    hr { border-color:rgba(255,255,255,0.06) !important; margin:1.5rem 0 !important; }
    h3 { font-size:1.1rem !important; font-weight:600 !important; }
    div[data-baseweb="select"] { border-radius:8px; }
</style>
""", unsafe_allow_html=True)


def fmt(v):
    if v >= 1_000_000_000: return f"R$ {v/1_000_000_000:,.2f} Bi"
    if v >= 1_000_000: return f"R$ {v/1_000_000:,.2f} Mi"
    if v >= 1_000: return f"R$ {v/1_000:,.1f} Mil"
    return f"R$ {v:,.2f}"

def df_para_excel(df):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as w:
        df.to_excel(w, index=False, sheet_name="Dados")
    return buf.getvalue()

def delta_yoy(atual, anterior):
    if anterior == 0: return "—"
    return f"{(atual-anterior)/anterior*100:+.1f}% vs ano ant."


st.markdown("<h1 style='text-align:center;font-size:2rem;'>Inteligência Promocional Casa Freitas</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#777;font-size:0.9rem;'>Análise de performance · Simulação de metas · Previsão sazonal</p>", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<h2 style='color:#4285f4;font-size:1.2rem;'>Fonte de Dados</h2>", unsafe_allow_html=True)
    fonte = st.radio("Carregar de:", ["Google Drive", "Upload de arquivos"], horizontal=True)
    uploads = None
    if fonte == "Upload de arquivos":
        uploads = st.file_uploader("Selecione os .xlsx", type=["xlsx"], accept_multiple_files=True)
    st.divider()
    st.caption("© 2026 · Casa Freitas")


# ── Carregamento ──────────────────────────────────────────────────────────────
with st.spinner("Carregando dados..."):
    GDRIVE_FOLDER_ID = "1Ebf46TXLIkRv9TI9xhWAEPzWwjVWdfN0"
if fonte == "Upload de arquivos":
    df_raw = carregar_uploads(uploads) if uploads else pd.DataFrame()
else:
    df_raw = carregar_google_drive(GDRIVE_FOLDER_ID)


if df_raw.empty:
    st.info("Nenhum dado carregado. Use o upload ou verifique a pasta de dados.")
    st.stop()

# ── Opções de filtro ──────────────────────────────────────────────────────────
anos_disp = sorted(df_raw["Ano"].dropna().unique().astype(str).tolist())
meses_disp = [m for m in MESES_ORDEM if m in df_raw["Mês"].astype(str).unique()]
cats_disp = sorted(df_raw["Categoria"].dropna().unique().tolist()) if "Categoria" in df_raw.columns else []

with st.sidebar:
    st.markdown("<h2 style='color:#4285f4;font-size:1.2rem;'>Filtros Globais</h2>", unsafe_allow_html=True)
    anos_sel = st.multiselect("Ano(s)", options=anos_disp, default=[], placeholder="Todos os anos")
    meses_sel = st.multiselect("Mês(es)", options=meses_disp, default=[], placeholder="Todos os meses")
    cats_sel = st.multiselect("Categoria(s)", options=cats_disp, default=[], placeholder="Todas")
    st.divider()
    st.markdown("**Filtros das Abas**")
    mes_aba = st.selectbox("Mês (Ranking / Curva A):", options=meses_disp)
    cat_aba = st.selectbox("Categoria (Curva A):", options=cats_disp)

# ── Filtragem ─────────────────────────────────────────────────────────────────
df = df_raw.copy()
if anos_sel: df = df[df["Ano"].astype(str).isin(anos_sel)]
if meses_sel: df = df[df["Mês"].astype(str).isin(meses_sel)]
if cats_sel: df = df[df["Categoria"].isin(cats_sel)]

# ── KPIs ──────────────────────────────────────────────────────────────────────
total_venda = df["Venda"].sum() if "Venda" in df.columns else 0
n_produtos = df["Produto"].nunique() if "Produto" in df.columns else 0
n_cats = df["Categoria"].nunique() if "Categoria" in df.columns else 0
n_subcats = df["Subcategoria"].nunique() if "Subcategoria" in df.columns else 0

anos_num = sorted(df_raw["Ano"].dropna().unique())
delta_total = None
if len(anos_num) >= 2 and not anos_sel:
    v_at = df_raw[df_raw["Ano"] == anos_num[-1]]["Venda"].sum()
    v_an = df_raw[df_raw["Ano"] == anos_num[-2]]["Venda"].sum()
    delta_total = delta_yoy(v_at, v_an)

k1, k2, k3, k4 = st.columns(4, gap="medium")
k1.metric("Total de Vendas", fmt(total_venda), delta=delta_total)
k2.metric("Produtos Únicos", f"{n_produtos:,}")
k3.metric("Categorias", str(n_cats))
k4.metric("Subcategorias", str(n_subcats))
st.divider()

# ── Top 10 + Detalhe ──────────────────────────────────────────────────────────
col_bar, col_det = st.columns(2, gap="large")

with col_bar:
    st.subheader("Top 10 Subcategorias")
    if "Subcategoria" in df.columns and not df.empty:
        df_bar = df.groupby("Subcategoria", as_index=False)["Venda"].sum().sort_values("Venda", ascending=True).tail(10)
        df_bar["Texto"] = df_bar["Venda"].apply(fmt)
        evento = st.plotly_chart(barras_horizontais(df_bar, "Venda", "Subcategoria", "Texto",
                                 titulo="Clique numa barra para detalhar"),
                                 use_container_width=True, on_select="rerun", key="bar_top10")

with col_det:
    st.subheader("Detalhe por Subcategoria")
    if "Subcategoria" in df.columns and not df.empty:
        top10 = df.groupby("Subcategoria", as_index=False)["Venda"].sum().sort_values("Venda", ascending=False).head(10)["Subcategoria"].tolist()
        clicada = None
        try:
            pts = st.session_state["bar_top10"]["selection"]["points"]
            if pts: clicada = pts[0].get("y")
        except: pass
        if clicada and clicada in top10: st.session_state["subcat_dd"] = clicada
        elif "subcat_dd" not in st.session_state: st.session_state["subcat_dd"] = top10[0] if top10 else None
        subcat_sel = st.selectbox("Subcategoria:", options=top10, key="subcat_dd")
        if subcat_sel:
            ddet = df[df["Subcategoria"] == subcat_sel]
            gcols = [c for c in ["Código","Produto"] if c in ddet.columns]
            acols = [c for c in ["Venda","Quantidade"] if c in ddet.columns]
            if not ddet.empty and gcols and acols:
                dg = ddet.groupby(gcols, as_index=False)[acols].sum().sort_values("Venda", ascending=False).reset_index(drop=True)
                if "Quantidade" in dg and "Venda" in dg:
                    dg["Ticket Médio"] = (dg["Venda"] / dg["Quantidade"].replace(0, np.nan)).fillna(0)
                st.caption(f"**{len(dg)}** produto(s) · {subcat_sel}")
                st.dataframe(dg.style.format({"Venda":"R$ {:,.2f}","Ticket Médio":"R$ {:,.2f}","Quantidade":"{:,.0f}"}),
                             use_container_width=True, hide_index=True, height=380)
                st.download_button("⬇ Exportar Excel", data=df_para_excel(dg),
                                   file_name=f"detalhe_{subcat_sel}.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.divider()

# ── Evolução + YoY ────────────────────────────────────────────────────────────
col_ev, col_yoy = st.columns(2, gap="large")
with col_ev:
    st.subheader("Evolução Mensal")
    if "Mês" in df.columns and not df.empty:
        dl = df.groupby("Mês", observed=False)["Venda"].sum().reset_index().fillna(0)
        st.plotly_chart(area_temporal(dl, "Mês", "Venda"), use_container_width=True)

with col_yoy:
    st.subheader("Comparativo Ano a Ano")
    if "Mês" in df_raw.columns and "Ano" in df_raw.columns:
        dy = df_raw.groupby(["Ano","Mês"], observed=False)["Venda"].sum().reset_index()
        dy["Ano"] = dy["Ano"].astype(str)
        st.plotly_chart(yoy_multilinhas(dy, "Mês", "Venda", "Ano", titulo="Vendas mensais por ano"),
                        use_container_width=True)

st.divider()

# ── Abas ──────────────────────────────────────────────────────────────────────
tab_rank, tab_curva, tab_sim, tab_abc, tab_prev, tab_alerta, tab_encarte = st.tabs([
    "📊 Ranking do Mês","📈 Curva A","🎯 Simulador de Metas",
    "🔺 Curva ABC","🔮 Previsão","⚠️ Alertas de Estoque","📋 Sugestão de Encarte"
])

with tab_rank:
    st.header(f"Ranking — {mes_aba}")
    if mes_aba and "Categoria" in df_raw.columns:
        dr = df_raw[df_raw["Mês"].astype(str) == mes_aba]
        ranking = dr.groupby(["Categoria","Subcategoria"])["Venda"].sum().reset_index().sort_values("Venda", ascending=False).reset_index(drop=True)
        if len(anos_num) >= 2:
            vc = df_raw[(df_raw["Mês"].astype(str)==mes_aba)&(df_raw["Ano"]==anos_num[-1])].groupby(["Categoria","Subcategoria"])["Venda"].sum()
            va = df_raw[(df_raw["Mês"].astype(str)==mes_aba)&(df_raw["Ano"]==anos_num[-2])].groupby(["Categoria","Subcategoria"])["Venda"].sum()
            var = ((vc-va)/va.replace(0,np.nan)*100).fillna(0)
            ranking = ranking.merge(var.rename("Var. YoY (%)").reset_index(), on=["Categoria","Subcategoria"], how="left").fillna({"Var. YoY (%)":0})
        fmt_r = {"Venda":"R$ {:,.2f}"}
        if "Var. YoY (%)" in ranking.columns: fmt_r["Var. YoY (%)"] = "{:+.1f}%"
        st.dataframe(ranking.style.format(fmt_r), use_container_width=True, hide_index=True)
        st.download_button("⬇ Exportar Excel", data=df_para_excel(ranking),
                           file_name=f"ranking_{mes_aba}.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

with tab_curva:
    st.header(f"Curva A — {cat_aba} em {mes_aba}")
    if mes_aba and cat_aba:
        dc = df_raw[(df_raw["Mês"].astype(str)==mes_aba)&(df_raw["Categoria"]==cat_aba)]
        if not dc.empty and "Produto" in dc.columns:
            ca = dc.groupby("Produto")[["Venda","Quantidade"]].sum().reset_index().sort_values("Venda", ascending=False).reset_index(drop=True)
            ca["Ticket Médio"] = (ca["Venda"]/ca["Quantidade"].replace(0,np.nan)).fillna(0)
            ca["% Repres."] = ca["Venda"]/ca["Venda"].sum()
            ca["% Acum."] = ca["% Repres."].cumsum()
            ca["Classe"] = ca["% Acum."].apply(lambda x: "A" if x<=0.80 else ("B" if x<=0.95 else "C"))
            st.dataframe(ca.style.format({"Venda":"R$ {:,.2f}","Ticket Médio":"R$ {:,.2f}","Quantidade":"{:,.0f}","% Repres.":"{:.2%}","% Acum.":"{:.2%}"}),
                         use_container_width=True, hide_index=True)
            st.download_button("⬇ Exportar Excel", data=df_para_excel(ca),
                               file_name=f"curva_a_{cat_aba}_{mes_aba}.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        else: st.info("Nenhum dado para esta combinação.")

with tab_sim:
    st.header("Simulador de Metas")
    c1, c2 = st.columns(2, gap="large")
    mes_alvo = c1.selectbox("Mês Alvo", options=meses_disp, key="mes_alvo_sim")
    meta_global = c2.number_input("Meta Global (R$)", min_value=0.0, value=40_000_000.0, step=1_000_000.0, format="%.2f")
    st.divider()
    if mes_alvo and meta_global > 0:
        res = simular_metas(df_raw, mes_alvo, meta_global)
        if not res: st.info(f"Sem histórico para {mes_alvo}.")
        else:
            total_proj = res["por_produto"]["Meta de Venda (R$)"].sum()
            st.plotly_chart(gauge_meta(total_proj, meta_global, f"Meta {mes_alvo} — {fmt(meta_global)}"), use_container_width=True)
            st.divider()
            st.subheader("Foco por Categoria")
            top_list = list(res["top_cats"].items())
            for row_i in range(0, len(top_list), 5):
                cols = st.columns(min(5, len(top_list)-row_i), gap="medium")
                for ci, (cat, val) in enumerate(top_list[row_i:row_i+5]):
                    cols[ci].metric(cat, fmt(val), f"{val/meta_global*100:.1f}% da meta")
            st.divider()
            st.subheader("Projeção por Subcategoria")
            ts = res["por_subcat"][["Subcategoria","Categoria","Preço Médio Estimado","Meta de Venda (R$)","Quantidade Necessária (Un)"]].sort_values("Meta de Venda (R$)", ascending=False).reset_index(drop=True)
            st.dataframe(ts.style.format({"Preço Médio Estimado":"R$ {:,.2f}","Meta de Venda (R$)":"R$ {:,.2f}","Quantidade Necessária (Un)":"{:,.0f}"}), use_container_width=True, hide_index=True, height=400)
            st.download_button("⬇ Exportar", data=df_para_excel(ts), file_name=f"meta_subcat_{mes_alvo}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            st.divider()
            st.subheader("Projeção por Produto")
            tp = res["por_produto"][["Produto","Categoria","Subcategoria","Preço Médio Estimado","Meta de Venda (R$)","Quantidade Necessária (Un)"]].sort_values("Meta de Venda (R$)", ascending=False).reset_index(drop=True)
            st.dataframe(tp.style.format({"Preço Médio Estimado":"R$ {:,.2f}","Meta de Venda (R$)":"R$ {:,.2f}","Quantidade Necessária (Un)":"{:,.0f}"}), use_container_width=True, hide_index=True, height=400)
            st.download_button("⬇ Exportar", data=df_para_excel(tp), file_name=f"meta_produtos_{mes_alvo}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            st.divider()
            r1,r2,r3 = st.columns(3, gap="medium")
            r1.metric("Meta Global", fmt(meta_global))
            r2.metric("Total Projetado", fmt(total_proj))
            r3.metric("Peças Necessárias", f"{tp['Quantidade Necessária (Un)'].sum():,.0f}")

with tab_abc:
    st.header("Curva ABC (Pareto)")
    nivel = st.radio("Nível:", ["Categoria","Subcategoria","Produto"], horizontal=True)
    if nivel in df.columns and not df.empty:
        dabc = calcular_abc(df, nivel)
        dabc.rename(columns={nivel:"Categoria"}, inplace=True)
        cla = dabc[dabc["Classe"]=="A"]
        tot = dabc["Venda"].sum()
        ca1,ca2,ca3 = st.columns(3, gap="medium")
        ca1.metric("Itens Classe A", str(len(cla)))
        ca2.metric("Faturamento Classe A", fmt(cla["Venda"].sum()))
        ca3.metric("% do Total", f"{cla['Venda'].sum()/tot*100:.1f}%" if tot>0 else "—")
        st.plotly_chart(pareto_abc(dabc, titulo=f"Curva ABC por {nivel}"), use_container_width=True)
        st.dataframe(dabc[["Categoria","Venda","% Representatividade","% Acumulado","Classe"]].style.format({"Venda":"R$ {:,.2f}","% Representatividade":"{:.2%}","% Acumulado":"{:.2%}"}), use_container_width=True, hide_index=True)
        st.download_button("⬇ Exportar", data=df_para_excel(dabc), file_name=f"abc_{nivel.lower()}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

with tab_prev:
    st.header("Previsão Sazonal")
    n_prev = st.slider("Meses a prever:", 1, 6, 3)
    dp = prever_proximos_meses(df_raw, n_meses=n_prev)
    if dp.empty: st.info("São necessários pelo menos 4 meses de histórico.")
    else:
        dp["Rótulo"] = dp["Mês"].astype(str) + "/" + dp["Ano"].astype(str)
        dh = dp[dp["Tipo"]=="Histórico"]
        df_ = dp[dp["Tipo"]=="Previsão"]
        fig_p = go.Figure()
        fig_p.add_trace(go.Scatter(x=dh["Rótulo"], y=dh["Previsão de Vendas (R$)"], name="Histórico", mode="lines+markers", line=dict(color="#4285f4",width=2.5), marker=dict(size=6)))
        fig_p.add_trace(go.Scatter(x=df_["Rótulo"], y=df_["Previsão de Vendas (R$)"], name="Previsão", mode="lines+markers", line=dict(color="#fbbc04",width=2.5,dash="dot"), marker=dict(size=9,symbol="diamond")))
        fig_p.update_xaxes(showgrid=False, tickangle=-35)
        fig_p.update_yaxes(title_text="Vendas (R$)", showgrid=True, gridcolor="rgba(255,255,255,0.06)")
        fig_p.update_layout(**LAYOUT_BASE, title_text="Histórico + Previsão", height=400, legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1))
        st.plotly_chart(fig_p, use_container_width=True)
        prev_tab = df_[["Rótulo","Previsão de Vendas (R$)"]].copy()
        prev_tab.columns = ["Período","Previsão (R$)"]
        st.dataframe(prev_tab.style.format({"Previsão (R$)":"R$ {:,.2f}"}), use_container_width=True, hide_index=True)

with tab_alerta:
    st.header("Alertas de Estoque Crítico")
    cobertura = st.slider("Meses de cobertura mínima:", 1, 6, 2)
    alertas = alertas_estoque(df_raw, meses_cobertura=float(cobertura))
    if alertas.empty: st.success("Nenhum produto abaixo do limiar.")
    else:
        al1,al2 = st.columns(2, gap="medium")
        al1.metric("🔴 Crítico (< 1 mês)", str(len(alertas[alertas["Status"].str.startswith("🔴")])))
        al2.metric("🟡 Atenção", str(len(alertas[alertas["Status"].str.startswith("🟡")])))
        st.divider()
        st.dataframe(alertas.style.format({"Saldo":"{:,.0f}","Média Mensal (Un)":"{:,.1f}","Meses de Cobertura":"{:.1f}"}), use_container_width=True, hide_index=True, height=450)
        st.download_button("⬇ Exportar Alertas", data=df_para_excel(alertas), file_name="alertas_estoque.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

with tab_encarte:
    st.header("Sugestão de Encarte")
    st.caption("Score: 70% volume de vendas + 30% crescimento YoY. Hierarquia: Departamento → Categoria → Subcategoria → 3 Produtos · 12 por página.")

    ce1, ce2, ce3 = st.columns(3, gap="medium")
    mes_enc = ce1.selectbox("Mês de Referência", options=meses_disp, key="mes_enc")
    anos_enc = ce2.multiselect("Ano(s)", options=anos_disp,
                               default=[anos_disp[-1]] if anos_disp else [], key="anos_enc")
    n_pags = ce3.selectbox("Páginas do Encarte", options=[4, 8, 12, 16], key="n_pags")

    ma1, ma2, ma3 = st.columns(3, gap="medium")
    meta_anual_enc = ma1.number_input("Meta Anual (R$)", min_value=0.0, value=0.0,
                                      step=100_000.0, format="%.2f", key="meta_anual_enc")
    meta_mensal_enc = ma2.number_input("Meta do Mês (R$)", min_value=0.0, value=0.0,
                                       step=10_000.0, format="%.2f", key="meta_mensal_enc")
    usar_meta_enc = ma3.checkbox("Calcular qtd. sobre a meta", value=False, key="usar_meta_enc")

    meta_efetiva_enc = meta_mensal_enc if meta_mensal_enc > 0 else (meta_anual_enc / 12 if meta_anual_enc > 0 else 0)
    if usar_meta_enc and meta_anual_enc > 0 and meta_mensal_enc == 0:
        st.caption(f"Meta mensal calculada automaticamente: **R$ {meta_anual_enc/12:,.2f}** (anual ÷ 12)")

    filtrar_est = st.checkbox("Excluir produtos sem estoque", value=True, key="filtrar_est_enc")
    st.divider()

    if mes_enc:
        meta_param = meta_efetiva_enc if usar_meta_enc and meta_efetiva_enc > 0 else None
        res_enc = sugerir_encarte(df_raw, mes_enc, anos_enc, n_pags,
                                  filtrar_sem_estoque=filtrar_est, meta_mensal=meta_param)

        if not res_enc:
            st.info(f"Sem dados para {mes_enc}.")
        else:
            tem_dep_enc = res_enc.get('tem_dep', False)
            nivel_pag_enc = res_enc.get('nivel_pag', 'Categoria')

            ke1, ke2, ke3, ke4 = st.columns(4, gap="medium")
            ke1.metric("Páginas", str(n_pags))
            ke2.metric("Total de Produtos", str(res_enc['total_itens']))
            ke3.metric("Capa", "9 produtos · mista")
            ke4.metric("Organização", nivel_pag_enc)
            if meta_param:
                km1, km2 = st.columns(2, gap="medium")
                km1.metric("Meta do Mês", f"R$ {meta_param:,.2f}")
                km2.metric("Meta Anual", f"R$ {meta_anual_enc:,.2f}" if meta_anual_enc > 0 else "—")
            st.divider()

            fmt_enc = {
                "Venda": "R$ {:,.2f}", "Ticket Médio": "R$ {:,.2f}",
                "Meta Est. (R$)": "R$ {:,.2f}", "Quantidade": "{:,.0f}",
                "Qtd. Necessária": "{:,.0f}", "Crescimento YoY (%)": "{:+.1f}%"
            }

            # Colunas para exibição da capa (lista plana)
            capa_cols = [c for c in (
                ["Posição"] +
                (["Departamento"] if tem_dep_enc else []) +
                ["Categoria", "Subcategoria"] +
                ([c2 for c2 in ["Código"] if c2 in res_enc['capa'].columns]) +
                ["Produto", "Venda", "Quantidade", "Ticket Médio", "Crescimento YoY (%)", "Classe"] +
                (["Meta Est. (R$)", "Qtd. Necessária"] + (["Estoque OK?"] if "Estoque OK?" in res_enc['capa'].columns else []) if meta_param else [])
            ) if c in res_enc['capa'].columns]

            st.subheader("🏷️ Capa — 9 Produtos Destaque")
            st.dataframe(
                res_enc['capa'][capa_cols].style.format(
                    {k: v for k, v in fmt_enc.items() if k in capa_cols}
                ), use_container_width=True, hide_index=True
            )
            st.divider()

            # Colunas para exibição dentro de cada subcat (sem Cat/SubCat pois viram títulos)
            prod_detail_cols = [c for c in (
                ([c2 for c2 in ["Código"] if c2 in res_enc['paginas'].columns]) +
                ["Produto", "Venda", "Quantidade", "Ticket Médio", "Crescimento YoY (%)", "Classe"] +
                (["Meta Est. (R$)", "Qtd. Necessária"] + (["Estoque OK?"] if "Estoque OK?" in res_enc['paginas'].columns else []) if meta_param else [])
            ) if c in (res_enc['paginas'].columns if not res_enc['paginas'].empty else [])]

            st.subheader(f"📄 Páginas Internas — por {nivel_pag_enc}")
            if not res_enc['paginas'].empty:
                pags_unicas = res_enc['paginas']['Página'].unique()
                first_pag = pags_unicas[0] if len(pags_unicas) > 0 else None
                for pag in pags_unicas:
                    df_p = res_enc['paginas'][res_enc['paginas']['Página'] == pag]
                    with st.expander(f"📄 {pag} — {len(df_p)} produtos", expanded=(pag == first_pag)):
                        if 'Categoria' in df_p.columns:
                            for cat in df_p['Categoria'].unique():
                                df_cat = df_p[df_p['Categoria'] == cat]
                                st.markdown(f"**{cat}**")
                                if 'Subcategoria' in df_cat.columns:
                                    for subcat in df_cat['Subcategoria'].unique():
                                        df_sub = df_cat[df_cat['Subcategoria'] == subcat]
                                        show = [c for c in prod_detail_cols if c in df_sub.columns]
                                        st.caption(f"↳ {subcat}")
                                        st.dataframe(
                                            df_sub[show].style.format(
                                                {k: v for k, v in fmt_enc.items() if k in show}
                                            ), use_container_width=True, hide_index=True
                                        )
                                else:
                                    show = [c for c in prod_detail_cols if c in df_cat.columns]
                                    st.dataframe(
                                        df_cat[show].style.format(
                                            {k: v for k, v in fmt_enc.items() if k in show}
                                        ), use_container_width=True, hide_index=True
                                    )
                        else:
                            show = [c for c in prod_detail_cols if c in df_p.columns]
                            st.dataframe(
                                df_p[show].style.format(
                                    {k: v for k, v in fmt_enc.items() if k in show}
                                ), use_container_width=True, hide_index=True
                            )
            st.divider()

            st.subheader("📊 Resumo por Página")
            st.dataframe(
                res_enc['resumo'].style.format({"Venda_Ref": "R$ {:,.2f}"}),
                use_container_width=True, hide_index=True
            )
            st.divider()

            def gerar_excel_encarte(res):
                import re as _re
                extra = [c for c in ["Meta Est. (R$)", "Qtd. Necessária", "Estoque OK?"]
                         if c in res['completo'].columns]
                cols_exp = [c for c in (
                    ["Página", "Posição"] +
                    (["Departamento"] if res.get('tem_dep') else []) +
                    ["Categoria", "Subcategoria"] +
                    ([c2 for c2 in ["Código"] if c2 in res['completo'].columns]) +
                    ["Produto", "Venda", "Quantidade", "Ticket Médio",
                     "Crescimento YoY (%)", "Classe"] + extra
                ) if c in res['completo'].columns]
                buf = io.BytesIO()
                with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
                    res['completo'][cols_exp].to_excel(writer, sheet_name='Encarte Completo', index=False)
                    res['resumo'].to_excel(writer, sheet_name='Resumo', index=False)
                    capa_exp = [c for c in cols_exp if c in res['capa'].columns]
                    res['capa'][capa_exp].to_excel(writer, sheet_name='Capa', index=False)
                    if not res['paginas'].empty:
                        for pag in res['paginas']['Página'].unique():
                            safe = _re.sub(r'[\\/*?:\[\]]', '', str(pag))[:31]
                            df_p = res['paginas'][res['paginas']['Página'] == pag]
                            pag_exp = [c for c in cols_exp if c in df_p.columns]
                            df_p[pag_exp].to_excel(writer, sheet_name=safe, index=False)
                return buf.getvalue()

            st.download_button(
                "⬇️ Exportar Sugestão de Encarte (Excel)",
                data=gerar_excel_encarte(res_enc),
                file_name=f"encarte_{mes_enc}_{n_pags}pags.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )