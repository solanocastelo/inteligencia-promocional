import math
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from loader import MESES_ORDEM, MESES_NUM


def calcular_abc(df, coluna_grupo):
    resumo = (
        df.groupby(coluna_grupo, observed=True)["Venda"]
        .sum().reset_index()
        .sort_values("Venda", ascending=False)
        .reset_index(drop=True)
    )
    total = resumo["Venda"].sum()
    resumo["% Representatividade"] = resumo["Venda"] / total if total > 0 else 0
    resumo["% Acumulado"] = resumo["% Representatividade"].cumsum()
    resumo["Classe"] = resumo["% Acumulado"].apply(
        lambda x: "A" if x <= 0.80 else ("B" if x <= 0.95 else "C")
    )
    return resumo


def simular_metas(df, mes_alvo, meta_global):
    cols = ["Mês","Categoria","Subcategoria","Produto","Venda","Quantidade"]
    if not all(c in df.columns for c in cols): return {}
    df_mes = df[df["Mês"].astype(str) == mes_alvo].copy()
    if df_mes.empty: return {}
    total_hist = df_mes["Venda"].sum()
    proj = (
        df_mes.groupby(["Categoria","Subcategoria","Produto"], observed=True)
        .agg(Venda_Hist=("Venda","sum"), Qtd_Hist=("Quantidade","sum"))
        .reset_index()
    )
    proj["Ticket Médio"] = (proj["Venda_Hist"] / proj["Qtd_Hist"].replace(0, np.nan)).fillna(0)
    proj["Representatividade"] = proj["Venda_Hist"] / total_hist if total_hist > 0 else 0
    proj["Meta de Venda (R$)"] = proj["Representatividade"] * meta_global
    proj["Quantidade Necessária (Un)"] = proj.apply(
        lambda r: math.ceil(r["Meta de Venda (R$)"] / r["Ticket Médio"])
        if r["Ticket Médio"] > 0 else 0, axis=1
    )
    proj.rename(columns={"Ticket Médio":"Preço Médio Estimado"}, inplace=True)
    por_subcat = (
        proj.groupby(["Categoria","Subcategoria"], observed=True)
        .agg({"Meta de Venda (R$)":"sum","Quantidade Necessária (Un)":"sum",
              "Venda_Hist":"sum","Qtd_Hist":"sum"})
        .reset_index()
    )
    por_subcat["Preço Médio Estimado"] = (
        por_subcat["Venda_Hist"] / por_subcat["Qtd_Hist"].replace(0, np.nan)
    ).fillna(0)
    top_cats = (
        proj.groupby("Categoria", observed=True)["Meta de Venda (R$)"]
        .sum().sort_values(ascending=False).head(10)
    )
    return {"por_produto":proj, "por_subcat":por_subcat,
            "top_cats":top_cats, "total_hist":total_hist}


def prever_proximos_meses(df, n_meses=3):
    if "Período" not in df.columns or "Venda" not in df.columns: return pd.DataFrame()
    historico = (
        df.groupby(["Período","Mês","Ano","MêsNum"], observed=True)["Venda"]
        .sum().reset_index()
        .sort_values("Período").reset_index(drop=True)
    )
    if len(historico) < 4: return pd.DataFrame()
    historico["t"] = range(len(historico))
    dummies = pd.get_dummies(historico["MêsNum"], prefix="m", drop_first=True)
    X = pd.concat([historico[["t"]], dummies], axis=1).values
    y = historico["Venda"].values
    modelo = LinearRegression().fit(X, y)
    ultimo = historico.iloc[-1]
    ultimo_ano = int(ultimo["Ano"])
    ultimo_mes_num = int(ultimo["MêsNum"])
    previsoes = []
    for i in range(1, n_meses + 1):
        mes_num = (ultimo_mes_num - 1 + i) % 12 + 1
        ano = ultimo_ano + ((ultimo_mes_num - 1 + i) // 12)
        mes_nome = MESES_ORDEM[mes_num - 1]
        t_novo = len(historico) + i - 1
        row_dummy = {f"m_{m}": (1 if m == mes_num else 0) for m in range(2, 13)}
        x_novo = np.array([[t_novo] + [row_dummy.get(f"m_{m}", 0) for m in range(2, 13)]])
        venda_prev = max(float(modelo.predict(x_novo)[0]), 0)
        previsoes.append({"Mês":mes_nome,"Ano":ano,
                          "Previsão de Vendas (R$)":venda_prev,"Tipo":"Previsão"})
    hist_fmt = historico[["Mês","Ano","Venda"]].copy()
    hist_fmt.rename(columns={"Venda":"Previsão de Vendas (R$)"}, inplace=True)
    hist_fmt["Tipo"] = "Histórico"
    return pd.concat([hist_fmt, pd.DataFrame(previsoes)], ignore_index=True)


def alertas_estoque(df, meses_cobertura=2.0):
    cols_req = ["Código","Produto","Venda","Quantidade","Saldo","Período"]
    if not all(c in df.columns for c in cols_req): return pd.DataFrame()
    ultimo_periodo = df["Período"].max()
    saldo_atual = (
        df[df["Período"] == ultimo_periodo]
        .groupby(["Código","Produto"], as_index=False)["Saldo"].max()
    )
    media_mensal = (
        df.groupby(["Código","Produto","Período"], observed=True)["Quantidade"]
        .sum().reset_index()
        .groupby(["Código","Produto"], as_index=False)["Quantidade"].mean()
        .rename(columns={"Quantidade":"Média Mensal (Un)"})
    )
    merged = saldo_atual.merge(media_mensal, on=["Código","Produto"])
    merged["Meses de Cobertura"] = (
        merged["Saldo"] / merged["Média Mensal (Un)"].replace(0, np.nan)
    ).fillna(0)
    merged["Status"] = merged["Meses de Cobertura"].apply(
        lambda x: "🔴 Crítico" if x < 1 else ("🟡 Atenção" if x < meses_cobertura else "🟢 OK")
    )
    alerta = merged[merged["Meses de Cobertura"] < meses_cobertura].sort_values("Meses de Cobertura").reset_index(drop=True)
    return alerta[["Código","Produto","Saldo","Média Mensal (Un)","Meses de Cobertura","Status"]]

def sugerir_encarte(df, mes_ref, anos_ref, n_paginas, organizar_por='Subcategoria', filtrar_sem_estoque=True):
    PROD_CAPA = 9
    PROD_PAG = 12

    cols_req = ['Mês','Ano','Código','Produto','Categoria','Subcategoria','Venda','Quantidade']
    if not all(c in df.columns for c in cols_req): return {}

    mask = df['Mês'].astype(str) == mes_ref
    if anos_ref: mask &= df['Ano'].astype(str).isin([str(a) for a in anos_ref])
    df_ref = df[mask].copy()
    if df_ref.empty: return {}

    agg = {'Venda':'sum','Quantidade':'sum'}
    if 'Saldo' in df.columns: agg['Saldo'] = 'max'

    df_prod = (
        df_ref.groupby(['Código','Produto','Categoria','Subcategoria'], as_index=False)
        .agg(agg)
        .sort_values('Venda', ascending=False)
        .reset_index(drop=True)
    )
    df_prod['Preço Médio'] = (df_prod['Venda'] / df_prod['Quantidade'].replace(0, np.nan)).fillna(0)

    if filtrar_sem_estoque and 'Saldo' in df_prod.columns:
        df_prod = df_prod[df_prod['Saldo'] > 0].reset_index(drop=True)
    if df_prod.empty: return {}

    anos_disp = sorted(df['Ano'].dropna().unique())
    if len(anos_disp) >= 2:
        ano_ref_num = max([int(a) for a in anos_ref]) if anos_ref else int(anos_disp[-1])
        df_ant = df[(df['Mês'].astype(str) == mes_ref) & (df['Ano'] == ano_ref_num - 1)]
        if not df_ant.empty:
            v_ant = df_ant.groupby(['Código','Produto'])['Venda'].sum().reset_index().rename(columns={'Venda':'Venda_Ant'})
            df_prod = df_prod.merge(v_ant, on=['Código','Produto'], how='left')
            df_prod['Venda_Ant'] = df_prod['Venda_Ant'].fillna(0)
            df_prod['Crescimento YoY (%)'] = (
                (df_prod['Venda'] - df_prod['Venda_Ant']) /
                df_prod['Venda_Ant'].replace(0, np.nan) * 100
            ).fillna(0).clip(-100, 500)
        else:
            df_prod['Crescimento YoY (%)'] = 0.0
    else:
        df_prod['Crescimento YoY (%)'] = 0.0

    v_max = df_prod['Venda'].max()
    c_max = df_prod['Crescimento YoY (%)'].clip(lower=0).max()
    df_prod['Score'] = (
        0.70 * (df_prod['Venda'] / v_max if v_max > 0 else 0) +
        0.30 * (df_prod['Crescimento YoY (%)'].clip(lower=0) / c_max if c_max > 0 else 0)
    )

    total_v = df_prod['Venda'].sum()
    df_prod['% Repres.'] = df_prod['Venda'] / total_v if total_v > 0 else 0
    df_prod['% Acum.'] = df_prod['% Repres.'].cumsum()
    df_prod['Classe'] = df_prod['% Acum.'].apply(lambda x: 'A' if x<=0.80 else ('B' if x<=0.95 else 'C'))
    df_prod = df_prod.sort_values('Score', ascending=False).reset_index(drop=True)

    # Capa: produtos mais importantes com diversidade de categorias (mista)
    capa_idx = []
    for cat in df_prod['Categoria'].unique():
        if len(capa_idx) >= PROD_CAPA: break
        idx = df_prod[~df_prod.index.isin(capa_idx) & (df_prod['Categoria'] == cat)].index
        if len(idx) > 0: capa_idx.append(idx[0])
    for i in df_prod.index:
        if len(capa_idx) >= PROD_CAPA: break
        if i not in capa_idx: capa_idx.append(i)

    capa = df_prod.loc[capa_idx].copy().reset_index(drop=True)
    capa['Posição'] = range(1, len(capa)+1)
    capa['Página'] = 'Capa'

    # Páginas internas: uma página por subcategoria
    restante = df_prod.loc[[i for i in df_prod.index if i not in capa_idx]].copy()
    subcat_rank = (
        restante.groupby('Subcategoria')['Venda']
        .sum().sort_values(ascending=False).index.tolist()
    )

    max_pags_internas = n_paginas - 1
    rows = []
    pags_usadas = 0

    for subcat in subcat_rank:
        if pags_usadas >= max_pags_internas: break
        prods_subcat = restante[restante['Subcategoria'] == subcat].sort_values('Score', ascending=False)
        for chunk_start in range(0, len(prods_subcat), PROD_PAG):
            if pags_usadas >= max_pags_internas: break
            chunk = prods_subcat.iloc[chunk_start:chunk_start + PROD_PAG]
            page_label = subcat if chunk_start == 0 else f"{subcat} (cont.)"
            for pos, (_, row) in enumerate(chunk.iterrows()):
                r = row.copy()
                r['Página'] = page_label
                r['Posição'] = pos + 1
                rows.append(r)
            pags_usadas += 1

    df_pags = pd.DataFrame(rows).reset_index(drop=True) if rows else pd.DataFrame()
    df_completo = pd.concat([capa, df_pags], ignore_index=True)

    capa_resumo = pd.DataFrame([{
        'Página': 'Capa', 'Subcategoria': 'Mista',
        'Produtos': len(capa), 'Venda_Ref': capa['Venda'].sum()
    }])
    if not df_pags.empty:
        pags_resumo = (
            df_pags.groupby('Página', sort=False)
            .agg(Produtos=('Produto','count'), Venda_Ref=('Venda','sum'))
            .reset_index()
        )
        subcat_por_pag = df_pags.groupby('Página', sort=False)['Subcategoria'].first().reset_index()
        pags_resumo = pags_resumo.merge(subcat_por_pag, on='Página')
        resumo = pd.concat([capa_resumo, pags_resumo], ignore_index=True)
    else:
        resumo = capa_resumo

    return {'completo': df_completo, 'capa': capa, 'paginas': df_pags,
            'resumo': resumo, 'total_produtos': len(df_completo)}