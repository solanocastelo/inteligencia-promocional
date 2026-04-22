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
