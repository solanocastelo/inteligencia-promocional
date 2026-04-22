import os, glob, re
import pandas as pd
import streamlit as st

MESES_ORDEM = ["Janeiro","Fevereiro","Março","Abril","Maio","Junho",
               "Julho","Agosto","Setembro","Outubro","Novembro","Dezembro"]
MESES_NUM = {m: i+1 for i,m in enumerate(MESES_ORDEM)}

COL_MAP = {
    "Produto Nome":"Produto","Produto Cód.":"Código",
    "Venda Total Líquida":"Venda","%Repres.":"Repres",
    "Venda Quantidade":"Quantidade","Saldo Disponível Estoque":"Saldo",
    "Saldo Estoque Disponível Jabuti":"Saldo Jabuti",
    ">Hierarquia Produto Departamento - Produto Departamento":"Departamento",
    ">Hierarquia Produto Departamento - Produto Categoria":"Categoria",
    ">Hierarquia Produto Departamento - Produto Subcategoria":"Subcategoria",
    "Produto Procedência":"Procedência",
}
SUBCATEGORIAS_EXCLUIDAS = ["APARELHOS DE ACADEMIA"]

def _extrair_mes_ano(nome):
    base = os.path.splitext(os.path.basename(nome))[0]
    parts = base.split()
    if len(parts) >= 2:
        mes = parts[-2].capitalize()
        try:
            ano = int(parts[-1])
            if mes in MESES_ORDEM: return mes, ano
        except ValueError: pass
    for m in MESES_ORDEM:
        match = re.search(rf"\b{m}\b.*?(\d{{4}})", base, re.IGNORECASE)
        if match: return m, int(match.group(1))
    return "Indefinido", 0

def _limpar_venda(s):
    if s.dtype == object:
        return (s.astype(str)
                .str.replace("R$","",regex=False)
                .str.replace(" ","",regex=False)
                .str.replace(".","",regex=False)
                .str.replace(",",".",regex=False)
                .pipe(pd.to_numeric, errors="coerce")
                .fillna(0))
    return s.fillna(0)

def _ler_arquivo(path):
    mes, ano = _extrair_mes_ano(path)
    df = pd.read_excel(path, engine="openpyxl")
    df.rename(columns=COL_MAP, inplace=True)
    df["Mês"] = mes
    df["Ano"] = ano
    return df

def _ler_uploaded(f):
    mes, ano = _extrair_mes_ano(f.name)
    df = pd.read_excel(f, engine="openpyxl")
    df.rename(columns=COL_MAP, inplace=True)
    df["Mês"] = mes
    df["Ano"] = ano
    return df

@st.cache_data(show_spinner=False)
def carregar_pasta(pasta="dados"):
    arquivos = glob.glob(os.path.join(pasta, "*.xlsx"))
    if not arquivos: return pd.DataFrame()
    dfs = []
    for f in arquivos:
        try: dfs.append(_ler_arquivo(f))
        except Exception as e: st.warning(f"Erro ao ler {os.path.basename(f)}: {e}")
    return _pos_processar(pd.concat(dfs, ignore_index=True)) if dfs else pd.DataFrame()

def carregar_uploads(uploads):
    if not uploads: return pd.DataFrame()
    dfs = []
    for f in uploads:
        try: dfs.append(_ler_uploaded(f))
        except Exception as e: st.warning(f"Erro: {e}")
    return _pos_processar(pd.concat(dfs, ignore_index=True)) if dfs else pd.DataFrame()

def _pos_processar(df):
    if "Venda" in df.columns: df["Venda"] = _limpar_venda(df["Venda"])
    if "Quantidade" in df.columns: df["Quantidade"] = pd.to_numeric(df["Quantidade"], errors="coerce").fillna(0)
    if "Saldo" in df.columns: df["Saldo"] = pd.to_numeric(df["Saldo"], errors="coerce").fillna(0)
    if "Código" in df.columns: df["Código"] = df["Código"].astype(str)
    if "Subcategoria" in df.columns:
        df = df[~df["Subcategoria"].isin(SUBCATEGORIAS_EXCLUIDAS)].copy()
    cat = pd.CategoricalDtype(categories=MESES_ORDEM, ordered=True)
    if "Mês" in df.columns: df["Mês"] = df["Mês"].astype(cat)
    if "Ano" in df.columns and "Mês" in df.columns:
        df["MêsNum"] = df["Mês"].map(MESES_NUM).fillna(0).astype(int)
        df["Período"] = df["Ano"].astype(str) + "-" + df["MêsNum"].astype(str).str.zfill(2)
    return df
