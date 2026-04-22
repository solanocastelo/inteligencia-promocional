import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

COR_PRIMARIA = "#4285f4"
COR_ACCENT = "#669df6"
COR_BRANCO = "#FFFFFF"
COR_POSITIVO = "#34a853"

PALETA = ["#4285f4","#5a95f5","#669df6","#7baaf7","#93bbf9",
          "#aaccfa","#c2dcfb","#d9ebfc","#e8f1fd","#f5f9ff"]

LAYOUT_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#FAFAFA", size=13),
    title_font=dict(size=16, color="#FAFAFA"),
    margin=dict(l=20, r=20, t=50, b=30),
    hoverlabel=dict(bgcolor="#1E1E1E", font_size=13),
)


def barras_horizontais(df, x, y, texto_col=None, titulo="", altura=520):
    fig = px.bar(df, x=x, y=y, orientation="h", text=texto_col,
                 color_discrete_sequence=[COR_PRIMARIA])
    fig.update_traces(textposition="outside", cliponaxis=False,
                      marker_line_width=0,
                      hovertemplate="<b>%{y}</b><br>Vendas: R$ %{x:,.2f}<extra></extra>")
    fig.update_xaxes(title_text="", showgrid=False, showticklabels=False, zeroline=False)
    fig.update_yaxes(title_text="", showgrid=False)
    fig.update_layout(**LAYOUT_BASE, title_text=titulo, clickmode="event+select",
                      height=altura, uniformtext_minsize=8, uniformtext_mode="hide")
    return fig


def area_temporal(df, x, y, titulo="Vendas por Mês", altura=360):
    fig = px.area(df, x=x, y=y, markers=True,
                  color_discrete_sequence=[COR_PRIMARIA])
    fig.update_traces(line=dict(width=3), marker=dict(size=8, color=COR_ACCENT),
                      fillcolor="rgba(66,133,244,0.08)",
                      hovertemplate="<b>%{x}</b><br>Vendas: R$ %{y:,.2f}<extra></extra>")
    fig.update_xaxes(title_text="", showgrid=False)
    fig.update_yaxes(title_text="Total Vendas (R$)", showgrid=True,
                     gridcolor="rgba(255,255,255,0.06)", gridwidth=1)
    fig.update_layout(**LAYOUT_BASE, title_text=titulo, height=altura)
    return fig


def yoy_multilinhas(df, x, y, color, titulo="Comparativo Ano a Ano", altura=400):
    fig = px.line(df, x=x, y=y, color=color, markers=True,
                  color_discrete_sequence=PALETA)
    fig.update_traces(line=dict(width=2.5), marker=dict(size=7))
    fig.update_xaxes(title_text="", showgrid=False)
    fig.update_yaxes(title_text="Total Vendas (R$)", showgrid=True,
                     gridcolor="rgba(255,255,255,0.06)", gridwidth=1)
    fig.update_layout(**LAYOUT_BASE, title_text=titulo, height=altura,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig


def pareto_abc(df_abc, titulo="Curva ABC — Faturamento por Categoria", altura=480):
    mapa_cores = {"A": COR_PRIMARIA, "B": COR_ACCENT, "C": "#888888"}
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(x=df_abc["Categoria"], y=df_abc["Venda"], name="Faturamento",
               marker_color=df_abc["Classe"].map(mapa_cores).tolist(),
               hovertemplate="<b>%{x}</b><br>Vendas: R$ %{y:,.2f}<extra></extra>"),
        secondary_y=False
    )
    fig.add_trace(
        go.Scatter(x=df_abc["Categoria"], y=df_abc["% Acumulado"], name="% Acumulado",
                   mode="lines+markers", line=dict(color=COR_BRANCO, width=2.5),
                   marker=dict(size=7, color=COR_BRANCO),
                   hovertemplate="<b>%{x}</b><br>Acumulado: %{y:.1%}<extra></extra>"),
        secondary_y=True
    )
    fig.add_hline(y=0.80, line_dash="dot", line_color="rgba(255,255,255,0.3)",
                  annotation_text="80%", annotation_position="top right", secondary_y=True)
    fig.add_hline(y=0.95, line_dash="dot", line_color="rgba(255,255,255,0.15)",
                  annotation_text="95%", annotation_position="top right", secondary_y=True)
    fig.update_xaxes(title_text="", showgrid=False, tickangle=-45)
    fig.update_yaxes(title_text="Faturamento (R$)", showgrid=False, secondary_y=False)
    fig.update_yaxes(title_text="% Acumulado", showgrid=False, tickformat=".0%",
                     range=[0, 1.05], secondary_y=True)
    fig.update_layout(**LAYOUT_BASE, title_text=titulo, height=altura, showlegend=True,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig


def gauge_meta(valor, meta, titulo="Progresso da Meta"):
    pct = min(valor / meta, 1.5) if meta > 0 else 0
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=valor,
        delta={"reference": meta, "valueformat": ",.0f"},
        title={"text": titulo, "font": {"color": "#FAFAFA", "size": 14}},
        number={"prefix": "R$ ", "valueformat": ",.0f", "font": {"color": "#FAFAFA"}},
        gauge={
            "axis": {"range": [0, meta * 1.2], "tickcolor": "#FAFAFA"},
            "bar": {"color": COR_PRIMARIA if pct < 1 else COR_POSITIVO},
            "bgcolor": "rgba(0,0,0,0)",
            "steps": [
                {"range": [0, meta * 0.7], "color": "rgba(232,69,60,0.15)"},
                {"range": [meta * 0.7, meta], "color": "rgba(66,133,244,0.15)"},
                {"range": [meta, meta * 1.2], "color": "rgba(52,168,83,0.15)"},
            ],
            "threshold": {"line": {"color": COR_BRANCO, "width": 2},
                          "thickness": 0.75, "value": meta}
        }
    ))
    fig.update_layout(**LAYOUT_BASE, height=280)
    return fig
