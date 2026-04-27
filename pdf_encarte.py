import io
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.platypus import (SimpleDocTemplate, Table, TableStyle,
                                  Paragraph, Spacer, PageBreak, Flowable)

C_AZUL      = colors.HexColor('#1565C0')
C_AZUL_MED  = colors.HexColor('#1976D2')
C_AZUL_CLR  = colors.HexColor('#BBDEFB')
C_CINZA_CLR = colors.HexColor('#F5F5F5')
C_CINZA     = colors.HexColor('#E0E0E0')
C_BRANCO    = colors.white
C_PRETO     = colors.black
C_CINZA_MED = colors.HexColor('#9E9E9E')

PAGE_W, PAGE_H = A4
MARGIN = 12 * mm


def _fmt_r(v):
    try:
        v = float(v)
        if v >= 1_000_000: return f"R${v/1_000_000:.1f}M"
        if v >= 1_000: return f"R${v/1_000:.1f}K"
        return f"R${v:,.0f}"
    except: return "—"

def _fmt_pct(v):
    try: return f"{float(v):+.1f}%"
    except: return "—"

def _fmt_n(v):
    try: return f"{float(v):,.0f}"
    except: return "—"

def _para(text, size=7, bold=False, color=C_PRETO, align=TA_LEFT):
    style = ParagraphStyle('s', fontSize=size, leading=size + 2,
                            alignment=align, textColor=color)
    t = f"<b>{text}</b>" if bold else str(text)
    return Paragraph(t, style)


class RotLabel(Flowable):
    """Preenche a célula com cor de fundo e texto rotacionado centralizado."""
    def __init__(self, text, bg=C_AZUL, fg=C_BRANCO, fontsize=8):
        super().__init__()
        self.text = str(text)[:24]
        self.bg = bg
        self.fg = fg
        self.fontsize = fontsize

    def wrap(self, w, h):
        self.width = w
        self.height = h
        return w, h

    def draw(self):
        cv = self.canv
        cv.saveState()
        cv.setFillColor(self.bg)
        cv.rect(0, 0, self.width, self.height, fill=1, stroke=0)
        cv.setFillColor(self.fg)
        cv.setFont('Helvetica-Bold', self.fontsize)
        cv.translate(self.width / 2, self.height / 2)
        cv.rotate(90)
        cv.drawCentredString(0, -self.fontsize * 0.3, self.text)
        cv.restoreState()


def _card(row, col_w, show_extra=False):
    def g(col, default='—'):
        try:
            v = row[col]
            return v if pd.notna(v) else default
        except: return default

    nome = str(g('Produto', '—'))[:26]
    h_sty = ParagraphStyle('h', fontSize=6.5, leading=8,
                            alignment=TA_CENTER, textColor=C_BRANCO)
    f_sty = ParagraphStyle('f', fontSize=5.5, leading=7,
                            alignment=TA_LEFT, textColor=C_PRETO)

    linhas = [
        [Paragraph(f'<b>{nome}</b>', h_sty)],
        [Paragraph(f'Cód.: {str(g("Código", "—"))}', f_sty)],
        [Paragraph(f'Ticket: {_fmt_r(g("Ticket Médio", 0))}', f_sty)],
        [Paragraph(f'YoY: {_fmt_pct(g("Crescimento YoY (%)", 0))}', f_sty)],
        [Paragraph(f'Curva: {str(g("Classe", "—"))}', f_sty)],
        [Paragraph(f'Saldo: {_fmt_n(g("Saldo", 0))}', f_sty)],
    ]
    if show_extra and 'Meta Est. (R$)' in row.index:
        linhas.append([Paragraph(f'Meta: {_fmt_r(g("Meta Est. (R$)", 0))}', f_sty)])
    if show_extra and 'Estoque OK?' in row.index:
        linhas.append([Paragraph(f'Est.: {str(g("Estoque OK?", ""))}', f_sty)])

    t = Table(linhas, colWidths=[col_w - 4])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), C_AZUL_MED),
        ('BACKGROUND', (0, 1), (-1, -1), C_CINZA_CLR),
        ('BOX', (0, 0), (-1, -1), 0.5, C_AZUL_CLR),
        ('INNERGRID', (0, 0), (-1, -1), 0.25, C_CINZA),
        ('TOPPADDING', (0, 0), (-1, -1), 1),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 1),
        ('LEFTPADDING', (0, 0), (-1, -1), 3),
        ('RIGHTPADDING', (0, 0), (-1, -1), 2),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    return t


def _card_vazio(col_w):
    t = Table([[_para('—', size=6, color=C_CINZA_MED, align=TA_CENTER)]],
              colWidths=[col_w - 4])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), C_CINZA_CLR),
        ('BOX', (0, 0), (-1, -1), 0.3, C_CINZA),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    return t


def _build_capa(df_capa, mes_ref, usable_w):
    story = []
    story.append(_para("SUGESTÃO DE ENCARTE", size=14, bold=True,
                        color=C_AZUL, align=TA_CENTER))
    story.append(_para(f"Referência: {mes_ref}   ·   9 Produtos Destaque (Produtos Isca)",
                        size=8, color=C_CINZA_MED, align=TA_CENTER))
    story.append(Spacer(1, 5 * mm))

    # Área reservada para logo/imagem
    logo = Table([[_para("[ Logo / Imagem da Campanha ]", size=9,
                          color=C_CINZA_MED, align=TA_CENTER)]],
                 colWidths=[usable_w], rowHeights=[38 * mm])
    logo.setStyle(TableStyle([
        ('BOX', (0, 0), (-1, -1), 1, C_AZUL_CLR),
        ('BACKGROUND', (0, 0), (-1, -1), C_CINZA_CLR),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(logo)
    story.append(Spacer(1, 4 * mm))

    cw = usable_w / 3
    has_extra = 'Meta Est. (R$)' in df_capa.columns or 'Estoque OK?' in df_capa.columns
    prods = df_capa.reset_index(drop=True)

    linhas = []
    for ri in range(3):
        linha = []
        for ci in range(3):
            idx = ri * 3 + ci
            if idx < len(prods):
                linha.append(_card(prods.iloc[idx], cw, has_extra))
            else:
                linha.append(_card_vazio(cw))
        linhas.append(linha)

    grid = Table(linhas, colWidths=[cw] * 3)
    grid.setStyle(TableStyle([
        ('BOX', (0, 0), (-1, -1), 0.5, C_AZUL_CLR),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, C_AZUL_CLR),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ('LEFTPADDING', (0, 0), (-1, -1), 2),
        ('RIGHTPADDING', (0, 0), (-1, -1), 2),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    story.append(grid)
    return story


def _build_pagina(pag_name, df_pag, tem_dep, usable_w):
    story = []
    story.append(_para(f"ENCARTE  ·  {pag_name.upper()}", size=10, bold=True, color=C_AZUL))
    story.append(Spacer(1, 2 * mm))

    has_extra = 'Meta Est. (R$)' in df_pag.columns or 'Estoque OK?' in df_pag.columns
    dep_w = 11 * mm if tem_dep else 0
    cat_w = 11 * mm
    sub_w = 13 * mm
    prod_w = (usable_w - dep_w - cat_w - sub_w) / 3
    col_widths = ([dep_w] if tem_dep else []) + [cat_w, sub_w, prod_w, prod_w, prod_w]

    cats = list(df_pag['Categoria'].unique()) if 'Categoria' in df_pag.columns else [pag_name]
    linhas = []
    spans = []
    r = 0

    for cat in cats:
        df_cat = df_pag[df_pag['Categoria'] == cat] if 'Categoria' in df_pag.columns else df_pag
        subcats = list(df_cat['Subcategoria'].unique()) if 'Subcategoria' in df_cat.columns else ['']
        cat_start = r

        for si, subcat in enumerate(subcats):
            df_sub = (df_cat[df_cat['Subcategoria'] == subcat]
                      if 'Subcategoria' in df_cat.columns else df_cat).reset_index(drop=True)
            prods = [df_sub.iloc[i] if i < len(df_sub) else None for i in range(3)]
            cards = [_card(p, prod_w, has_extra) if p is not None else _card_vazio(prod_w)
                     for p in prods]

            cat_cell = RotLabel(cat, C_AZUL_MED, C_BRANCO, 7) if si == 0 else ''
            sub_cell = RotLabel(subcat, C_AZUL_CLR, C_AZUL, 6.5)

            if tem_dep:
                dep_cell = RotLabel(pag_name, C_AZUL, C_BRANCO, 8) if r == 0 else ''
                linhas.append([dep_cell, cat_cell, sub_cell] + cards)
            else:
                linhas.append([cat_cell, sub_cell] + cards)
            r += 1

        cat_col = 1 if tem_dep else 0
        if len(subcats) > 1:
            spans.append(('SPAN', (cat_col, cat_start), (cat_col, r - 1)))

    if tem_dep and r > 1:
        spans.append(('SPAN', (0, 0), (0, r - 1)))

    avail_h = PAGE_H - 2 * MARGIN - 16 * mm
    row_h = avail_h / max(r, 1)

    tbl = Table(linhas, colWidths=col_widths, rowHeights=[row_h] * r)
    tbl.setStyle(TableStyle([
        ('BOX', (0, 0), (-1, -1), 1, C_AZUL),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, C_AZUL_CLR),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ('LEFTPADDING', (0, 0), (-1, -1), 2),
        ('RIGHTPADDING', (0, 0), (-1, -1), 2),
    ] + spans))
    story.append(tbl)
    return story


def gerar_pdf_encarte(res_enc, mes_ref):
    """Gera o PDF completo do encarte. Retorna bytes."""
    buf = io.BytesIO()
    usable_w = PAGE_W - 2 * MARGIN
    tem_dep = res_enc.get('tem_dep', False)

    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=MARGIN, rightMargin=MARGIN,
                            topMargin=MARGIN, bottomMargin=MARGIN,
                            title=f"Encarte {mes_ref}")
    story = []

    story += _build_capa(res_enc['capa'], mes_ref, usable_w)

    if not res_enc['paginas'].empty:
        for pag in res_enc['paginas']['Página'].unique():
            story.append(PageBreak())
            df_pag = res_enc['paginas'][res_enc['paginas']['Página'] == pag]
            story += _build_pagina(pag, df_pag, tem_dep, usable_w)

    doc.build(story)
    buf.seek(0)
    return buf.getvalue()