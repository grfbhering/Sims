"""SimTrigo — Trigo/Ferro two-good dashboard.

The dashboard follows the SimTrigo interaction pattern:
- A, l and X are the first model inputs.
- rho(A), R and B are calculated immediately from those inputs.
- shocks use one dropdown containing the model parameters.
- indexation mechanisms are independently activated with checkboxes.
- the b-r diagram plots the actual simulated pairs (m_t, b_t), with b on the vertical axis and aggregate real profit margin m on the horizontal axis.
"""
from __future__ import annotations

import io
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
)
import streamlit as st

from model import simulate_two_good_model, simulate_one_good_special_case

st.set_page_config(page_title="Simulator Tabajara", page_icon="📈", layout="wide")

st.markdown("""
<style>
.block-container{max-width:1800px;padding-top:2.8rem;padding-bottom:1.4rem;padding-left:1rem;padding-right:1rem}
.tab-header{margin-top:0;background:linear-gradient(90deg,#163f68 0%,#245b89 100%);color:#fff;margin:0 -1rem 1rem;padding:2.05rem 2.25rem 1.35rem;border-bottom:1px solid #173b5f;overflow:visible}
[data-testid="stDownloadButton"]{width:fit-content!important;display:block!important;margin:0 auto!important}
[data-testid="stDownloadButton"] button{border-radius:5px!important;border:1px solid #17365d!important;background:#17365d!important;background-color:#17365d!important;color:#fff!important;font-weight:700!important;box-shadow:0 1px 2px rgba(0,0,0,.12)!important;padding:.45rem .9rem!important;width:auto!important;min-width:0!important;display:inline-flex!important;white-space:nowrap!important}
[data-testid="stDownloadButton"] button p,[data-testid="stDownloadButton"] button span{color:#fff!important}
[data-testid="stDownloadButton"] button:hover{border-color:#244a78!important;background:#244a78!important;color:#fff!important}
[data-testid="stDownloadButton"] button:hover p,[data-testid="stDownloadButton"] button:hover span{color:#fff!important}
.tab-title{font-family:Georgia,serif;font-size:2.15rem;line-height:1.2;font-weight:700;letter-spacing:.01em;white-space:nowrap}
.tab-subtitle{font-family:Georgia,serif;font-size:1.08rem;font-weight:600;margin-top:.38rem}
div[data-testid="stHorizontalBlock"]:has(.tab-title){background:linear-gradient(90deg,#163f68 0%,#245b89 100%);color:#fff;margin:0 -1rem 1rem;padding:1.45rem 2.25rem 1.05rem;border-bottom:1px solid #173b5f;align-items:center}
div[data-testid="stHorizontalBlock"]:has(.tab-title) [data-testid="stButton"],div[data-testid="stHorizontalBlock"]:has(.tab-title) [data-testid="stDownloadButton"]{display:flex;justify-content:flex-end;align-items:center;height:100%}
div[data-testid="stHorizontalBlock"]:has(.tab-title) [data-testid="stButton"] button,div[data-testid="stHorizontalBlock"]:has(.tab-title) [data-testid="stDownloadButton"] button{min-width:0;width:auto!important;border-radius:5px!important;border:1px solid #fff!important;background:#fff!important;color:#17365d!important;font-weight:700!important;box-shadow:0 1px 2px rgba(0,0,0,.12)!important;padding:.45rem .8rem!important;white-space:nowrap}
div[data-testid="stHorizontalBlock"]:has(.tab-title) [data-testid="stButton"] button:hover,div[data-testid="stHorizontalBlock"]:has(.tab-title) [data-testid="stDownloadButton"] button:hover{background:#eef3f7!important;border-color:#fff!important;color:#17365d!important}
div[data-testid="stHorizontalBlock"]:has(.tab-title) [data-testid="stDownloadButton"] button p,div[data-testid="stHorizontalBlock"]:has(.tab-title) [data-testid="stDownloadButton"] button span{color:#17365d!important}
section[data-testid="stSidebar"],section[data-testid="stSidebar"]>div{background:#f3f6f9!important}
section[data-testid="stSidebar"]{border-right:1px solid #d7e0e8}
section[data-testid="stSidebar"]{min-width:292px!important;max-width:292px!important}
section[data-testid="stSidebar"]>div{padding:1.05rem .75rem .9rem!important}
section[data-testid="stSidebar"] [data-testid="stExpander"]{border:1px solid #d7e0e8!important;border-radius:4px!important;background:#eef3f7!important;margin:.35rem 0!important}
section[data-testid="stSidebar"] [data-testid="stExpander"] details summary{font-family:Georgia,serif!important;color:#17365d!important;font-weight:700!important}
section[data-testid="stSidebar"] .stRadio>div{gap:.1rem!important}
section[data-testid="stSidebar"] .stRadio label{font-size:.86rem!important}
section[data-testid="stSidebar"] .stButton>button{width:100%;border-radius:4px;border:1px solid #d0dce7}

section[data-testid="stSidebar"] .stMarkdown,section[data-testid="stSidebar"] label,section[data-testid="stSidebar"] [data-testid="stWidgetLabel"],section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"]{color:#17365d!important}
section[data-testid="stSidebar"] input{background:#fff!important;color:#17365d!important}
/* Tighten LaTeX parameter labels and the input boxes directly below them. */
section[data-testid="stSidebar"] [data-testid="stLatex"]{
    margin-bottom:-1.8rem!important;
}
section[data-testid="stSidebar"] div.element-container:has([data-testid="stLatex"]) + div.element-container:has([data-testid="stNumberInput"]){
    margin-top:-0.8rem!important;
}
.sidebar-section{font-family:Georgia,serif;color:#17365d;font-size:1.02rem;font-weight:700;padding:.15rem 0 .65rem;border-bottom:1px solid #d6dee6;margin-bottom:.65rem;letter-spacing:.01em}
.calc-card{background:#e9f2fa;border:1px solid #c8d8e7;border-radius:7px;padding:.65rem .75rem;margin-top:.55rem}
.kpi-row{background:#fff;border-bottom:1px solid #dce4eb;margin-bottom:1rem}
.kpi-label{font-size:.77rem;color:#65727f;margin-bottom:.15rem}
.kpi-value{font-family:Georgia,serif;font-size:1.08rem;font-weight:700;color:#17365d}
.kpi-status{font-size:.82rem;color:#334155;line-height:1.45}
.status-dot{display:inline-block;width:10px;height:10px;border-radius:50%;background:#159447;margin-right:6px}
.chart-card{background:#fff;border:1px solid #d6e0e8;border-radius:7px;padding:.55rem .7rem .25rem;margin-bottom:.8rem;box-shadow:0 1px 2px rgba(15,23,42,.025)}
.chart-title{font-family:Georgia,serif;color:#17365d;font-size:1.08rem;font-weight:700;line-height:1.15}
.chart-subtitle{color:#607080;font-size:.76rem;margin-top:.12rem}
.footer-note{color:#718096;font-size:.72rem;text-align:right;margin-top:.25rem}
[data-testid="stMetric"]{background:transparent;border:0;padding:.1rem .2rem}
[data-testid="stMetricLabel"]{color:#65727f!important;font-size:.77rem!important}
[data-testid="stMetricValue"]{color:#17365d!important;font-family:Georgia,serif;font-size:1.12rem!important}
/* Keep the graph controls compact and visually integrated with each chart header. */
.graph-controls{display:flex;justify-content:flex-end;align-items:center;gap:.45rem;flex-wrap:wrap;margin-top:-.15rem;margin-bottom:.1rem}
</style>
""", unsafe_allow_html=True)

LABELS = {
    "P":"Preço agregado (P)", "P1":"Preço do bem 1 (P₁)", "P2":"Preço do bem 2 (P₂)",
    "p":"Inflação (p)", "p1":"Inflação do bem 1 (p₁)", "p2":"Inflação do bem 2 (p₂)",
    "W":"Salário nominal (W)", "e":"Câmbio (E)", "w":"Crescimento do salário nominal (w)",
    "e_growth":"Crescimento do câmbio (e)", "b_hat":"Crescimento do salário real (b̂)", "pm1_growth":"Inflação do preço monitorado 1 (pᵐ₁)", "pm2_growth":"Inflação do preço monitorado 2 (pᵐ₂)", "w_share":"Parcela do salário (w_share)",
    "b":"Salário real (b)", "b1":"Salário real em trigo (b₁)", "b2":"Salário real em ferro (b₂)",
    "WE":"W/E", "n":"n agregado", "n1":"n₁", "n2":"n₂",
    "n1d":"n₁ᵈ", "n1m":"n₁ᵐ", "n1X":"n₁ˣ", "n2d":"n₂ᵈ", "n2m":"n₂ᵐ", "n2X":"n₂ˣ",
    "m":"m agregado", "m1":"m₁", "m2":"m₂", "m1d":"m₁ᵈ", "m1m":"m₁ᵐ", "m1X":"m₁ˣ", "m2d":"m₂ᵈ", "m2m":"m₂ᵐ", "m2X":"m₂ˣ",
    "X1":"Produção X₁", "X2":"Produção X₂", "l1":"Coeficiente l₁", "l2":"Coeficiente l₂",
    "a11":"a₁₁", "a12":"a₁₂", "a21":"a₂₁", "a22":"a₂₂", "x1":"Peso x₁", "x2":"Peso x₂",
    "P1d":"P₁ᵈ", "P1m":"P₁ᵐ", "P1X":"P₁ˣ", "P2d":"P₂ᵈ", "P2m":"P₂ᵐ", "P2X":"P₂ˣ",
    "d11":"d₁₁", "d12":"d₁₂", "d21":"d₂₁", "d22":"d₂₂", "T1":"T₁", "T2":"T₂",
    "rho_A":"ρ(A)", "R":"Taxa máxima de lucro (R)", "B":"B",
    "P_rel":"P/P₁", "P2_rel":"P₂/P₁", "P1d_rel":"P₁ᵈ/P₁", "P1m_rel":"P₁ᵐ/P₁", "P1X_rel":"EP₁ˣ/P₁",
    "P2d_rel":"P₂ᵈ/P₁", "P2m_rel":"P₂ᵐ/P₁", "P2X_rel":"EP₂ˣ/P₁",
}

GRAPH_GROUPS = {
    "Inflação": ["w", "e_growth", "p", "p1", "p2", "b_hat"],
    "Salário real, W/E e parcela do salário": ["b", "b1", "b2", "WE", "w_share"],
    "n — margens nominais": ["n", "n1", "n2", "n1d", "n1m", "n1X", "n2d", "n2m", "n2X"],
    "m — margens reais": ["m", "m1", "m2", "m1d", "m1m", "m1X", "m2d", "m2m", "m2X"],
    "Preços em unidades do bem 1": ["P_rel", "P2_rel", "P1d_rel", "P1m_rel", "P1X_rel", "P2d_rel", "P2m_rel", "P2X_rel"],
}

PARAMETERS = {
    "a11":"a₁₁", "a12":"a₁₂", "a21":"a₂₁", "a22":"a₂₂",
    "l1":"l₁", "l2":"l₂", "X1":"X₁", "X2":"X₂",
    "W":"W", "e":"E", "x1":"x₁",
    "phi1d":"φ₁ᵈ", "phi1m":"φ₁ᵐ", "phi2d":"φ₂ᵈ", "phi2m":"φ₂ᵐ",
    "P1X":"P₁ˣ", "P2X":"P₂ˣ",
    "n1d":"n₁ᵈ", "n1m":"n₁ᵐ", "n2d":"n₂ᵈ", "n2m":"n₂ᵐ",
    "d11":"d₁₁", "d12":"d₁₂", "d21":"d₂₁", "d22":"d₂₂", "T1":"T₁", "T2":"T₂",
}


def num(label, value, key, min_value=None, step=.01, max_value=None):
    kw = dict(label=label, value=float(value), step=float(step), format="%.4g", key=key)
    if min_value is not None:
        kw["min_value"] = float(min_value)
    if max_value is not None:
        kw["max_value"] = float(max_value)
    return st.number_input(**kw)




REPORT_EQUATIONS = [
    r"A_t=\begin{pmatrix}a_{11,t}&a_{12,t}\\a_{21,t}&a_{22,t}\end{pmatrix}",
    r"l_t=\begin{pmatrix}l_{1,t}\\l_{2,t}\end{pmatrix}",
    r"X_t=\begin{pmatrix}X_{1,t}\\X_{2,t}\end{pmatrix}",
    r"B_t=\frac{1}{l_t[I-A_t]^{-1}X_t}",
    r"\rho(A_t)=\max_i|\lambda_i(A_t)|",
    r"R_t=\frac{1}{\rho(A_t)}-1",
    r"P_{1,t}=(1+T_{1,t})[W_tl_{1,t}+C_{1,t-1}(1+n_{1,t})]",
    r"P_{2,t}=(1+T_{2,t})[W_tl_{2,t}+C_{2,t-1}(1+n_{2,t})]",
    r"C_{1,t}=d_{11,t}a_{11,t}P_{1,t}+(1-d_{11,t})a_{11,t}E_tP^X_{1,t}+d_{21,t}a_{21,t}P_{2,t}+(1-d_{21,t})a_{21,t}E_tP^X_{2,t}",
    r"C_{2,t}=d_{12,t}a_{12,t}P_{1,t}+(1-d_{12,t})a_{12,t}E_tP^X_{1,t}+d_{22,t}a_{22,t}P_{2,t}+(1-d_{22,t})a_{22,t}E_tP^X_{2,t}",
    r"1+n_{1,t}^{d}=\frac{P^d_{1,t}/(1+T_{1,t})-W_tl_{1,t}}{C_{1,t-1}}",
    r"1+n_{1,t}^{m}=\frac{P^m_{1,t}/(1+T_{1,t})-W_tl_{1,t}}{C_{1,t-1}}",
    r"1+n_{1,t}^{X}=\frac{E_tP^X_{1,t}/(1+T_{1,t})-W_tl_{1,t}}{C_{1,t-1}}",
    r"1+n_{2,t}^{d}=\frac{P^d_{2,t}/(1+T_{2,t})-W_tl_{2,t}}{C_{2,t-1}}",
    r"1+n_{2,t}^{m}=\frac{P^m_{2,t}/(1+T_{2,t})-W_tl_{2,t}}{C_{2,t-1}}",
    r"1+n_{2,t}^{X}=\frac{E_tP^X_{2,t}/(1+T_{2,t})-W_tl_{2,t}}{C_{2,t-1}}",
    r"P^d_{1,t}=(1+T_{1,t})[W_tl_{1,t}+C_{1,t-1}(1+n^d_{1,t})]",
    r"P^m_{1,t}=(1+T_{1,t})[W_tl_{1,t}+C_{1,t-1}(1+n^m_{1,t})]",
    r"E_tP^X_{1,t}=(1+T_{1,t})[W_tl_{1,t}+C_{1,t-1}(1+n^X_{1,t})]",
    r"P^d_{2,t}=(1+T_{2,t})[W_tl_{2,t}+C_{2,t-1}(1+n^d_{2,t})]",
    r"P^m_{2,t}=(1+T_{2,t})[W_tl_{2,t}+C_{2,t-1}(1+n^m_{2,t})]",
    r"E_tP^X_{2,t}=(1+T_{2,t})[W_tl_{2,t}+C_{2,t-1}(1+n^X_{2,t})]",
    r"1+n_{1,t}=\phi^d_{1,t}(1+n^d_{1,t})+\phi^m_{1,t}(1+n^m_{1,t})+\phi^X_{1,t}(1+n^X_{1,t})",
    r"1+n_{2,t}=\phi^d_{2,t}(1+n^d_{2,t})+\phi^m_{2,t}(1+n^m_{2,t})+\phi^X_{2,t}(1+n^X_{2,t})",
    r"P_{1,t}=\phi^d_{1,t}P^d_{1,t}+\phi^m_{1,t}P^m_{1,t}+\phi^X_{1,t}E_tP^X_{1,t}",
    r"P_{2,t}=\phi^d_{2,t}P^d_{2,t}+\phi^m_{2,t}P^m_{2,t}+\phi^X_{2,t}E_tP^X_{2,t}",
    r"P_t=x_{1,t}P_{1,t}+x_{2,t}P_{2,t},\qquad x_{1,t}+x_{2,t}=1",
    r"1+n_t=x_{1,t}(1+n_{1,t})+x_{2,t}(1+n_{2,t})",
    r"b_{1,t}=\frac{W_t}{P_{1,t}},\qquad b_{2,t}=\frac{W_t}{P_{2,t}},\qquad b_t=\frac{W_t}{P_t}",
    r"\hat w_t=\frac{W_t}{W_{t-1}}-1",
    r"e_t=\frac{E_t}{E_{t-1}}-1",
    r"\hat p_{1,t}=\frac{P_{1,t}}{P_{1,t-1}}-1,\qquad \hat p_{2,t}=\frac{P_{2,t}}{P_{2,t-1}}-1,\qquad \hat p_t=\frac{P_t}{P_{t-1}}-1",
    r"1+m^d_{1,t}=\frac{P^d_{1,t}/(1+T_{1,t})-W_tl_{1,t}}{C_{1,t}}",
    r"1+m^m_{1,t}=\frac{P^m_{1,t}/(1+T_{1,t})-W_tl_{1,t}}{C_{1,t}}",
    r"1+m^X_{1,t}=\frac{E_tP^X_{1,t}/(1+T_{1,t})-W_tl_{1,t}}{C_{1,t}}",
    r"1+m^d_{2,t}=\frac{P^d_{2,t}/(1+T_{2,t})-W_tl_{2,t}}{C_{2,t}}",
    r"1+m^m_{2,t}=\frac{P^m_{2,t}/(1+T_{2,t})-W_tl_{2,t}}{C_{2,t}}",
    r"1+m^X_{2,t}=\frac{E_tP^X_{2,t}/(1+T_{2,t})-W_tl_{2,t}}{C_{2,t}}",
    r"1+m_{1,t}=\phi^d_{1,t}(1+m^d_{1,t})+\phi^m_{1,t}(1+m^m_{1,t})+\phi^X_{1,t}(1+m^X_{1,t})",
    r"1+m_{2,t}=\phi^d_{2,t}(1+m^d_{2,t})+\phi^m_{2,t}(1+m^m_{2,t})+\phi^X_{2,t}(1+m^X_{2,t})",
    r"1+m_t=x_{1,t}(1+m_{1,t})+x_{2,t}(1+m_{2,t})",
    r"\hat w_t=k_W+\alpha_W\hat p_{t-1},\qquad \alpha_W=\frac{F_W}{F_P+F_W}",
    r"e_t=k_E+\alpha_E\hat p_{t-1}",
    r"\hat p^m_{1,t}=k_{m1}+\alpha_{m1}\hat p_{t-1}",
    r"\hat p^m_{2,t}=k_{m2}+\alpha_{m2}\hat p_{t-1}",
    r"P^m_{1,t}=P^m_{1,t-1}(1+\hat p^m_{1,t-1})",
    r"P^m_{2,t}=P^m_{2,t-1}(1+\hat p^m_{2,t-1})",
    r"w_{share,t}=\frac{W_tl_tX_t}{\mathbf{P}_t[I-A_t]X_t}",
]

def build_results_pdf(df, input_rows, calculated_rows, graph_groups, special=False):
    """Create a clean PDF report with initial conditions and high-resolution graphs."""
    buffer = io.BytesIO()
    page_w, page_h = landscape(A4)

    navy = colors.HexColor("#17365D")
    light_blue = colors.HexColor("#EEF3F7")
    grid = colors.HexColor("#D7E0E8")
    text = colors.HexColor("#263746")
    muted = colors.HexColor("#667786")
    white = colors.white

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=23,
        leading=27, textColor=navy, alignment=TA_LEFT, spaceAfter=5,
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle", parent=styles["Normal"], fontName="Helvetica", fontSize=10.5,
        leading=14, textColor=muted, spaceAfter=10,
    )
    section_style = ParagraphStyle(
        "Section", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=14,
        leading=17, textColor=navy, spaceBefore=5, spaceAfter=7,
    )
    small_style = ParagraphStyle(
        "Small", parent=styles["Normal"], fontName="Helvetica", fontSize=8.5,
        leading=11, textColor=text,
    )
    value_style = ParagraphStyle(
        "Value", parent=small_style, fontName="Helvetica-Bold", fontSize=9,
        leading=12, textColor=navy,
    )
    note_style = ParagraphStyle(
        "Note", parent=styles["Normal"], fontName="Helvetica", fontSize=8,
        leading=10.5, textColor=muted,
    )

    def fmt_pdf(v):
        if v is None:
            return "—"
        try:
            x = float(v)
            if np.isnan(x):
                return "—"
            if np.isinf(x):
                return "∞" if x > 0 else "−∞"
            if abs(x) >= 1000 or (0 < abs(x) < 1e-5):
                return f"{x:.5e}"
            return f"{x:.8f}".rstrip("0").rstrip(".")
        except (TypeError, ValueError):
            return str(v)

    def add_page_number(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(grid)
        canvas.line(15*mm, 11*mm, page_w-15*mm, 11*mm)
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(muted)
        canvas.drawString(15*mm, 6.5*mm, model_name + " — Relatório de resultados")
        canvas.drawRightString(page_w-15*mm, 6.5*mm, f"Página {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(
        buffer, pagesize=landscape(A4), rightMargin=15*mm, leftMargin=15*mm,
        topMargin=14*mm, bottomMargin=16*mm,
        title=f"{model_name} — Resultados da simulação",
        author="Trigo-Ferro / Trigo",
    )
    story = []
    now = datetime.now().strftime("%d/%m/%Y às %H:%M")

    story.append(Paragraph(model_name, title_style))
    story.append(Paragraph(
        f"Relatório de resultados · condições iniciais e trajetórias simuladas · gerado em {now}",
        subtitle_style,
    ))
    story.append(Spacer(1, 3*mm))

    # Initial conditions are presented as compact parameter cards rather than
    # a numerical table. This keeps the report visual and avoids the previous
    # cramped glyph rendering in the table.
    story.append(Paragraph("Condições iniciais", section_style))
    story.append(Paragraph(
        "Valores dos parâmetros e configurações utilizados no momento da geração deste relatório.",
        note_style,
    ))
    story.append(Spacer(1, 3*mm))

    def pdf_label(label):
        """Convert Unicode subscripts/symbols to ReportLab-safe markup."""
        label = str(label)
        sub_map = {
            "₀": "<sub>0</sub>", "₁": "<sub>1</sub>", "₂": "<sub>2</sub>",
            "₃": "<sub>3</sub>", "₄": "<sub>4</sub>", "₅": "<sub>5</sub>",
            "₆": "<sub>6</sub>", "₇": "<sub>7</sub>", "₈": "<sub>8</sub>",
            "₉": "<sub>9</sub>",
        }
        super_map = {"ˣ": "<super>x</super>"}
        for src, dst in sub_map.items():
            label = label.replace(src, dst)
        for src, dst in super_map.items():
            label = label.replace(src, dst)
        label = label.replace("φ", "&phi;").replace("α", "&alpha;").replace("ρ", "&rho;")
        return label

    cards = []
    for label, value in input_rows:
        cards.append(
            Paragraph(
                f"<font color='#667786'>{pdf_label(label)}</font><br/><font size='10'><b>{fmt_pdf(value)}</b></font>",
                small_style,
            )
        )

    # Five compact columns give substantially more room than the old two-column table.
    ncols = 5
    card_width = (page_w - 30*mm - 4*4*mm) / ncols
    card_rows = [cards[i:i+ncols] for i in range(0, len(cards), ncols)]
    if card_rows:
        while len(card_rows[-1]) < ncols:
            card_rows[-1].append(Paragraph("", small_style))
        cards_table = Table(card_rows, colWidths=[card_width]*ncols, hAlign="LEFT")
        cards_table.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,-1), light_blue),
            ("BOX", (0,0), (-1,-1), 0.45, grid),
            ("INNERGRID", (0,0), (-1,-1), 0.35, grid),
            ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
            ("LEFTPADDING", (0,0), (-1,-1), 6),
            ("RIGHTPADDING", (0,0), (-1,-1), 6),
            ("TOPPADDING", (0,0), (-1,-1), 5),
            ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ]))
        story.append(cards_table)

    story.append(Spacer(1, 7*mm))
    story.append(Paragraph("Trajetórias simuladas", section_style))
    story.append(Paragraph(
        "Cada série é apresentada em um gráfico individual, evitando que diferenças de escala ocultem trajetórias menores. A exportação é feita em alta resolução para preservar a legibilidade em impressão e ampliação.",
        note_style,
    ))
    story.append(PageBreak())

    def graph_png(series_title, variable):
        # Each series gets its own graph so differences in scale cannot hide
        # the trajectory of smaller-magnitude variables. 300 dpi gives
        # print-quality raster graphics while keeping the PDF size reasonable.
        fig, ax = plt.subplots(figsize=(11.0, 5.4), dpi=300)
        ax.set_facecolor("#FFFFFF")
        fig.patch.set_facecolor("#FFFFFF")
        if variable in df.columns:
            y = pd.to_numeric(df[variable], errors="coerce").round(12)
            ax.plot(
                df["period"], y, linewidth=2.0,
                label=LABELS.get(variable, variable),
                antialiased=True,
            )
        ax.set_title(series_title, loc="left", fontsize=16, fontweight="bold", color="#17365D", pad=12)
        ax.set_xlabel("Período", color="#536675", fontsize=10)
        ax.set_ylabel("Valor", color="#536675", fontsize=10)
        ax.grid(True, axis="y", linewidth=0.5, alpha=0.28)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=9, colors="#536675")
        ax.legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.12),
            ncol=1, frameon=False, fontsize=9,
        )
        fig.tight_layout(rect=[0, 0.05, 1, 1])
        out = io.BytesIO()
        fig.savefig(out, format="png", dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        out.seek(0)
        return out

    graph_index = 0
    # One figure per series: never combine variables on the same axes.
    graph_items = [(group_title, variable) for group_title, variables in graph_groups for variable in variables if variable in df.columns]
    for group_title, variable in graph_items:
            label = LABELS.get(variable, variable)
            # The graph already contains its own title. Do not add a second
            # ReportLab title above it: that duplicate title also causes
            # Unicode/subscript rendering problems for some series labels.
            story.append(Image(graph_png(f"{group_title} — {label}", variable), width=252*mm, height=118*mm))
            graph_index += 1
            if graph_index < len(graph_items):
                story.append(PageBreak())

    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
    buffer.seek(0)
    return buffer.getvalue()

def fmt_num(value, decimals=4):
    """Presentation only: up to `decimals` decimals, without trailing zeros."""
    try:
        x = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(x):
        return str(x)
    return f"{x:.{decimals}f}".rstrip("0").rstrip(".")


def constant_path(v, T):
    return np.full(T + 1, float(v), dtype=float)


def apply_one_shock(arr, period, pct, permanent):
    factor = 1 + pct / 100.0
    if permanent:
        arr[period:] *= factor
    else:
        arr[period] *= factor
    return arr


def live_technology(A0, l0, X0):
    rho = float(np.max(np.abs(np.linalg.eigvals(A0))))
    try:
        B = float(1.0 / (l0 @ np.linalg.solve(np.eye(2) - A0, X0)))
        B_err = None
    except np.linalg.LinAlgError:
        B, B_err = np.nan, "I−A é singular; B não pode ser calculado."
    R = np.inf if abs(rho) < 1e-12 else 1 / rho - 1
    return rho, B, R, B_err


def wage_profit_curve(A, l, m1_ref, m2_ref, x1, x2, npoints=500):
    """Analytical b-m curve using the initial sectoral real margins as a path.

    Trigo (P1) is the numeraire: P1 = 1. The two sectoral margins are
    varied proportionally from zero while preserving their initial ratio:

        m1 = lambda * m1_ref
        m2 = lambda * m2_ref

    For each lambda, solve the initial two-good price system

        P = W*l + A.T @ diag(1+m1, 1+m2) @ P

    with P1=1. The horizontal coordinate is the x-weighted average margin

        m = x1*m1 + x2*m2,

    and the vertical coordinate is b = W/P1 = W.
    """
    A = np.asarray(A, dtype=float)
    l = np.asarray(l, dtype=float)
    m1_ref = float(m1_ref)
    m2_ref = float(m2_ref)
    x1 = float(x1)
    x2 = float(x2)

    # Search for the largest admissible proportional margin before the
    # price system becomes singular or b ceases to be economically usable.
    lam_grid = np.linspace(0.0, 10.0, max(npoints * 4, 1000))
    ms = np.full_like(lam_grid, np.nan, dtype=float)
    bs = np.full_like(lam_grid, np.nan, dtype=float)

    last_valid = 0
    for k, lam in enumerate(lam_grid):
        m1 = lam * m1_ref
        m2 = lam * m2_ref
        # With P1=1, solve directly for W and P2.
        M = np.array([
            [l[0], (1.0 + m2) * A[1, 0]],
            [l[1], -(1.0 - (1.0 + m2) * A[1, 1])],
        ], dtype=float)
        rhs = np.array([
            1.0 - (1.0 + m1) * A[0, 0],
            -(1.0 + m1) * A[0, 1],
        ], dtype=float)
        try:
            W, P2 = np.linalg.solve(M, rhs)
        except np.linalg.LinAlgError:
            break
        mbar = x1 * m1 + x2 * m2
        if not (np.isfinite(W) and np.isfinite(P2) and W >= 0 and P2 > 0):
            break
        ms[k] = mbar
        bs[k] = W
        last_valid = k

    # Keep the valid branch and resample smoothly.
    if last_valid < 2:
        return np.array([]), np.array([])
    valid_m = ms[:last_valid + 1]
    valid_b = bs[:last_valid + 1]
    # Avoid duplicate x values when the reference margins imply zero average.
    keep = np.r_[True, np.abs(np.diff(valid_m)) > 1e-12]
    return valid_m[keep], valid_b[keep]


def plot_series(df, variables, title=None, height=250):
    fig = go.Figure()
    for v in variables:
        if v not in df.columns:
            continue
        y = pd.to_numeric(df[v], errors="coerce")
        if y.notna().sum() == 0:
            continue
        # Suppress floating-point noise in the plotted series.  The simulation
        # can produce values such as 0.0299999999999998 and 0.0300000000000005
        # for an economically constant rate. Plotly otherwise autos-scales to
        # those machine-precision differences and creates a false zig-zag.
        y = y.round(12)
        display_name = str(LABELS.get(v, v))
        fig.add_trace(
            go.Scatter(
                x=df["period"],
                y=y,
                mode="lines",
                name=display_name,
                hovertemplate=f"Período: %{{x}}<br>{display_name}: %{{y:.6g}}<extra></extra>",
                line=dict(width=1.7),
            )
        )
    # Do not create a Plotly title object here. The visible chart title is the
    # HTML heading above the figure; a null title can be rendered as “undefined”
    # by some Plotly/Streamlit combinations.
    fig.update_layout(
        template="plotly_white",
        height=height,
        hovermode="x unified",
        margin=dict(l=48, r=18, t=8, b=38),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        xaxis_title="Período",
        font=dict(size=11, color="#17365d"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
    )
    return fig

with st.sidebar:
    st.markdown('<div class="sidebar-section">⚙ CONTROLES DA SIMULAÇÃO</div>', unsafe_allow_html=True)
    T = int(st.number_input("Períodos (T)", 1, 500, 90, 1, key="T"))

    model_choice = st.radio(
        "Modelo",
        ["Trigo", "Trigo-Ferro"],
        index=1,
        key="model_choice",
        label_visibility="visible",
    )
    trigo = model_choice == "Trigo"
    trigo_ferro = model_choice == "Trigo-Ferro"
    special = trigo

    # In the one-good Trigo special case, sector 2 does not exist.
    # Keep all simulation variables intact, but do not expose nonexistent
    # sector-2 series in the dashboard graphs.
    if special:
        GRAPH_GROUPS_ACTIVE = {
            "Inflação": ["w", "e_growth", "p", "p1", "b_hat"],
            "Salário real, W/E e parcela do salário": ["b", "b1", "WE", "w_share"],
            "n — margens nominais": ["n", "n1", "n1d", "n1m", "n1X"],
            "m — margens reais": ["m", "m1", "m1d", "m1m", "m1X"],
            "Preços em unidades do bem 1": ["P_rel", "P1d_rel", "P1m_rel", "P1X_rel"],
        }
    else:
        GRAPH_GROUPS_ACTIVE = GRAPH_GROUPS

    # ------------------------------------------------------------------
    # INPUTS — organized as a professional control panel. The expanders
    # preserve all existing controls and calculations while keeping the
    # sidebar visually compact.
    # ------------------------------------------------------------------
    with st.expander("Tecnologia", expanded=True):
        st.markdown("**Matriz A**")
        r1c1, r1c2 = st.columns(2)
        r2c1, r2c2 = st.columns(2)
        with r1c1:
            a11 = num("a₁₁", .30, "a11", 0)
        with r1c2:
            a12 = 0.0 if special else num("a₁₂", .20, "a12", 0)
        with r2c1:
            a21 = 0.0 if special else num("a₂₁", .40, "a21", 0)
        with r2c2:
            a22 = 0.0 if special else num("a₂₂", .10, "a22", 0)
        st.latex(r"A=\begin{pmatrix}a_{11}&a_{12}\\a_{21}&a_{22}\end{pmatrix}")

        l1 = num("l₁", .125, "l1", 0)
        l2 = 0.0 if special else num("l₂", .20, "l2", 0)
        X1 = num("X₁", 1.0, "X1", 0)
        X2 = 1.0 if special else num("X₂", 1.0, "X2", 0)

        st.markdown("**Coeficiente de conteúdo doméstico**")
        dc1, dc2 = st.columns(2)
        dc3, dc4 = st.columns(2)
        with dc1:
            d11 = num("d₁₁", 0.6, "d11", 0.0, .01, 1.0)
        with dc2:
            d12 = 1.0 if special else num("d₁₂", 0.3, "d12", 0.0, .01, 1.0)
        with dc3:
            d21 = 1.0 if special else num("d₂₁", 0.8, "d21", 0.0, .01, 1.0)
        with dc4:
            d22 = 1.0 if special else num("d₂₂", 0.5, "d22", 0.0, .01, 1.0)

        st.markdown("**Alíquotas de imposto indireto**")
        tc1, tc2 = st.columns(2)
        with tc1:
            T1 = num("T₁", 0.1, "T1", -.99, .01)
        with tc2:
            T2 = 0.0 if special else num("T₂", 0.15, "T2", -.99, .01)

        apply_l_growth = st.checkbox("Aplicar crescimento aos coeficientes de trabalho (l)", False, key="apply_l_growth")
        if apply_l_growth:
            g_l1 = num("Taxa de decrescimento de l₁", 0.0, "g_l1", min_value=-1.0, max_value=.99, step=.005)
            if special:
                g_l2 = 0.0
            else:
                g_l2 = num("Taxa de decrescimento de l₂", 0.0, "g_l2", min_value=-1.0, max_value=.99, step=.005)
            st.latex(r"l_{1,t}=\frac{l_{1,t-1}}{1+g_{l1}}")
            if not special:
                st.latex(r"l_{2,t}=\frac{l_{2,t-1}}{1+g_{l2}}")
            st.caption("g_li > 0 reduz o respectivo coeficiente l ao longo do tempo; a taxa representa crescimento da produtividade do trabalho.")
        else:
            g_l1 = g_l2 = 0.0

        A0 = np.array([[a11,a12],[a21,a22]], dtype=float)
        l0 = np.array([l1,l2], dtype=float)
        X0 = np.array([X1,X2], dtype=float)
        rho0, B0, R0, B_err = live_technology(A0,l0,X0)

        st.markdown('<div class="calc-card">', unsafe_allow_html=True)
        st.markdown("**Resultados calculados**")
        c1, c2, c3 = st.columns(3)
        c1.metric("ρ(A)", fmt_num(rho0))
        c2.metric("R", "∞" if np.isinf(R0) else fmt_num(R0))
        c3.metric("B", "—" if not np.isfinite(B0) else fmt_num(B0))
        if B_err:
            st.caption(B_err)
        st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("Ponderadores", expanded=False):
        if special:
            x1 = 1.0
            x2 = 0.0
            st.number_input("x₁", value=1.0, min_value=0.0, max_value=1.0, step=0.01, format="%.4g", disabled=True, key="x1_trigo_display")
            st.caption("No caso Trigo: x₁=1 e x₂=0.")
        else:
            x1 = num("x₁", .50, "x1", 0, .01)
            x1 = min(max(x1,0),1)
            x2 = 1-x1
        phi1d = num("φ₁d", .60, "phi1d", 0, .01)
        phi1m = num("φ₁m", .20, "phi1m", 0, .01)
        phi1x = 1-phi1d-phi1m
        if special:
            phi2d,phi2m,phi2x = 1.,0.,0.
        else:
            phi2d = num("φ₂d", .60, "phi2d", 0, .01)
            phi2m = num("φ₂m", .20, "phi2m", 0, .01)
            phi2x = 1-phi2d-phi2m
        st.caption(f"x₂={fmt_num(x2)} | φ₁x={fmt_num(phi1x)} | φ₂x={fmt_num(phi2x)}")
        if phi1x < 0 or phi2x < 0:
            st.error("Os pesos φᵈ+φᵐ não podem exceder 1.")

    with st.expander("Condições iniciais", expanded=False):
        W0 = num("W inicial",1.0,"W0",1e-9)
        E0 = num("E inicial",1.0,"E0",1e-9)
        P1d0 = num("P₁d inicial",1.0,"P1d0",1e-9)
        P1m0 = num("P₁m inicial",1.0,"P1m0",1e-9)
        P2d0 = 1.0 if special else num("P₂d inicial",1.0,"P2d0",1e-9)
        P2m0 = 1.0 if special else num("P₂m inicial",1.0,"P2m0",1e-9)
        P1X0 = num("P₁ˣ",1.0,"P1X0",1e-9)
        P2X0 = 1.0 if special else num("P₂ˣ",1.0,"P2X0",1e-9)

        st.markdown("**Taxas de crescimento iniciais**")
        gc1, gc2, gc3 = st.columns(3)
        with gc1:
            initial_inflation = num("Inflação inicial (p₀)",0.03,"initial_inflation",-0.99,.005)
            initial_w_growth = num("Crescimento inicial de W (w₀)",0.03,"initial_w_growth",-0.99,.005)
        with gc2:
            initial_e_growth = num("Crescimento inicial de E (e₀)",0.03,"initial_e_growth",-0.99,.005)
            initial_pm1_growth = num("Crescimento inicial de Pᵐ₁ (pᵐ₁,₀)",0.03,"initial_pm1_growth",-0.99,.005)
        with gc3:
            initial_pm2_growth = (0.0 if special else
                                  num("Crescimento inicial de Pᵐ₂ (pᵐ₂,₀)",0.03,"initial_pm2_growth",-0.99,.005))
        st.caption("Taxas herdadas do período anterior. Elas determinam a primeira transição (de t=0 para t=1) e não precisam ser iguais aos k's. A partir de t=1, quando a indexação está ativa, entram as equações de indexação definidas abaixo.")

        # Initial sectoral prices are implied by the market-specific prices,
        # the exchange rate and the φ shares; they are not independent inputs.
        P1_live = phi1d*P1d0 + phi1m*P1m0 + phi1x*E0*P1X0
        P2_live = phi2d*P2d0 + phi2m*P2m0 + phi2x*E0*P2X0
        P0_live = x1*P1_live + x2*P2_live
        b0_live = W0/P0_live if abs(P0_live)>1e-12 else np.nan
        WE0_live = W0/E0 if abs(E0)>1e-12 else np.nan
        c10_live = d11*a11*P1_live + (1-d11)*a11*E0*P1X0 + d21*a21*P2_live + (1-d21)*a21*E0*P2X0
        c20_live = d12*a12*P1_live + (1-d12)*a12*E0*P1X0 + d22*a22*P2_live + (1-d22)*a22*E0*P2X0
        m1d0_live = (P1d0/(1+T1)-W0*l1)/c10_live-1 if abs(c10_live)>1e-12 else np.nan
        m1m0_live = (P1m0/(1+T1)-W0*l1)/c10_live-1 if abs(c10_live)>1e-12 else np.nan
        m1X0_live = (E0*P1X0/(1+T1)-W0*l1)/c10_live-1 if abs(c10_live)>1e-12 else np.nan
        m2d0_live = (P2d0/(1+T2)-W0*l2)/c20_live-1 if (not special and abs(c20_live)>1e-12) else np.nan
        m2m0_live = (P2m0/(1+T2)-W0*l2)/c20_live-1 if (not special and abs(c20_live)>1e-12) else np.nan
        m2X0_live = (E0*P2X0/(1+T2)-W0*l2)/c20_live-1 if (not special and abs(c20_live)>1e-12) else np.nan

        pc1, pc2 = st.columns(2)
        with pc1:
            st.metric("P₁ implícito", fmt_num(P1_live))
        with pc2:
            st.metric("P₂ implícito", fmt_num(P2_live))
        st.caption("P₁ e P₂ são médias ponderadas dos preços de mercado: φᵈPᵈ + φᵐPᵐ + φˣEPˣ.")

        # Convert initial real margins into nominal margins using the
        # inherited inflation rate. These nominal margins are exogenous and
        # remain fixed unless an explicit margin shock is applied.
        n1d0 = (1.0 + m1d0_live) * (1.0 + initial_inflation) - 1.0
        n1m0 = (1.0 + m1m0_live) * (1.0 + initial_inflation) - 1.0
        n2d0 = ((1.0 + m2d0_live) * (1.0 + initial_inflation) - 1.0) if not special else 0.0
        n2m0 = ((1.0 + m2m0_live) * (1.0 + initial_inflation) - 1.0) if not special else 0.0

    with st.expander("Indexação", expanded=False):
        if special:
            pm_options = ["Nenhum", "Preço monitorado"]
            pm_default = 1
        else:
            pm_options = ["Nenhum", "Monitorados no bem 1", "Monitorados no bem 2", "Ambos"]
            pm_default = 3

        pm_indexation = st.selectbox(
            "Indexar preço monitorado" if special else "Indexar preços monitorados",
            pm_options,
            index=pm_default,
            key="pm_indexation",
        )
        if special:
            idx_pm1 = pm_indexation == "Preço monitorado"
            idx_pm2 = False
        else:
            idx_pm1 = pm_indexation in {"Monitorados no bem 1", "Ambos"}
            idx_pm2 = pm_indexation in {"Monitorados no bem 2", "Ambos"}
        if idx_pm1:
            st.markdown("**Monitorados no bem 1**")
            pm1c1, pm1c2 = st.columns(2)
            with pm1c1:
                st.latex(r"k_{m1}")
                km1 = num("", 0.0, "km1", step=.005)
            with pm1c2:
                st.latex(r"\alpha_{m1}")
                alpha_m1 = num("", 1.0, "alpha_m1", step=.05)
            st.latex(r"p^m_{1,t}=k_{m1}+\alpha_{m1}p_{t-1}")
            st.latex(r"P^m_{1,t}=P^m_{1,t-1}(1+p^m_{1,t})")
        else:
            km1 = alpha_m1 = 0.0

        if idx_pm2:
            st.markdown("**Monitorados no bem 2**")
            pm2c1, pm2c2 = st.columns(2)
            with pm2c1:
                st.latex(r"k_{m2}")
                km2 = num("", 0.0, "km2", step=.005)
            with pm2c2:
                st.latex(r"\alpha_{m2}")
                alpha_m2 = num("", 1.0, "alpha_m2", step=.05)
            st.latex(r"p^m_{2,t}=k_{m2}+\alpha_{m2}p_{t-1}")
            st.latex(r"P^m_{2,t}=P^m_{2,t-1}(1+p^m_{2,t})")
        else:
            km2 = alpha_m2 = 0.0

        idx_w = st.checkbox("Indexar salário nominal (W)", True, key="idx_w")
        if idx_w:
            wc1, wc2 = st.columns(2)
            with wc1:
                st.latex(r"k_W")
                kW = num("",0.03,"kW",step=.005)
            with wc2:
                fwc1, fwc2 = st.columns(2)
                with fwc1:
                    st.latex(r"F_W")
                    Fw = num("",0.0,"Fw",0.0,.25)
                with fwc2:
                    st.latex(r"F_P")
                    Fp = num("",1.0,"Fp",0.01,.25)
            st.latex(r"\alpha_W=\frac{F_W}{F_W+F_P}")
            alphaW = Fw/(Fw+Fp)
            st.latex(r"w_t=k_W+\alpha_Wp_{t-1}")
        else:
            Fw=Fp=kW=alphaW=0.0

        idx_e = st.checkbox("Indexar taxa de câmbio (E)", True, key="idx_e")
        if idx_e:
            ec1, ec2 = st.columns(2)
            with ec1:
                st.latex(r"k_E")
                kE = num("",0.0,"kE",step=.005)
            with ec2:
                st.latex(r"\alpha_E")
                alphaE = num("",1.0,"alphaE",step=.05)
            st.latex(r"e_t=k_E+\alpha_Ep_{t-1}")
        else:
            alphaE=kE=0.0

    with st.expander("Choques", expanded=False):
        use_shock = st.checkbox("Ativar choque", False, key="use_shock")
        shock_param = None
        if use_shock:
            shock_param = st.selectbox("Parâmetro a sofrer o choque", list(PARAMETERS), format_func=lambda k: PARAMETERS[k], key="shock_param")
            shock_pct = st.number_input("Tamanho do choque (%)", -99.0, 500.0, 5.0, .5, key="shock_pct")
            shock_period = int(st.number_input("Período do choque", 1, T, min(2,T), 1, key="shock_period"))
            shock_duration = st.selectbox("Duração", ["Permanente","Um período"], key="shock_duration")
            st.caption("O choque é aplicado à trajetória do parâmetro selecionado.")

    st.markdown("<div style='height:.25rem'></div>", unsafe_allow_html=True)
    st.button("Restaurar padrão", key="restore_default", use_container_width=True)


# Build dynamic paths.
A = np.zeros((T+1,2,2)); A[:,0,0]=a11; A[:,0,1]=a12; A[:,1,0]=a21; A[:,1,1]=a22
l = np.zeros((T+1,2), dtype=float)
l[0] = [l1, l2]
if apply_l_growth:
    for t in range(1, T+1):
        l[t,0] = l[t-1,0] / (1.0 + g_l1)
        l[t,1] = l[t-1,1] / (1.0 + g_l2)
else:
    l[:] = [l1, l2]
X = np.tile([X1,X2], (T+1,1))
x1_path = constant_path(x1,T); x2_path = 1-x1_path
phi = np.tile(np.array([[phi1d,phi1m,phi1x],[phi2d,phi2m,phi2x]],float),(T+1,1,1))
W = constant_path(W0,T); E = constant_path(E0,T)
P1X = constant_path(P1X0,T); P2X = constant_path(P2X0,T)
n1d = constant_path(n1d0,T); n1m = constant_path(n1m0,T); n2d = constant_path(n2d0,T); n2m = constant_path(n2m0,T)
d11_path = constant_path(d11,T); d12_path = constant_path(d12,T); d21_path = constant_path(d21,T); d22_path = constant_path(d22,T)
T1_path = constant_path(T1,T); T2_path = constant_path(T2,T)

# Apply the single generic shock to whichever parameter the user selected.
if use_shock:
    permanent = shock_duration == "Permanente"
    if shock_param in {"a11","a12","a21","a22"}:
        # Explicit mapping: the sector indices are 0/1, while the parameter
        # names use economic notation a11, a12, a21, a22.
        idx = {"a11": (0, 0), "a12": (0, 1), "a21": (1, 0), "a22": (1, 1)}
        i, j = idx[shock_param]
        A[:, i, j] = {"a11": a11, "a12": a12, "a21": a21, "a22": a22}[shock_param]
        apply_one_shock(A[:, i, j], shock_period, shock_pct, permanent)
    elif shock_param == "l1": apply_one_shock(l[:,0],shock_period,shock_pct,permanent)
    elif shock_param == "l2": apply_one_shock(l[:,1],shock_period,shock_pct,permanent)
    elif shock_param == "X1": apply_one_shock(X[:,0],shock_period,shock_pct,permanent)
    elif shock_param == "X2": apply_one_shock(X[:,1],shock_period,shock_pct,permanent)
    elif shock_param == "W":
        # If wage indexation is active, the shock is applied to the indexed
        # level at the shock date; the shocked level then becomes the base
        # for subsequent indexed periods. If indexation is off, shock the
        # exogenous W path directly.
        if not idx_w:
            apply_one_shock(W,shock_period,shock_pct,permanent)
    elif shock_param == "e":
        # Same treatment for the exchange-rate level when exchange-rate
        # indexation is active.
        if not idx_e:
            apply_one_shock(E,shock_period,shock_pct,permanent)
    elif shock_param == "x1": apply_one_shock(x1_path,shock_period,shock_pct,permanent); x1_path=np.clip(x1_path,0,1); x2_path=1-x1_path
    elif shock_param == "phi1d": apply_one_shock(phi[:,0,0],shock_period,shock_pct,permanent); phi[:,0,2]=1-phi[:,0,0]-phi[:,0,1]
    elif shock_param == "phi1m": apply_one_shock(phi[:,0,1],shock_period,shock_pct,permanent); phi[:,0,2]=1-phi[:,0,0]-phi[:,0,1]
    elif shock_param == "phi2d": apply_one_shock(phi[:,1,0],shock_period,shock_pct,permanent); phi[:,1,2]=1-phi[:,1,0]-phi[:,1,1]
    elif shock_param == "phi2m": apply_one_shock(phi[:,1,1],shock_period,shock_pct,permanent); phi[:,1,2]=1-phi[:,1,0]-phi[:,1,1]
    elif shock_param == "P1X": apply_one_shock(P1X,shock_period,shock_pct,permanent)
    elif shock_param == "P2X": apply_one_shock(P2X,shock_period,shock_pct,permanent)
    elif shock_param == "n1d": apply_one_shock(n1d,shock_period,shock_pct,permanent)
    elif shock_param == "n1m": apply_one_shock(n1m,shock_period,shock_pct,permanent)
    elif shock_param == "n2d": apply_one_shock(n2d,shock_period,shock_pct,permanent)
    elif shock_param == "n2m": apply_one_shock(n2m,shock_period,shock_pct,permanent)
    elif shock_param == "d11": apply_one_shock(d11_path,shock_period,shock_pct,permanent)
    elif shock_param == "d12": apply_one_shock(d12_path,shock_period,shock_pct,permanent)
    elif shock_param == "d21": apply_one_shock(d21_path,shock_period,shock_pct,permanent)
    elif shock_param == "d22": apply_one_shock(d22_path,shock_period,shock_pct,permanent)
    elif shock_param == "T1": apply_one_shock(T1_path,shock_period,shock_pct,permanent)
    elif shock_param == "T2": apply_one_shock(T2_path,shock_period,shock_pct,permanent)

# Validate phi after shocks.
if np.any(phi[:,:,2] < -1e-12):
    st.error("O choque escolhido tornou algum peso φˣ negativo. Reduza o tamanho do choque.")
    st.stop()

# Sequential simulation with causal indexation.
# W_t, E_t and indexed P^m_t depend only on p_{t-1}; they therefore must be
# updated period by period, not by a global fixed-point iteration.
def run_model(W_path,E_path,n1m_path,n2m_path, horizon):
    if special:
        return simulate_one_good_special_case(horizon,A[:horizon+1,0,0],l[:horizon+1,0],X[:horizon+1,0],W_path[:horizon+1],E_path[:horizon+1],phi[:horizon+1,0,:],n1d[:horizon+1],n1m_path[:horizon+1],P1X[:horizon+1],P1d0=P1d0,P1m0=P1m0,d11=d11_path[:horizon+1],T1=T1_path[:horizon+1],initial_inflation=initial_inflation)
    return simulate_two_good_model(horizon,A[:horizon+1],l[:horizon+1],X[:horizon+1],W_path[:horizon+1],E_path[:horizon+1],phi[:horizon+1],x1_path[:horizon+1],x2_path[:horizon+1],P1d0=P1d0,P1m0=P1m0,P2d0=P2d0,P2m0=P2m0,n1d=n1d[:horizon+1],n1m=n1m_path[:horizon+1],n2d=n2d[:horizon+1],n2m=n2m_path[:horizon+1],P1X=P1X[:horizon+1],P2X=P2X[:horizon+1],d11=d11_path[:horizon+1],d12=d12_path[:horizon+1],d21=d21_path[:horizon+1],d22=d22_path[:horizon+1],T1=T1_path[:horizon+1],T2=T2_path[:horizon+1],initial_inflation=initial_inflation)

# Start from the exogenous/shocked paths at t=0. Indexation then generates
# subsequent periods causally from realized lagged inflation.
# W/E shocks remain effective even when their respective indexation is on.
shock_W_active = use_shock and shock_param == "W" and idx_w
shock_E_active = use_shock and shock_param == "e" and idx_e
shock_factor = 1.0 + shock_pct/100.0 if use_shock else 1.0
P1m_path = constant_path(P1m0,T); P2m_path = constant_path(P2m0,T)
W_base = W.copy(); E_base = E.copy()
W = W_base.copy(); E = E_base.copy()

# First-period values are given. For t>=1, indexed variables are generated
# from p_{t-1}; non-indexed variables retain their exogenous/shocked paths.
for t in range(1, T + 1):
    try:
        df_t, checks_t = run_model(W,E,n1m,n2m,t)
    except Exception as exc:
        st.error(f"A simulação não pôde ser executada no período {t}: {exc}"); st.stop()
    p_prev = float(np.nan_to_num(df_t["p"].iloc[t-1], nan=0.0))

    # t=0 carries the inherited growth rates supplied in the initial
    # conditions. Those rates generate the first transition, t=0 -> t=1.
    # From t=2 onward, if indexation is active, the indexation equations
    # determine the subsequent growth rates using the previous period's
    # realized inflation. This keeps the inherited rates independent of k's.
    if idx_w:
        W_growth = initial_w_growth if t == 1 else (kW + alphaW * p_prev)
        W[t] = W[t-1] * (1 + W_growth)
        if shock_W_active and t == shock_period:
            W[t] *= shock_factor
    else:
        W[t] = W_base[t]
    if idx_e:
        E_growth = initial_e_growth if t == 1 else (kE + alphaE * p_prev)
        E[t] = E[t-1] * (1 + E_growth)
        if shock_E_active and t == shock_period:
            E[t] *= shock_factor
    else:
        E[t] = E_base[t]

    prevP1 = df_t.P1.iloc[t-1]
    prevP2 = df_t.P2.iloc[t-1]
    c1 = (d11_path[t-1]*A[t-1,0,0]*prevP1
          + (1-d11_path[t-1])*A[t-1,0,0]*E[t-1]*P1X[t-1]
          + d21_path[t-1]*A[t-1,1,0]*prevP2
          + (1-d21_path[t-1])*A[t-1,1,0]*E[t-1]*P2X[t-1])
    c2 = (d12_path[t-1]*A[t-1,0,1]*prevP1
          + (1-d12_path[t-1])*A[t-1,0,1]*E[t-1]*P1X[t-1]
          + d22_path[t-1]*A[t-1,1,1]*prevP2
          + (1-d22_path[t-1])*A[t-1,1,1]*E[t-1]*P2X[t-1])
    if idx_pm1:
        p_m1_prev = initial_pm1_growth if t == 1 else (km1 + alpha_m1 * p_prev)
        P1m_path[t] = P1m_path[t-1] * (1 + p_m1_prev)
        if abs(c1) > 1e-12:
            n1m[t] = (P1m_path[t] / (1+T1_path[t]) - W[t]*l[t,0]) / c1 - 1
    if idx_pm2:
        p_m2_prev = initial_pm2_growth if t == 1 else (km2 + alpha_m2 * p_prev)
        P2m_path[t] = P2m_path[t-1] * (1 + p_m2_prev)
        if abs(c2) > 1e-12:
            n2m[t] = (P2m_path[t] / (1+T2_path[t]) - W[t]*l[t,1]) / c2 - 1

# Final full run using the causally generated paths.
try:
    df, checks = run_model(W,E,n1m,n2m,T)
except Exception as exc:
    st.error(f"A simulação não pôde ser executada: {exc}"); st.stop()

# Derived quantities used by the requested graph groups.
df["e_growth"] = df["e"].pct_change()
df["pm1_growth"] = df["P1m"].pct_change()
df["pm2_growth"] = df["P2m"].pct_change()
df["b_hat"] = df["b"].pct_change()

# Period 0 displays the user-specified inherited/initial growth rates.
# From period 1 onward, the indexation equations determine subsequent growth.
if idx_w:
    df.loc[df.index[0], "w"] = initial_w_growth
else:
    df.loc[df.index[0], "w"] = 0.0

# Initial real-wage growth: b = W/P, so b̂₀ = (1+w₀)/(1+p₀)-1.
# Subsequent periods are obtained directly from the simulated real-wage path.
df.loc[df.index[0], "b_hat"] = (1.0 + initial_w_growth) / (1.0 + initial_inflation) - 1.0
if idx_e:
    df.loc[df.index[0], "e_growth"] = initial_e_growth
else:
    df.loc[df.index[0], "e_growth"] = 0.0

# Period 0 also records the inherited growth rates of monitored prices.
# From period 1 onward these are calculated directly from the simulated Pᵐ paths.
df.loc[df.index[0], "pm1_growth"] = initial_pm1_growth
if special:
    df.loc[df.index[0], "pm2_growth"] = np.nan
else:
    df.loc[df.index[0], "pm2_growth"] = initial_pm2_growth

df["WE"] = df["W"] / df["e"]
for col in ["P","P1d","P1m","P1X","P2d","P2m","P2X"]:
    pass
df["P_rel"] = df["P"] / df["P1"]
df["P2_rel"] = df["P2"] / df["P1"]
df["P1d_rel"] = df["P1d"] / df["P1"]
df["P1m_rel"] = df["P1m"] / df["P1"]
df["P1X_rel"] = df["e"] * df["P1X"] / df["P1"]
df["P2d_rel"] = df["P2d"] / df["P1"]
df["P2m_rel"] = df["P2m"] / df["P1"]
df["P2X_rel"] = df["e"] * df["P2X"] / df["P1"]

# -----------------------------------------------------------------------------
# MAIN DASHBOARD — institutional + two-panel redesign
# -----------------------------------------------------------------------------
model_name = "Trigo" if special else "Trigo-Ferro"
status_ok = checks.get("max_rho_A", np.nan) < 1 if isinstance(checks, dict) else False

# -----------------------------------------------------------------------------
# PDF REPORT — generated from the exact current state of the simulation.
# It includes the current inputs, calculated indicators, all dashboard graphs,
# and a complete numerical appendix with every simulated period.
# -----------------------------------------------------------------------------
pdf_input_rows = [
    ("Modelo", model_name), ("Períodos (T)", T),
    ("a₁₁", a11), ("a₁₂", a12), ("a₂₁", a21), ("a₂₂", a22),
    ("l₁", l1), ("l₂", l2), ("X₁", X1), ("X₂", X2),
    ("d₁₁", d11), ("d₁₂", d12), ("d₂₁", d21), ("d₂₂", d22),
    ("T₁", T1), ("T₂", T2),
    ("x₁", x1), ("x₂", x2),
    ("φ₁d", phi1d), ("φ₁m", phi1m), ("φ₁x", phi1x),
    ("φ₂d", phi2d), ("φ₂m", phi2m), ("φ₂x", phi2x),
    ("W inicial", W0), ("w₀", initial_w_growth), ("p₀", initial_inflation),
    ("E inicial", E0), ("e₀", initial_e_growth),
    ("P₁d inicial", P1d0), ("P₁m inicial", P1m0),
    ("P₂d inicial", P2d0), ("P₂m inicial", P2m0),
    ("P₁ˣ inicial", P1X0), ("P₂ˣ inicial", P2X0),
    ("Indexação de preços monitorados", pm_indexation),
    ("Indexação de W", "Ativa" if idx_w else "Desativada"),
    ("F_W", Fw), ("F_P", Fp), ("k_W", kW), ("α_W", alphaW),
    ("Indexação de E", "Ativa" if idx_e else "Desativada"),
    ("k_E", kE), ("α_E", alphaE),
    ("k_m1", km1), ("α_m1", alpha_m1), ("pᵐ₁,₀", initial_pm1_growth),
    ("k_m2", km2), ("α_m2", alpha_m2), ("pᵐ₂,₀", initial_pm2_growth),
    ("Crescimento de l ativo", "Sim" if apply_l_growth else "Não"),
    ("g_l1", g_l1), ("g_l2", g_l2),
    ("Choque ativo", "Sim" if use_shock else "Não"),
]
if use_shock:
    pdf_input_rows.extend([
        ("Parâmetro do choque", PARAMETERS.get(shock_param, shock_param)),
        ("Tamanho do choque (%)", shock_pct),
        ("Período do choque", shock_period),
        ("Duração", shock_duration),
    ])

pdf_calculated_rows = [
    ("ρ(A)", fmt_num(rho0)),
    ("R", "∞" if np.isinf(R0) else fmt_num(R0)),
    ("B", "—" if not np.isfinite(B0) else fmt_num(B0)),
    ("P₁ implícito", fmt_num(P1_live)),
    ("P₂ implícito", fmt_num(P2_live)),
]

pdf_graph_groups = [
    ("INFLAÇÃO", GRAPH_GROUPS_ACTIVE["Inflação"]),
    ("SALÁRIO REAL", ["b", "b1"] if special else ["b", "b1", "b2"]),
    ("SALÁRIOS NOMINAIS E CÂMBIO", ["W", "e", "WE"]),
    ("PARCELA DO SALÁRIO E B", ["w_share", "B"]),
    ("MARGENS NOMINAIS", GRAPH_GROUPS_ACTIVE["n — margens nominais"]),
    ("MARGENS REAIS", GRAPH_GROUPS_ACTIVE["m — margens reais"]),
    ("PREÇOS EM UNIDADES DO BEM 1", GRAPH_GROUPS_ACTIVE["Preços em unidades do bem 1"]),
]

# PDF generation is intentionally lazy: building the report creates one
# high-resolution figure per series, so doing it on every Streamlit rerun
# makes the dashboard unnecessarily slow. The report is generated only when
# the user explicitly requests it.
pdf_signature = (
    model_name, T,
    tuple((str(k), repr(v)) for k, v in pdf_input_rows),
    tuple((str(k), repr(v)) for k, v in pdf_calculated_rows),
    tuple((str(g), tuple(vars_)) for g, vars_ in pdf_graph_groups),
    int(pd.util.hash_pandas_object(df, index=True).sum()),
)
if st.session_state.get("pdf_signature") != pdf_signature:
    st.session_state.pop("pdf_bytes", None)
    st.session_state.pop("pdf_signature", None)

# The report controls stay together in the blue banner.
pdf_ready = (
    st.session_state.get("pdf_signature") == pdf_signature
    and bool(st.session_state.get("pdf_bytes"))
)

header_left, header_generate, header_download = st.columns([4.2, 1.35, 1.35], gap="small")
with header_left:
    st.markdown(
        '<div class="tab-title">SIMULATOR TABAJARA</div>'
        '<div class="tab-subtitle">Seus problemas acabaram!</div>',
        unsafe_allow_html=True,
    )
with header_generate:
    generate_report = st.button(
        "Gerar relatório PDF",
        use_container_width=False,
        key="generate_pdf_report",
    )
with header_download:
    if pdf_ready:
        st.download_button(
            "Baixar relatório PDF",
            data=st.session_state["pdf_bytes"],
            file_name=f"{'trigo' if special else 'trigo_ferro'}_relatorio.pdf",
            mime="application/pdf",
            use_container_width=False,
            key="download_pdf_report",
        )

if generate_report:
    with st.spinner("Gerando relatório PDF…"):
        st.session_state["pdf_bytes"] = build_results_pdf(
            df, pdf_input_rows, pdf_calculated_rows, pdf_graph_groups, special=special
        )
        st.session_state["pdf_signature"] = pdf_signature
    st.rerun()

kpi = st.columns([1.1, 1.0, 1.0, 1.0, 1.0, 1.55])
summary = [
    ("Modelo", model_name),
    ("Períodos (T)", f"{T}"),
    ("ρ(A)", fmt_num(rho0)),
    ("R", "∞" if np.isinf(R0) else fmt_num(R0)),
    ("B", "—" if not np.isfinite(B0) else fmt_num(B0)),
]
for c, (label, value) in zip(kpi[:5], summary):
    with c:
        st.markdown(f'<div class="kpi-label">{label}</div><div class="kpi-value">{value}</div>', unsafe_allow_html=True)
with kpi[5]:
    st.markdown('<div class="kpi-label">Status do modelo</div>', unsafe_allow_html=True)
    if status_ok:
        st.markdown('<div class="kpi-status"><span class="status-dot"></span>Tecnologia válida<br>&nbsp;&nbsp;&nbsp;ρ(A) &lt; 1</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="kpi-status">⚠ Tecnologia fora da condição ρ(A) &lt; 1</div>', unsafe_allow_html=True)


def check_group(options, defaults, key_prefix, cols_per_row=None):
    """Visible checkbox toolbar: every series can be turned on/off individually."""
    n = len(options) if cols_per_row is None else cols_per_row
    selected = []
    for i in range(0, len(options), n):
        row = st.columns(min(n, len(options)-i), gap="small")
        for col, v in zip(row, options[i:i+n]):
            with col:
                if st.checkbox(LABELS.get(v, v), value=v in defaults, key=f"{key_prefix}_{v}"):
                    selected.append(v)
    return selected

def render_chart(title, subtitle, variables, defaults, key_prefix, height=280, cols_per_row=5):
    with st.container(border=True):
        st.markdown(
            f'<div class="chart-title">{title}</div><div class="chart-subtitle">{subtitle}</div>',
            unsafe_allow_html=True,
        )
        selected = check_group(variables, defaults, key_prefix, cols_per_row)
        if selected:
            st.plotly_chart(
                plot_series(df, selected, title, height),
                use_container_width=True,
                key=f"graph_{key_prefix}",
                config={"displayModeBar": False},
            )

# The order and proportions follow the approved A+D mockup. Margin and price
# charts are intentionally full-width and stacked vertically for readability.
render_chart(
    "INFLAÇÃO",
    "Crescimento nominal dos salários, câmbio e preços",
    GRAPH_GROUPS_ACTIVE["Inflação"],
    GRAPH_GROUPS_ACTIVE["Inflação"],
    "inflation",
    440,
    5,
)

c1, c2 = st.columns(2, gap="medium")
with c1:
    with st.container(border=True):
        st.markdown('<div class="chart-title">SALÁRIO REAL</div><div class="chart-subtitle">Salário real agregado e por setor</div>', unsafe_allow_html=True)
        real_wage_vars = ["b", "b1"] if special else ["b", "b1", "b2"]
        selected_real_wage = check_group(real_wage_vars, real_wage_vars, "real", 3)
        if selected_real_wage:
            st.plotly_chart(plot_series(df, selected_real_wage, "Salário real", 380), use_container_width=True, key="graph_real_wage", config={"displayModeBar": False})
with c2:
    with st.container(border=True):
        st.markdown('<div class="chart-title">SALÁRIOS NOMINAIS E CÂMBIO</div><div class="chart-subtitle">W, E e razão W/E</div>', unsafe_allow_html=True)
        selected_we = check_group(["W", "e", "WE"], ["W", "e", "WE"], "we", 3)
        if selected_we:
            st.plotly_chart(plot_series(df, selected_we, "W, E e W/E", 380), use_container_width=True, key="graph_we", config={"displayModeBar": False})

render_chart(
    "PARCELA DO SALÁRIO E B",
    "Parcela do salário na renda e parâmetro B",
    ["w_share", "B"],
    ["w_share", "B"],
    "wshare_B",
    300,
    2,
)

render_chart(
    "MARGENS NOMINAIS",
    "Margens por setor e origem",
    GRAPH_GROUPS_ACTIVE["n — margens nominais"],
    GRAPH_GROUPS_ACTIVE["n — margens nominais"][:3],
    "nominal_margins",
    430,
    5,
)

render_chart(
    "MARGENS REAIS",
    "Margens em termos do bem 1",
    GRAPH_GROUPS_ACTIVE["m — margens reais"],
    GRAPH_GROUPS_ACTIVE["m — margens reais"][:3],
    "real_margins",
    430,
    5,
)

render_chart(
    "PREÇOS EM UNIDADES DO BEM 1",
    "Preços relativos",
    GRAPH_GROUPS_ACTIVE["Preços em unidades do bem 1"],
    GRAPH_GROUPS_ACTIVE["Preços em unidades do bem 1"][:3],
    "relative_prices",
    430,
    5,
)


with st.expander("▸  Ver todas as equações", expanded=False):
    for q in REPORT_EQUATIONS:
        st.latex(q)

with st.expander("Dados da simulação", False):
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button("Baixar resultados em CSV", df.to_csv(index=False).encode(), "simtrigo_trigo_ferro_results.csv", "text/csv")
