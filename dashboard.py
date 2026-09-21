"""SimTrigo — Trigo/Ferro two-good dashboard.

The dashboard follows the SimTrigo interaction pattern:
- A, l and X are the first model inputs.
- rho(A), R and B are calculated immediately from those inputs.
- shocks use one dropdown containing the model parameters.
- indexation mechanisms are independently activated with checkboxes.
- the b-r diagram plots the actual simulated pairs (m_t, b_t), with b on the vertical axis and aggregate real profit margin m on the horizontal axis.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from model import simulate_two_good_model, simulate_one_good_special_case

st.set_page_config(page_title="Simulator Tabajara", page_icon="📈", layout="wide")

st.markdown("""
<style>
.block-container{max-width:1800px;padding-top:2.8rem;padding-bottom:1.4rem;padding-left:1rem;padding-right:1rem}
.tab-header{margin-top:0;background:linear-gradient(90deg,#163f68 0%,#245b89 100%);color:#fff;margin:0 -1rem 1rem;padding:2.05rem 2.25rem 1.35rem;border-bottom:1px solid #173b5f;overflow:visible}
.tab-title{font-family:Georgia,serif;font-size:2.15rem;line-height:1.2;font-weight:700;letter-spacing:.01em;white-space:nowrap}
.tab-subtitle{font-family:Georgia,serif;font-size:1.08rem;font-weight:600;margin-top:.38rem}
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
    "e_growth":"Crescimento do câmbio (e)", "w_share":"Parcela do salário (w_share)",
    "b":"Salário real (b)", "b1":"Salário real em trigo (b₁)", "b2":"Salário real em ferro (b₂)",
    "WE":"W/E", "n":"n agregado", "n1":"n₁", "n2":"n₂",
    "n1d":"n₁ᵈ", "n1m":"n₁ᵐ", "n1X":"n₁ˣ", "n2d":"n₂ᵈ", "n2m":"n₂ᵐ", "n2X":"n₂ˣ",
    "m":"m agregado", "m1":"m₁", "m2":"m₂", "m1d":"m₁ᵈ", "m1m":"m₁ᵐ", "m1X":"m₁ˣ", "m2d":"m₂ᵈ", "m2m":"m₂ᵐ", "m2X":"m₂ˣ",
    "X1":"Produção X₁", "X2":"Produção X₂", "l1":"Coeficiente l₁", "l2":"Coeficiente l₂",
    "a11":"a₁₁", "a12":"a₁₂", "a21":"a₂₁", "a22":"a₂₂", "x1":"Peso x₁", "x2":"Peso x₂",
    "P1d":"P₁ᵈ", "P1m":"P₁ᵐ", "P1X":"P₁ˣ", "P2d":"P₂ᵈ", "P2m":"P₂ᵐ", "P2X":"P₂ˣ",
    "rho_A":"ρ(A)", "R":"Taxa máxima de lucro (R)", "B":"B",
    "P_rel":"P/P₁", "P2_rel":"P₂/P₁", "P1d_rel":"P₁ᵈ/P₁", "P1m_rel":"P₁ᵐ/P₁", "P1X_rel":"EP₁ˣ/P₁",
    "P2d_rel":"P₂ᵈ/P₁", "P2m_rel":"P₂ᵐ/P₁", "P2X_rel":"EP₂ˣ/P₁",
}

GRAPH_GROUPS = {
    "Inflação": ["w", "e_growth", "p", "p1", "p2"],
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
}


def num(label, value, key, min_value=None, step=.01, max_value=None):
    kw = dict(label=label, value=float(value), step=float(step), format="%.6f", key=key)
    if min_value is not None:
        kw["min_value"] = float(min_value)
    if max_value is not None:
        kw["max_value"] = float(max_value)
    return st.number_input(**kw)


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
            "Inflação": ["w", "e_growth", "p", "p1"],
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
        X1 = num("X₁", .10, "X1", 0)
        X2 = 0.0 if special else num("X₂", .10, "X2", 0)

        apply_l_growth = st.checkbox("Aplicar crescimento aos coeficientes de trabalho (l)", False, key="apply_l_growth")
        if apply_l_growth:
            g_l1 = num("Taxa de crescimento de l₁", 0.0, "g_l1", min_value=-1.0, max_value=.99, step=.005)
            if special:
                g_l2 = 0.0
            else:
                g_l2 = num("Taxa de crescimento de l₂", 0.0, "g_l2", min_value=-1.0, max_value=.99, step=.005)
            st.latex(r"l_{1,t}=(1-g_{l1})l_{1,t-1}")
            if not special:
                st.latex(r"l_{2,t}=(1-g_{l2})l_{2,t-1}")
            st.caption("As taxas entram com sinal invertido: g_li > 0 reduz o respectivo coeficiente l ao longo do tempo.")
        else:
            g_l1 = g_l2 = 0.0

        A0 = np.array([[a11,a12],[a21,a22]], dtype=float)
        l0 = np.array([l1,l2], dtype=float)
        X0 = np.array([X1,X2], dtype=float)
        rho0, B0, R0, B_err = live_technology(A0,l0,X0)

        st.markdown('<div class="calc-card">', unsafe_allow_html=True)
        st.markdown("**Resultados calculados**")
        c1, c2, c3 = st.columns(3)
        c1.metric("ρ(A)", f"{rho0:.4f}")
        c2.metric("R", "∞" if np.isinf(R0) else f"{R0:.6f}")
        c3.metric("B", "—" if not np.isfinite(B0) else f"{B0:.4f}")
        if B_err:
            st.caption(B_err)
        st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("Estrutura de mercados", expanded=False):
        if special:
            x1 = 1.0
            x2 = 0.0
            st.number_input("x₁", value=1.0, min_value=0.0, max_value=1.0, step=0.01, format="%.6f", disabled=True, key="x1_trigo_display")
            st.caption("No caso Trigo: x₁=1 e x₂=0.")
        else:
            x1 = num("x₁", .60, "x1", 0, .01)
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
        st.caption(f"x₂={x2:.4f} | φ₁x={phi1x:.4f} | φ₂x={phi2x:.4f}")
        if phi1x < 0 or phi2x < 0:
            st.error("Os pesos φᵈ+φᵐ não podem exceder 1.")

    with st.expander("Condições iniciais", expanded=False):
        P10 = num("P₁ inicial",1.0,"P10",1e-9)
        P20 = 1.0 if special else num("P₂ inicial",1.0,"P20",1e-9)
        W0 = num("W inicial",1.0,"W0",1e-9)
        E0 = num("E inicial",1.0,"E0",1e-9)
        P1d0 = num("P₁d inicial",1.0,"P1d0",1e-9)
        P1m0 = num("P₁m inicial",1.0,"P1m0",1e-9)
        P2d0 = 1.0 if special else num("P₂d inicial",1.0,"P2d0",1e-9)
        P2m0 = 1.0 if special else num("P₂m inicial",1.0,"P2m0",1e-9)
        P1X0 = num("P₁ˣ",1.0,"P1X0",1e-9)
        P2X0 = 1.0 if special else num("P₂ˣ",1.0,"P2X0",1e-9)

        # Initial endogenous variables implied directly by the inputs.
        P0_live = x1*P10 + x2*P20
        b0_live = W0/P0_live if abs(P0_live)>1e-12 else np.nan
        WE0_live = W0/E0 if abs(E0)>1e-12 else np.nan
        c10_live = a11*P10 + a21*P20
        c20_live = a12*P10 + a22*P20
        m1d0_live = (P1d0-W0*l1)/c10_live-1 if abs(c10_live)>1e-12 else np.nan
        m1m0_live = (P1m0-W0*l1)/c10_live-1 if abs(c10_live)>1e-12 else np.nan
        m1X0_live = (E0*P1X0-W0*l1)/c10_live-1 if abs(c10_live)>1e-12 else np.nan
        m2d0_live = (P2d0-W0*l2)/c20_live-1 if (not special and abs(c20_live)>1e-12) else np.nan
        m2m0_live = (P2m0-W0*l2)/c20_live-1 if (not special and abs(c20_live)>1e-12) else np.nan
        m2X0_live = (E0*P2X0-W0*l2)/c20_live-1 if (not special and abs(c20_live)>1e-12) else np.nan

        # At t=0 there is no t=-1. The model therefore initializes the
        # exogenous nominal margins from the calculated real margins: n(0)=m(0).
        n1d0 = m1d0_live
        n1m0 = m1m0_live
        n2d0 = m2d0_live if not special else 0.0
        n2m0 = m2m0_live if not special else 0.0

    with st.expander("Indexação", expanded=False):
        pm_indexation = st.selectbox(
            "Indexar preços monitorados",
            ["Nenhum", "Monitorados no bem 1", "Monitorados no bem 2", "Ambos"],
            index=0,
            key="pm_indexation",
        )
        idx_pm1 = pm_indexation in {"Monitorados no bem 1", "Ambos"}
        idx_pm2 = (not special) and pm_indexation in {"Monitorados no bem 2", "Ambos"}
        if idx_pm1 or idx_pm2:
            st.latex(r"P^m_{i,t}=P^m_{i,t-1}(1+p_{t-1})")

        idx_w = st.checkbox("Indexar salário nominal (W)", True, key="idx_w")
        if idx_w:
            Fw = num("F_W",1.0,"Fw",0.0,.25)
            Fp = num("F_P",1.0,"Fp",0.01,.25)
            kW = num("k_W",0.01,"kW",step=.005)
            alphaW = Fw/(Fw+Fp)
            st.latex(r"\alpha_W=\frac{F_W}{F_W+F_P}=%.6f" % alphaW)
            st.latex(r"w_t=k_W+\alpha_Wp_{t-1}")
        else:
            Fw=Fp=kW=alphaW=0.0

        idx_e = st.checkbox("Indexar taxa de câmbio (E)", True, key="idx_e")
        if idx_e:
            alphaE = num("α_E",1.0,"alphaE",step=.05)
            kE = num("k_E",0.0,"kE",step=.005)
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
        l[t,0] = l[t-1,0] * (1.0 - g_l1)
        l[t,1] = l[t-1,1] * (1.0 - g_l2)
else:
    l[:] = [l1, l2]
X = np.tile([X1,X2], (T+1,1))
x1_path = constant_path(x1,T); x2_path = 1-x1_path
phi = np.tile(np.array([[phi1d,phi1m,phi1x],[phi2d,phi2m,phi2x]],float),(T+1,1,1))
W = constant_path(W0,T); E = constant_path(E0,T)
P1X = constant_path(P1X0,T); P2X = constant_path(P2X0,T)
n1d = constant_path(n1d0,T); n1m = constant_path(n1m0,T); n2d = constant_path(n2d0,T); n2m = constant_path(n2m0,T)

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

# Validate phi after shocks.
if np.any(phi[:,:,2] < -1e-12):
    st.error("O choque escolhido tornou algum peso φˣ negativo. Reduza o tamanho do choque.")
    st.stop()

# Sequential simulation with causal indexation.
# W_t, E_t and indexed P^m_t depend only on p_{t-1}; they therefore must be
# updated period by period, not by a global fixed-point iteration.
def run_model(W_path,E_path,n1m_path,n2m_path, horizon):
    if special:
        return simulate_one_good_special_case(horizon,A[:horizon+1,0,0],l[:horizon+1,0],X[:horizon+1,0],W_path[:horizon+1],E_path[:horizon+1],phi[:horizon+1,0,:],n1d[:horizon+1],n1m_path[:horizon+1],P1X[:horizon+1],P10=P10,P1d0=P1d0,P1m0=P1m0)
    return simulate_two_good_model(horizon,A[:horizon+1],l[:horizon+1],X[:horizon+1],W_path[:horizon+1],E_path[:horizon+1],phi[:horizon+1],x1_path[:horizon+1],x2_path[:horizon+1],P10=P10,P20=P20,P1d0=P1d0,P1m0=P1m0,P2d0=P2d0,P2m0=P2m0,n1d=n1d[:horizon+1],n1m=n1m_path[:horizon+1],n2d=n2d[:horizon+1],n2m=n2m_path[:horizon+1],P1X=P1X[:horizon+1],P2X=P2X[:horizon+1])

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

    if idx_w:
        W[t] = W[t-1] * (1 + kW + alphaW * p_prev)
        if shock_W_active and t == shock_period:
            W[t] *= shock_factor
    else:
        W[t] = W_base[t]
    if idx_e:
        E[t] = E[t-1] * (1 + kE + alphaE * p_prev)
        if shock_E_active and t == shock_period:
            E[t] *= shock_factor
    else:
        E[t] = E_base[t]

    prevP1 = P10 if t == 1 else df_t.P1.iloc[t-1]
    prevP2 = P20 if t == 1 else df_t.P2.iloc[t-1]
    c1 = A[t-1,0,0]*prevP1 + A[t-1,1,0]*prevP2
    c2 = A[t-1,0,1]*prevP1 + A[t-1,1,1]*prevP2
    if idx_pm1:
        P1m_path[t] = P1m_path[t-1] * (1 + p_prev)
        if abs(c1) > 1e-12:
            n1m[t] = (P1m_path[t] - W[t]*l[t,0]) / c1 - 1
    if idx_pm2:
        P2m_path[t] = P2m_path[t-1] * (1 + p_prev)
        if abs(c2) > 1e-12:
            n2m[t] = (P2m_path[t] - W[t]*l[t,1]) / c2 - 1

# Final full run using the causally generated paths.
try:
    df, checks = run_model(W,E,n1m,n2m,T)
except Exception as exc:
    st.error(f"A simulação não pôde ser executada: {exc}"); st.stop()

# Derived quantities used by the requested graph groups.
df["e_growth"] = df["e"].pct_change()
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
st.markdown(
    '<div class="tab-header"><div class="tab-title">SIMULATOR TABAJARA</div>'
    '<div class="tab-subtitle">Seus problemas acabaram!</div></div>',
    unsafe_allow_html=True,
)

model_name = "Trigo" if special else "Trigo-Ferro"
status_ok = checks.get("max_rho_A", np.nan) < 1 if isinstance(checks, dict) else False

kpi = st.columns([1.1, 1.0, 1.0, 1.0, 1.0, 1.55])
summary = [
    ("Modelo", model_name),
    ("Períodos (T)", f"{T}"),
    ("ρ(A)", f"{rho0:.6f}"),
    ("R", "∞" if np.isinf(R0) else f"{R0:.6f}"),
    ("B", "—" if not np.isfinite(B0) else f"{B0:.6f}"),
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
    "PARCELA DO SALÁRIO",
    "Parcela do salário na renda",
    ["w_share"],
    ["w_share"],
    "wshare",
    300,
    1,
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
    equations = [
        r"A_t=\begin{pmatrix}a_{11,t}&a_{12,t}\\a_{21,t}&a_{22,t}\end{pmatrix}",
        r"B_t=\frac{1}{l_t[I-A_t]^{-1}X_t}",
        r"\rho(A_t)=\max_i|\lambda_i(A_t)|",
        r"R_t=\frac{1}{\rho(A_t)}-1",
        r"P_t=x_{1,t}P_{1,t}+x_{2,t}P_{2,t},\quad x_{1,t}+x_{2,t}=1",
        r"p_t=\frac{P_t}{P_{t-1}}-1",
        r"w_t=\frac{W_t}{W_{t-1}}-1,\qquad e_t=\frac{E_t}{E_{t-1}}-1",
        r"1+n_{1,t}^{d}=\frac{P_{1,t}^{d}-W_tl_{1,t}}{a_{11,t-1}P_{1,t-1}+a_{21,t-1}P_{2,t-1}}",
        r"1+n_{1,t}^{m}=\frac{P_{1,t}^{m}-W_tl_{1,t}}{a_{11,t-1}P_{1,t-1}+a_{21,t-1}P_{2,t-1}}",
        r"1+n_{1,t}^{X}=\frac{E_tP_{1,t}^{X}-W_tl_{1,t}}{a_{11,t-1}P_{1,t-1}+a_{21,t-1}P_{2,t-1}}",
        r"1+n_{2,t}^{d}=\frac{P_{2,t}^{d}-W_tl_{2,t}}{a_{12,t-1}P_{1,t-1}+a_{22,t-1}P_{2,t-1}}",
        r"1+n_{2,t}^{m}=\frac{P_{2,t}^{m}-W_tl_{2,t}}{a_{12,t-1}P_{1,t-1}+a_{22,t-1}P_{2,t-1}}",
        r"1+n_{2,t}^{X}=\frac{E_tP_{2,t}^{X}-W_tl_{2,t}}{a_{12,t-1}P_{1,t-1}+a_{22,t-1}P_{2,t-1}}",
        r"1+m_{1,t}^{d}=\frac{P_{1,t}^{d}-W_tl_{1,t}}{a_{11,t}P_{1,t}+a_{21,t}P_{2,t}}",
        r"1+m_{1,t}^{m}=\frac{P_{1,t}^{m}-W_tl_{1,t}}{a_{11,t}P_{1,t}+a_{21,t}P_{2,t}}",
        r"1+m_{1,t}^{X}=\frac{E_tP_{1,t}^{X}-W_tl_{1,t}}{a_{11,t}P_{1,t}+a_{21,t}P_{2,t}}",
        r"1+m_{2,t}^{d}=\frac{P_{2,t}^{d}-W_tl_{2,t}}{a_{12,t}P_{1,t}+a_{22,t}P_{2,t}}",
        r"1+m_{2,t}^{m}=\frac{P_{2,t}^{m}-W_tl_{2,t}}{a_{12,t}P_{1,t}+a_{22,t}P_{2,t}}",
        r"1+m_{2,t}^{X}=\frac{E_tP_{2,t}^{X}-W_tl_{2,t}}{a_{12,t}P_{1,t}+a_{22,t}P_{2,t}}",
        r"P_{i,t}^{m}=P_{i,t-1}^{m}(1+p_{t-1})\quad\text{(quando indexado)}",
        r"w_t=k_W+\alpha_Wp_{t-1},\quad \alpha_W=\frac{F_W}{F_W+F_P}\quad\text{(quando indexado)}",
        r"e_t=k_E+\alpha_Ep_{t-1}\quad\text{(quando indexado)}",
        r"w_{share,t}=\frac{W_t\,l_tX_t}{\mathbf P_t[I-A_t]X_t}",
    ]
    for q in equations:
        st.latex(q)

with st.expander("Dados da simulação", False):
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button("Baixar resultados em CSV", df.to_csv(index=False).encode(), "simtrigo_trigo_ferro_results.csv", "text/csv")
