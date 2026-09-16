import streamlit as st
import re
import numpy as np
import plotly.graph_objects as go
from model import (
    INITIAL_VALUES, SHOCKABLE, OUTPUTS,
    baseline_model, simulate_parameter_shock,
    simulate_custom_indexation, simulate_pa_indexation, simulate_L_growth,
    simulate_combined
)


st.set_page_config(
    page_title="SimTrigo — Simulador",
    page_icon="📈",
    layout="wide"
)

st.markdown("""
<style>
/* ------------------------------------------------------------------
   SimTrigo — institutional light interface
   ------------------------------------------------------------------ */
.block-container {
    max-width: 1480px;
    padding-top: 1.25rem;
    padding-bottom: 3rem;
}

.simtrigo-header {
    border-left: 5px solid #1f4e79;
    padding: 1.1rem 1.4rem;
    margin: 0 0 1.25rem 0;
    background: linear-gradient(90deg, #f3f6f9 0%, #ffffff 100%);
    border-top: 1px solid #d9e2ea;
    border-right: 1px solid #d9e2ea;
    border-bottom: 1px solid #d9e2ea;
    border-radius: 0 8px 8px 0;
    box-shadow: 0 1px 2px rgba(15, 23, 42, .04);
}
.simtrigo-title {
    margin: 0;
    color: #17365d;
    font-size: 2.15rem;
    font-weight: 750;
    letter-spacing: -0.03em;
}
.simtrigo-subtitle {
    margin-top: .35rem;
    color: #5d6872;
    font-size: .95rem;
}

/* Sidebar */
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] > div {
    background: #e7f0f8 !important;
}
section[data-testid="stSidebar"] {
    border-right: 1px solid #c5d5e4;
}
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
    color: #17212b !important;
}
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] textarea,
section[data-testid="stSidebar"] [data-baseweb="select"] > div {
    color: #17212b !important;
    background: #ffffff !important;
    border-color: #b9c9d8 !important;
}
section[data-testid="stSidebar"] [data-testid="stNumberInputContainer"] button {
    background: #f7fafc !important;
    color: #17365d !important;
}
.sidebar-section {
    margin: 1rem 0 .45rem 0;
    padding: .55rem .7rem;
    background: rgba(255,255,255,.78);
    border: 1px solid #bfd0df;
    border-left: 3px solid #1f4e79;
    border-radius: 6px;
    color: #17365d;
    font-size: .76rem;
    font-weight: 750;
    text-transform: uppercase;
    letter-spacing: .055em;
}

/* Metrics */
[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #d6dfe7;
    border-radius: 8px;
    padding: .85rem .95rem;
    box-shadow: 0 1px 2px rgba(15, 23, 42, .04);
}
[data-testid="stMetricLabel"] {
    color: #5d6872 !important;
    font-size: .78rem !important;
    font-weight: 600 !important;
}
[data-testid="stMetricValue"] {
    color: #17365d !important;
    font-size: 1.38rem !important;
    font-weight: 700 !important;
}

/* Sections and cards */
.result-section {
    margin-top: .35rem;
    padding: .75rem 0 .35rem 0;
    border-bottom: 2px solid #dce4eb;
}
.result-section-title {
    color: #17365d;
    font-size: 1.25rem;
    font-weight: 750;
}
.result-section-note {
    color: #68737d;
    font-size: .86rem;
    margin-top: .15rem;
}
.chart-card {
    border: 1px solid #d8e0e7;
    border-radius: 8px;
    padding: .25rem .45rem .1rem .45rem;
    background: #ffffff;
    margin: .65rem 0 1rem 0;
    box-shadow: 0 1px 3px rgba(15, 23, 42, .035);
}
hr {
    border-color: #dce3e9 !important;
}

/* Expanders / controls */
div[data-testid="stExpander"] {
    border: 1px solid #d7e0e7;
    border-radius: 8px;
    background: #ffffff;
}
div[data-testid="stExpander"] summary {
    color: #17365d;
    font-weight: 650;
}
button[kind="secondary"],
button[kind="primary"] {
    border-radius: 6px !important;
    font-weight: 600 !important;
}

/* Download button */
.stDownloadButton button {
    border: 1px solid #1f4e79 !important;
    color: #1f4e79 !important;
    background: #ffffff !important;
    border-radius: 6px !important;
    font-weight: 650 !important;
}
.stDownloadButton button:hover {
    background: #eef4f9 !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="simtrigo-header">
    <div class="simtrigo-title">SimTrigo</div>
    <div class="simtrigo-subtitle">
        Simulador do modelo de formação de preços, salários, margens e indexação
    </div>
</div>
""", unsafe_allow_html=True)

DISPLAY_LABELS = {
    "p": "Inflação (p)",
    "W": "Salário nominal (W)",
    "w": "Crescimento do salário nominal (w)",
    "b": "Salário real (b)",
    "WE": "Salário/câmbio (W/E)",
    "N": "1+Margem nominal agregada (N)",
    "N_x": "1+Margem nominal — exportações (N_x)",
    "N_a": "1+Margem nominal — administrados (N_m)",
    "N_d": "1+Margem nominal — domésticos (N_d)",
    "M": "1+Margem real agregada (M)",
    "M_x": "1+Margem real — exportações (M_x)",
    "M_a": "1+Margem real — administrados (M_m)",
    "M_d": "1+Margem real — domésticos (M_d)",
    "P": "Preço agregado (P)",
    "P_ap": "Preço adminsitrados relativo",
    "P_dp": "Preço doméstico relativo",
    "P_xp": "Preço de exportação relativo",
}

def graph_label(var):
    return DISPLAY_LABELS.get(var, var)

VARIABLE_GROUPS = {
    "Salários, câmbio e inflação": ["p", "w", "W", "b", "WE"],
    "Margens nominais": ["N", "N_x", "N_a", "N_d"],
    "Margens reais": ["M", "M_x", "M_a", "M_d"],
    "Preços": ["P", "P_ap", "P_dp", "P_xp"],
}





# ---------------------------------------------------------------------------
# Fundamental equations
# ---------------------------------------------------------------------------
fundamental_equations = [
    r"""P_t = W_t l + aP_{t-1}(1+n_t)""",
    r"""P_t = \phi_d P^d_t + \phi_m P^m_t + \phi_x E_t P^X_t""",
    r"""1+n^X_t =\frac{E_tP^X_t-W_tl}{aP_{t-1}}""",
    r"""1+n^X_t =\frac{E_tP^X_t-W_tl}{a(\phi_d P^d_{t-1} + \phi_m P^m_{t-1} + \phi_x E_{t-1}P^X_{t-1})}""",
    r"""1+n_t =\phi_d(1+n^d_t)+\phi_m(1+n^m_t) +\phi_x(1+n^X_t)""",
    r"""P^d_t =aP_{t-1}(1+n^d_t)+W_tl""",
    r"""P^m_t = aP_{t-1}(1+n^m_t)+W_tl""",
    r"""b_t=\frac{W_t}{P_t}""",
    r"""p_t=\frac{P_t}{P_{t-1}}-1""",
    r"""w_t=\frac{W_t}{W_{t-1}}-1""",
    r"""e_t=\frac{E_t}{E_{t-1}}-1""",
    r"""1+m_t=\frac{1+n_t}{1+p_t}"""
]

with st.expander("📐 Equações", expanded=False):
    st.markdown("### Equações")
    
    for i, equation in enumerate(fundamental_equations, start=1):
        st.latex(equation)

with st.sidebar:
    st.header("Simulação")

    st.markdown('<div class="sidebar-section">Parâmetros do modelo</div>', unsafe_allow_html=True)
    T = st.number_input("Períodos (T)", 3, 500, 90, 1)
    a_0 = st.number_input("$a$ inicial", value=INITIAL_VALUES["a_0"], format="%.6f")
    L_0 = st.number_input("$l$ inicial", value=INITIAL_VALUES["L_0"], format="%.6f")
    P_0 = st.number_input("$P$ inicial", value=INITIAL_VALUES["P_0"], format="%.6f")
    W_0 = st.number_input("$W$ inicial", value=INITIAL_VALUES["W_0"], format="%.6f")
    E_0 = st.number_input("$E$ inicial", value=INITIAL_VALUES["E_0"], format="%.6f")
    P_d0 = st.number_input("$P^d$ inicial", value=INITIAL_VALUES["P_d0"], format="%.6f")
    P_a0 = st.number_input("$P^m$ inicial", value=INITIAL_VALUES["P_a0"], format="%.6f")
    P_x0 = st.number_input("$P^x$ inicial", value=INITIAL_VALUES["P_x0"], format="%.6f")

    st.markdown('<div class="sidebar-section">Tridentes</div>', unsafe_allow_html=True)
    f_d = st.number_input(r"$\varphi_d$", value=0.3, format="%.6f")
    f_a = st.number_input(r"$\varphi_m$", value=0.3, format="%.6f")
    f_x = st.number_input(r"$\varphi_x$", value=0.4, format="%.6f")

    st.markdown('<div class="sidebar-section">Indexação de W e E</div>', unsafe_allow_html=True)
    st.markdown(r"$w_t = k_W + \alpha_W p_{t-1}$")
    wage_k = st.number_input("$k$ do salário", value=0.05, step=0.001, format="%.4f")
    F_w = st.number_input("$F_W$ — frequência de ajuste dos salários", min_value=0.01, value=1.0, step=0.25, format="%.2f")
    F_p = st.number_input("$F_P$ — frequência de ajuste dos preços", min_value=0.01, value=1.0, step=0.25, format="%.2f")
    alpha_w = F_w / (F_w + F_p)
    st.caption(f"$\\alpha_W = F_W/(F_W+F_P) = {alpha_w:.4f}$")

    st.markdown(r"$e_t = k_E + \alpha_E p_{t-1}$")
    exchange_k = st.number_input("$k$ do câmbio", value=0.0, step=0.001, format="%.4f")
    alpha_e = st.number_input("$\\alpha_E$", value=1.0, step=0.05, format="%.4f")

    # Optional effects: only the three switches are shown initially.
    # Their configuration widgets are rendered immediately below each switch
    # only when that effect is activated.
    variable = SHOCKABLE[0]
    shock_pct = 5.0
    shock_period = 2
    persistence = "Permanente"

    pa_k = 0.0
    pa_alpha = 1.0

    L_growth = 0.05

    activate_shock = st.checkbox("Ativar choque estrutural", value=False, key="activate_shock_checkbox")
    if activate_shock:
        st.markdown("**Configuração do choque estrutural**")
        variable = st.selectbox(
            "Parâmetro / variável afetada",
            SHOCKABLE,
            format_func=lambda v: DISPLAY_LABELS.get(v, v),
        )
        shock_pct = st.number_input("Tamanho do choque (%)", value=5.0, format="%.3f")
        shock_period = st.number_input(
            "Período do choque", min_value=1, max_value=int(T), value=2, step=1
        )
        persistence = st.selectbox(
            "Duração do choque", ["Permanente", "Um período"]
        )

    activate_pa = st.checkbox("Ativar indexação de $P^m$", value=False, key="activate_pa_checkbox")
    if activate_pa:
        st.markdown("**Configuração da indexação de $P^m$**")
        st.markdown(r"$p^m_t = k_m + \alpha_m p_{t-1}$")
        pa_k = st.number_input("$k$ de $P^m$", value=0.0, step=0.001, format="%.4f")
        pa_alpha = st.number_input(
            "$\\alpha_m$ de $P^m$", value=1.0, step=0.05, format="%.4f"
        )

    activate_L = st.checkbox("Ativar crescimento de $l$", value=False, key="activate_L_checkbox")
    if activate_L:
        st.markdown("**Configuração do progresso técnico (redução de $l$)**")
        L_growth = st.number_input(
            "Taxa de progresso técnico (redução de $l$)", value=0.05, step=0.001, format="%.4f"
        )

    st.divider()
    st.markdown('<div class="sidebar-section">Efeitos ativados</div>', unsafe_allow_html=True)



par = {
    "T": T, "a_0": a_0, "L_0": L_0, "P_0": P_0, "W_0": W_0, "E_0": E_0,
    "P_d0": P_d0, "P_a0": P_a0, "P_x0": P_x0,
    "f_d": f_d, "f_a": f_a, "f_x": f_x
}

df = simulate_combined(
    par,
    wage_k=wage_k, F_w=F_w, F_p=F_p,
    exchange_k=exchange_k, exchange_alpha=alpha_e,
    activate_shock=activate_shock,
    variable=variable, shock_pct=shock_pct,
    shock_period=shock_period, persistence=persistence,
    activate_pa=activate_pa, pa_k=pa_k, pa_alpha=pa_alpha,
    activate_L=activate_L, L_growth=L_growth,
)

active_effects = []
if activate_shock:
    active_effects.append(f"choque estrutural: {DISPLAY_LABELS.get(variable, variable)} {shock_pct:+.2f}% no período {shock_period} ({persistence.lower()})")
if activate_pa:
    active_effects.append("indexação de $P^m$")
if activate_L:
    active_effects.append(f"crescimento de $l$ ({L_growth:+.4f})")

if active_effects:
    st.success("Efeitos ativados simultaneamente: " + "; ".join(active_effects) + ". W e E seguem a indexação configurada.")
else:
    st.info("Nenhum efeito opcional ativado. W e E seguem a indexação configurada.")



cols = st.columns(5)
for col, var, label in zip(
    cols,
    ["p", "b", "WE", "N", "M"],
    ["Inflação", "Salário Real", "Relação W/E", "Margem Nominal","Margem Real"]
):
    col.metric(label, f"{df[var].iloc[-1]:.5f}")

st.divider()

groups = VARIABLE_GROUPS


CHART_COLORS = ["#1F4E79", "#2E7D6B", "#B36B2C", "#6B5B95", "#A23B3B"]

def style_chart(fig, title, y_title=None, x_title="Período", height=470):
    fig.update_layout(
        template="plotly_white",
        title=dict(
            text=title,
            x=0.01,
            xanchor="left",
            y=0.98,
            yanchor="top",
            font=dict(size=17, color="#17365d")
        ),
        height=height,
        margin=dict(l=68, r=30, t=78, b=92),
        hovermode="x unified",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(size=12, color="#334155"),
        legend=dict(
            orientation="h",
            yanchor="top", y=-0.18,
            xanchor="left", x=0,
            bgcolor="rgba(255,255,255,0)",
            font=dict(size=11, color="#475569")
        )
    )
    fig.update_xaxes(
        title=x_title,
        title_font=dict(size=12, color="#475569"),
        showline=True, linewidth=1, linecolor="#9aa5af",
        ticks="outside", showgrid=False, zeroline=False,
        tickfont=dict(color="#475569")
    )
    fig.update_yaxes(
        title=y_title,
        title_font=dict(size=12, color="#475569"),
        showline=True, linewidth=1, linecolor="#9aa5af",
        ticks="outside", gridcolor="#e6ebef", gridwidth=1,
        zeroline=True, zerolinecolor="#b8c0c7", zerolinewidth=1,
        tickfont=dict(color="#475569")
    )
    return fig

def add_series(fig, df, variables):
    for i, v in enumerate(variables):
        fig.add_trace(go.Scatter(
            x=df["period"], y=df[v], mode="lines",
            name=DISPLAY_LABELS.get(v, v),
            line=dict(width=2.25, color=CHART_COLORS[i % len(CHART_COLORS)]),
            hovertemplate=DISPLAY_LABELS.get(v, v) + ": %{y:.5f}<extra></extra>"
        ))
    return fig

st.markdown("""
<div class="result-section">
  <div class="result-section-title">Relação salário–lucro</div>
  <div class="result-section-note">
    Relação teórica entre salário real <i>b</i> e margem real <i>r</i>.
  </div>
</div>
""", unsafe_allow_html=True)

r0, rf = float(df["r"].iloc[0]), float(df["r"].iloc[-1])
b0, bf = float(df["b"].iloc[0]), float(df["b"].iloc[-1])
a0, L0 = float(df["a"].iloc[0]), float(df["L"].iloc[0])
a1, L1 = float(df["a"].iloc[-1]), float(df["L"].iloc[-1])

r_intercept_0 = max((1 - a0) / a0, 0) if a0 > 0 else 1.0
r_intercept_1 = max((1 - a1) / a1, 0) if a1 > 0 else 1.0
rmax = max(r0, rf, r_intercept_0, r_intercept_1, 0.25) * 1.08

b_intercept_0 = max((1 - a0) / L0, 0) if L0 > 0 else 1.0
b_intercept_1 = max((1 - a1) / L1, 0) if L1 > 0 else 1.0
bmax = max(b0, bf, b_intercept_0, b_intercept_1, 0.25) * 1.08

rg = np.linspace(0, rmax, 300)
fig_br = go.Figure()

y0 = (1-a0)/L0 - (a0/L0)*rg
fig_br.add_trace(go.Scatter(
    x=rg, y=y0, mode="lines", name="Reta inicial",
    line=dict(width=2.5),
    hovertemplate="r = %{x:.4f}<br>b = %{y:.4f}<extra>Reta inicial</extra>"
))

if abs(a1-a0) > 1e-12 or abs(L1-L0) > 1e-12:
    y1 = (1-a1)/L1 - (a1/L1)*rg
    fig_br.add_trace(go.Scatter(
        x=rg, y=y1, mode="lines", name="Reta final",
        line=dict(width=2.5, dash="dash"),
        hovertemplate="r = %{x:.4f}<br>b = %{y:.4f}<extra>Reta final</extra>"
    ))

fig_br.add_trace(go.Scatter(
    x=[r0], y=[max(b0, 0)], mode="markers", name="Ponto inicial",
    marker=dict(size=9),
    hovertemplate="r = %{x:.4f}<br>b = %{y:.4f}<extra>Ponto inicial</extra>"
))
fig_br.add_trace(go.Scatter(
    x=[rf], y=[max(bf, 0)], mode="markers", name="Ponto final",
    marker=dict(size=10, symbol="diamond"),
    hovertemplate="r = %{x:.4f}<br>b = %{y:.4f}<extra>Ponto final</extra>"
))

fig_br.update_xaxes(range=[0, rmax], rangemode="tozero")
fig_br.update_yaxes(range=[0, bmax], rangemode="tozero")
style_chart(fig_br, "Curva salário–lucro", "Salário real (b)", "Margem real (r)", 560)

st.markdown('<div class="chart-card">', unsafe_allow_html=True)
st.plotly_chart(
    fig_br, use_container_width=True,
    config={
        "displaylogo": False,
        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        "toImageButtonOptions": {"format": "png", "filename": "simtrigo_salario_lucro", "scale": 2}
    }
)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<div class="result-section">
  <div class="result-section-title">Dinâmica temporal</div>
  <div class="result-section-note">
    Selecione as séries de interesse para visualizar sua trajetória ao longo da simulação.
  </div>
</div>
""", unsafe_allow_html=True)

for title, variables in groups.items():
    with st.expander(title, expanded=title == "Salários, câmbio e inflação"):
        selected = st.multiselect(
            "Séries apresentadas no gráfico",
            variables,
            default=variables[:min(3, len(variables))],
            format_func=lambda v: DISPLAY_LABELS.get(v, v),
            key=f"vars_{title}"
        )
        if selected:
            fig = go.Figure()
            add_series(fig, df, selected)
            style_chart(fig, title, "Valor", "Período", 455)
            fig.update_xaxes(dtick=max(1, int(T // 10)), tickformat="d")
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.plotly_chart(
                fig, use_container_width=True,
                config={
                    "displaylogo": False,
                    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                    "toImageButtonOptions": {
                        "format": "png",
                        "filename": "simtrigo_" + re.sub(r"[^a-zA-Z0-9]+", "_", title.lower()),
                        "scale": 2
                    }
                }
            )
            st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<div class="result-section">
  <div class="result-section-title">Resultados numéricos</div>
  <div class="result-section-note">Tabela completa da trajetória simulada.</div>
</div>
""", unsafe_allow_html=True)
st.dataframe(df, use_container_width=True, height=420)

st.download_button(
    "Baixar em CSV",
    df.to_csv(index=False).encode("utf-8"),
    file_name="resultados_simulacao.csv",
    mime="text/csv"
)



