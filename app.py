import streamlit as st

st.set_page_config(
    page_title="Simulações Econômicas",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
.main-title { font-size: 2.4rem; font-weight: 700; margin-bottom: .2rem; }
.subtitle { font-size: 1.1rem; opacity: .75; margin-bottom: 2.2rem; }
.card { border: 1px solid rgba(128,128,128,.28); border-radius: 14px; padding: 1.6rem; min-height: 210px; }
.card h3 { margin-top: 0; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Simulações Econômicas</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Modelos de simulação e análise de relações intersetoriais</div>',
    unsafe_allow_html=True,
)

if "selected_model" not in st.session_state:
    st.session_state.selected_model = None

if st.session_state.selected_model is None:
    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown('''
        <div class="card">
            <h3>Modelo de um setor</h3>
            <p>Simulação para um único setor produtivo, com configuração dos parâmetros,
            indexação, choques estruturais e progresso técnico.</p>
        </div>
        ''', unsafe_allow_html=True)
        if st.button("Abrir modelo de um setor", type="primary", use_container_width=True):
            st.session_state.selected_model = "one_sector"
            st.rerun()

    with c2:
        st.markdown('''
        <div class="card">
            <h3>Modelo de dois setores</h3>
            <p>Modelo em desenvolvimento. Esta opção ficará disponível quando a
            versão de dois setores estiver concluída.</p>
        </div>
        ''', unsafe_allow_html=True)
        if st.button("Modelo de dois setores — em desenvolvimento", use_container_width=True):
            st.session_state.selected_model = "two_sector"
            st.rerun()

elif st.session_state.selected_model == "one_sector":
    # Import only after the user selects the one-sector model, so the existing
    # dashboard remains the actual one-sector simulation application.
    import runpy
    runpy.run_path("dashboard.py", run_name="__main__")

else:
    st.markdown("## Modelo de dois setores")
    st.info("O modelo de dois setores está em desenvolvimento e ainda não está disponível para simulação.")
    if st.button("← Voltar aos modelos"):
        st.session_state.selected_model = None
        st.rerun()
