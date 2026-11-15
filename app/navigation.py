"""Navegação nativa do Streamlit 1.32, sem renomear entrypoint ou rotas."""

import streamlit as st
from streamlit.errors import StreamlitAPIException


def render_navigation():
    try:
        st.sidebar.page_link("app.py", label="Predição de risco", icon="🫀")
        st.sidebar.page_link("pages/1_Monitoramento_Didatico.py", label="Monitoramento", icon="📈")
    except StreamlitAPIException:
        # Executar pages/... diretamente não registra a página inicial.
        st.sidebar.info("Abra o projeto pela página principal para habilitar a navegação.")
        st.sidebar.code("python -m streamlit run app/app.py", language="text")
