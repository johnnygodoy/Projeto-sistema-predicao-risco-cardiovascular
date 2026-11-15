import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import os
from navigation import render_navigation

import warnings
warnings.filterwarnings("ignore")

# ===============================
# CONFIG (TEM QUE SER PRIMEIRO)
# ===============================
st.set_page_config(
    page_title="Predição de Risco Cardiovascular com IA",
    layout="wide",
    page_icon="🫀"
)
render_navigation()
# ===============================
# LOAD CSS
# ===============================
def load_css():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    css_path = os.path.join(base_dir, "assets", "style.css")

    if os.path.exists(css_path):
        with open(css_path, encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# ===============================
# LOAD MODEL
# ===============================
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(base_dir, "model", "model.pkl")

    with open(model_path, "rb") as f:
        return pickle.load(f)

model = load_model()

# ===============================
# HEADER
# ===============================
st.markdown("""
<div class="header-container">
    <h1>🫀 Sistema Inteligente de Predição de Risco Cardiovascular com Inteligência Artificial</h1>
    <p class="subtitulo">Modelo de Machine Learning para apoio à decisão clínica</p>
</div>
""", unsafe_allow_html=True)

# ===============================
# SIDEBAR
# ===============================
st.sidebar.header("📋 Dados do Paciente")

age = st.sidebar.slider("Idade", 18, 100, 45)
height = st.sidebar.slider("Altura (cm)", 140, 210, 170)
weight = st.sidebar.slider("Peso (kg)", 40, 150, 70)

ap_hi = st.sidebar.slider("Pressão Sistólica", 80, 240, 120)
ap_lo = st.sidebar.slider("Pressão Diastólica", 50, 140, 80)

colesterol_label = st.sidebar.selectbox(
    "Colesterol",
    ["Normal", "Acima do normal", "Muito alto"]
)
colesterol = {"Normal":1,"Acima do normal":2,"Muito alto":3}[colesterol_label]

glicose_label = st.sidebar.selectbox(
    "Glicose",
    ["Normal", "Acima do normal", "Muito alto"]
)
gluc = {"Normal":1,"Acima do normal":2,"Muito alto":3}[glicose_label]

fumante_label = st.sidebar.selectbox("Fumante", ["Não", "Sim"])
smoke = 1 if fumante_label == "Sim" else 0

alcool_label = st.sidebar.selectbox(
    "Consumo de Álcool",
    ["Não bebe", "Social", "Frequente"]
)
alco = {"Não bebe":0,"Social":1,"Frequente":1}[alcool_label]

atividade_label = st.sidebar.selectbox("Atividade Física", ["Não", "Sim"])
active = 1 if atividade_label == "Sim" else 0

# ===============================
# FEATURE ENGINEERING
# ===============================
bmi = weight / ((height / 100) ** 2)
pressure_age = ap_hi * age

# ===============================
# INPUT DATA (CORRETO)
# ===============================
input_data = pd.DataFrame([{
    "ap_hi": ap_hi,
    "ap_lo": ap_lo,
    "age_years": age,
    "bmi": bmi,
    "cholesterol": colesterol,
    "gluc": gluc,
    "smoke": smoke,
    "alco": alco,
    "active": active,
    "pressure_age": pressure_age
}])

# ===============================
# FUNÇÃO DE EXPLICAÇÃO
# ===============================
def gerar_explicacao():
    fatores = []

    if ap_hi > 140:
        fatores.append("pressão arterial elevada")

    if colesterol > 1:
        fatores.append("colesterol acima do normal")

    if gluc > 1:
        fatores.append("glicose alterada")

    if bmi > 30:
        fatores.append("IMC elevado")

    if age > 55:
        fatores.append("idade elevada")

    if smoke == 1:
        fatores.append("tabagismo")

    if active == 0:
        fatores.append("baixo nível de atividade física")

    return fatores

if st.sidebar.button("🔍 Analisar Risco"):

    prob = model.predict_proba(input_data)[0][1]

    # ===============================
    # CLASSIFICAÇÃO
    # ===============================
    if prob < 0.3:
        risk_level = "BAIXO RISCO"
        color = "#16a34a"
    elif prob < 0.6:
        risk_level = "MÉDIO RISCO"
        color = "#f59e0b"
    else:
        risk_level = "ALTO RISCO"
        color = "#dc2626"

    fatores = gerar_explicacao()

    if fatores:
        explicacao = "O risco é influenciado por: " + ", ".join(fatores) + "."
    else:
        explicacao = "Nenhum fator crítico identificado."

    st.markdown("---")

    # ===============================
    # SCORE CENTRAL
    # ===============================
    col1, col2, col3 = st.columns([1,2,1])

    with col2:
        st.metric("Probabilidade de Risco", f"{prob:.2%}")
        st.markdown(
            f"<h2 style='text-align:center; color:{color}'>{risk_level}</h2>",
            unsafe_allow_html=True
        )

    # ===============================
    # DASHBOARD PRINCIPAL
    # ===============================
    st.markdown("")

    col_left, col_right = st.columns([1.2, 2.2])

    # ===============================
    # 🔴 RISCO (AGORA COM CONTEXTO)
    # ===============================
    with col_left:
        st.subheader("📊 Nível de Risco")

        fig, ax = plt.subplots(figsize=(8, 3.5))

        ax.barh(["Risco"], [prob], color=color)

        ax.set_xlim(0,1)
        ax.set_xlabel("Probabilidade")

        ax.set_title("Escala de Risco", fontsize=15)
        ax.tick_params(axis='x', labelsize=11)

        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

        plt.tight_layout()

        st.pyplot(fig, use_container_width=True)

    # ===============================
    # 📊 FATORES
    # ===============================
    with col_right:
        st.subheader("📈 Principais Fatores")

        factors = {
            "Pressão": ap_hi,
            "Idade": age,
            "BMI": round(bmi,1),
            "Colesterol": colesterol,
        }

        df_factors = pd.DataFrame(list(factors.items()), columns=["Fator","Valor"])

        fig2, ax2 = plt.subplots(figsize=(6, 3))

        ax2.bar(df_factors["Fator"], df_factors["Valor"], color="#4DA6FF")

        for spine in ["top", "right", "left"]:
            ax2.spines[spine].set_visible(False)

        ax2.set_yticks([])

        plt.tight_layout()

        st.pyplot(fig2, use_container_width=True)

    # ===============================
    # 🧠 ANÁLISE (AGORA À ESQUERDA)
    # ===============================
    st.markdown("---")

    colA, colB = st.columns([1,2])

    with colA:
        st.subheader("🧠 Análise do Modelo")

        if prob > 0.6:
            st.error(explicacao)
        elif prob > 0.3:
            st.warning(explicacao)
        else:
            st.success(explicacao)

    with colB:
        st.subheader("📋 Interpretação")

        st.markdown(f"""
        - **Pressão:** {ap_hi} mmHg  
        - **Idade:** {age} anos  
        - **IMC:** {round(bmi,1)}  
        - **Colesterol:** {colesterol_label}  
        - **Glicose:** {glicose_label}  
        """)

    # ===============================
    # ⚠️ AVISO
    # ===============================
    st.markdown("---")
    st.warning("Sistema de apoio. Não substitui diagnóstico médico.")