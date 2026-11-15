"""Revisão determinística de registros; aplicação acadêmica, sem dados reais."""

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

APP = Path(__file__).resolve().parents[1]
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from navigation import render_navigation
from episodio_internacao_demo import (
    ALGORITHM_VERSION, Scenario, Status, build_demo, plan_rows,
    review_episode, review_rows, summarize_review, timeline_rows,
)

st.set_page_config(page_title="Internação | FIAP", page_icon="🏥", layout="wide")
render_navigation()
st.title("Internação")
st.caption("Prescrição e movimentação: plano, registro e pontos para conferência.")
st.info(
    "Exemplo simulado. Revisão de registros estruturados, sem uso clínico ou "
    "monitoramento contínuo. O item A não representa medicamento nem dose."
)

PLACEHOLDER = "Selecione um cenário"
RESULT_KEY = "_episode_review"
SOURCE_KEY = "_episode_scenario"


def clear():
    st.session_state.pop(RESULT_KEY, None)
    st.session_state[SOURCE_KEY] = PLACEHOLDER


st.sidebar.header("Comparação manual")
selection = st.sidebar.selectbox(
    "Cenário do episódio", [PLACEHOLDER] + [scenario.value for scenario in Scenario],
    key=SOURCE_KEY,
)
selection_key = (ALGORITHM_VERSION, selection)
if st.session_state.get("_episode_selected") != selection_key:
    st.session_state.pop(RESULT_KEY, None)
    st.session_state["_episode_selected"] = selection_key
st.sidebar.button("Limpar comparação", key="_episode_clear", on_click=clear, use_container_width=True)
st.sidebar.caption("Nada é enviado a profissionais ou serviços externos. Resultado apenas neste painel.")

if selection == PLACEHOLDER:
    st.info("Selecione um cenário para conferir os registros e compará-los com o plano.")
    st.stop()

scenario = Scenario(selection)
episode = build_demo(scenario)
st.caption(
    "Episódio 01 · 15/01/2026 às 10:00 UTC. Os planos A e B usam o mesmo histórico "
    "e os mesmos registros atuais; muda apenas o plano registrado antes da observação."
)

if st.sidebar.button("Comparar com o plano", key="_episode_run", type="primary", use_container_width=True):
    st.session_state[RESULT_KEY] = (selection_key, review_episode(episode))

saved = st.session_state.get(RESULT_KEY)
if saved and saved[0] == selection_key:
    result = saved[1]
    summary = summarize_review(episode, result)
    st.subheader("Resumo da revisão")
    message = f"{summary.status.value}. {summary.text}"
    if summary.status is Status.REVIEW:
        st.warning(message)
    else:
        st.info(message)
    columns = st.columns(3)
    columns[0].metric("Alertas", result.alert_count)
    columns[1].metric("Sem divergências", sum(item.status is Status.ALIGNED for item in result.assessments))
    columns[2].metric("Dados insuficientes", result.unavailable_count)
    for column, item in zip(st.columns(2), summary.items):
        with column:
            st.subheader(item.subject)
            st.caption(item.status.value)
            st.markdown(f"**Plano:** {item.plan}")
            st.markdown(f"**Registro:** {item.record}")
            st.markdown(f"**Diferença:** {item.difference}")
            st.markdown(f"**Próxima conferência:** {item.next_action}")
            st.caption(item.provenance)
    st.caption(
        "Sem divergências não significa normalidade clínica. O resultado indica somente "
        "a correspondência com o plano informado; dados insuficientes não são classificados como compatíveis."
    )
else:
    st.caption("Confira a origem dos dados abaixo e use Comparar com o plano para gerar o resumo.")

with st.expander("Histórico e plano registrados", expanded=not bool(saved)):
    st.subheader("Histórico e observação")
    st.dataframe(pd.DataFrame(timeline_rows(episode)), hide_index=True, use_container_width=True)
    st.subheader("Plano prospectivo")
    st.dataframe(pd.DataFrame(plan_rows(episode)), hide_index=True, use_container_width=True)
    st.caption("O início da vigência é inclusivo e o fim é exclusivo. Ausência de plano impede avaliar divergências.")
    if saved and saved[0] == selection_key:
        st.subheader("Rastreabilidade da comparação")
        st.dataframe(pd.DataFrame(review_rows(saved[1])), hide_index=True, use_container_width=True)

with st.expander("Regras e limites"):
    st.write(
        "C01: observado igual ao planejado. C02: observado diferente do planejado, "
        "com dados e plano válidos. Somente C02 gera alerta para conferência dos registros."
    )
    st.write(
        "Dados ausentes, sem histórico válido, fora de ordem ou sem plano prospectivo válido "
        "resultam em dados insuficientes. O plano precisa anteceder a observação e cobrir seu horário."
    )
    st.write(
        "A janela de atualidade de 15 minutos é uma convenção de teste, não um limiar clínico. "
        "As frases são determinísticas: não há chatbot e nenhuma regra prescreve ou altera tratamento."
    )
    st.caption(
        "Movimentos vêm de uma sequência tipada e fixa. Não são inferidos de vídeo, "
        "ângulo articular, áudio, câmera ou sensor; não classificam queda ou perigo."
    )
