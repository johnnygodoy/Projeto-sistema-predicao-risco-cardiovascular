"""Revisão determinística de cenários e CSV acadêmico, sem uso clínico."""

from pathlib import Path
import hashlib
import sys

import pandas as pd
import streamlit as st

APP = Path(__file__).resolve().parents[1]
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from navigation import render_navigation
from local_inputs import LOCAL_COMMAND, local_uploads_enabled
from episodio_internacao_demo import (
    ALGORITHM_VERSION, Scenario, Status, build_demo, plan_rows,
    review_episode, review_rows, summarize_review, timeline_rows,
)
from episodio_internacao_csv import CSVValidationError, MAX_BYTES, SCHEMA_VERSION, parse_episode_csv

PLACEHOLDER = "Selecione um cenário"
SCENARIOS = "Cenários"
CSV_SOURCE = "Upload CSV"
RESULT_KEY = "_episode_review"
SOURCE_KEY = "_episode_scenario"


def forget_upload():
    st.session_state["_episode_upload_generation"] = st.session_state.get("_episode_upload_generation", 0) + 1
    for key in list(st.session_state):
        if key.startswith("_episode_file_") or key in (RESULT_KEY, "_episode_csv", "_episode_selected"):
            del st.session_state[key]


def clear():
    forget_upload()
    st.session_state[SOURCE_KEY] = PLACEHOLDER
    st.session_state["_episode_source"] = SCENARIOS


def sync_selection(key):
    if st.session_state.get("_episode_selected") != key:
        st.session_state.pop(RESULT_KEY, None)
        st.session_state.pop("_episode_csv", None)
        st.session_state["_episode_selected"] = key


def schema_help():
    with st.expander("Esquema e exemplos do CSV"):
        st.write(
            "Use o esquema de Internação com registros history, observation e plan. "
            "O arquivo declara um episódio, um corte as_of e uma origem; datas incluem segundos e fuso. "
            "Sem plano anterior válido, a comparação fica com dados insuficientes."
        )
        st.caption("Exemplos: Datasets/csv/internacao_alertas.csv e internacao_controle.csv.")
        st.caption("Esquema completo: Datasets/csv/INTERNACAO_SCHEMA.md. Máximo: 128 KiB, 500 registros, 64 caracteres por célula.")
        st.caption(
            "O limite é conferido após o Streamlit receber o arquivo localmente. "
            "Não exponha esta execução por túnel ou proxy público. A validação não detecta dados sensíveis."
        )


def render_episode(episode, saved, precise_times=False):
    if saved:
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
        channels = (
            (result.treatment, episode.treatment_plan, episode.treatment_current),
            (result.movement, episode.movement_plan, episode.movement_current),
        )
        for column, item, (assessment, plan, current) in zip(st.columns(2), summary.items, channels):
            plan_text, record_text = item.plan, item.record
            if precise_times:
                # CSV pode abranger dias e segundos diferentes: não ocultar essa evidência.
                stamp = lambda value: value.strftime("%d/%m/%Y %H:%M:%S UTC")
                if plan is not None:
                    plan_text = (f"{assessment.expected}. Registrado em {stamp(plan.recorded_at)}, "
                                 f"com vigência de {stamp(plan.valid_from)} a {stamp(plan.valid_until)} (fim exclusivo).")
                observed_time = stamp(current.at) if current else "horário indisponível"
                record_text = (f"Anterior: {assessment.previous}. Atual: {assessment.observed} "
                               f"({observed_time}).")
            with column:
                st.subheader(item.subject)
                st.caption(item.status.value)
                st.markdown(f"**Plano:** {plan_text}")
                st.markdown(f"**Registro:** {record_text}")
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
        if saved:
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
            "Movimentos vêm de registros tipados. Não são inferidos de vídeo, "
            "ângulo articular, áudio, câmera ou sensor; não classificam queda ou perigo."
        )


def main():
    st.set_page_config(page_title="Internação | FIAP", page_icon="🏥", layout="wide")
    render_navigation()
    st.title("Internação")
    st.caption("Prescrição e movimentação: plano, registro e pontos para conferência.")
    st.sidebar.header("Dados de entrada")
    source = st.sidebar.selectbox("Fonte", [SCENARIOS, CSV_SOURCE], key="_episode_source", on_change=forget_upload)
    st.sidebar.button("Limpar comparação", key="_episode_clear", on_click=clear, use_container_width=True)
    st.sidebar.caption("Nada é enviado a profissionais ou serviços externos. Resultado apenas neste painel.")
    local = local_uploads_enabled(st.get_option("server.address"))
    if st.session_state.get("_episode_local_allowed") != local:
        if source == CSV_SOURCE:
            forget_upload()
        st.session_state["_episode_local_allowed"] = local
    episode = None
    payload = None
    if source == SCENARIOS:
        st.info(
            "Exemplo simulado. Revisão de registros estruturados, sem uso clínico ou "
            "monitoramento contínuo. O item A não representa medicamento nem dose."
        )
        selection = st.sidebar.selectbox(
            "Cenário do episódio", [PLACEHOLDER] + [scenario.value for scenario in Scenario], key=SOURCE_KEY,
        )
        key = (SCHEMA_VERSION, ALGORITHM_VERSION, source, selection)
        sync_selection(key)
        if selection == PLACEHOLDER:
            st.info("Selecione um cenário para conferir os registros e compará-los com o plano.")
            return
        episode = build_demo(Scenario(selection))
    else:
        st.info(
            "Uso acadêmico local. Envie somente dados sintéticos ou de atividade autorizada, "
            "sem identificadores pessoais, medicamentos ou doses. Não há uso clínico nem monitoramento contínuo."
        )
        if not local:
            sync_selection(None)
            st.info("Upload CSV desativado nesta execução. Inicie localmente, sem túnel ou proxy público:")
            st.code(LOCAL_COMMAND, language="text")
            return
        generation = st.session_state.get("_episode_upload_generation", 0)
        st.sidebar.caption("CSV processado em memória, sem gravação. Até 128 KiB e 500 registros.")
        uploaded = st.sidebar.file_uploader("CSV de histórico e plano", type=["csv"], key=f"_episode_file_{generation}")
        problem = None
        if uploaded is not None:
            if not isinstance(uploaded.size, int) or not 0 < uploaded.size <= MAX_BYTES:
                problem = "Use um CSV de 1 byte a 128 KiB."
            else:
                payload = uploaded.getvalue()
                if not isinstance(payload, bytes) or not payload or len(payload) > MAX_BYTES:
                    problem = "Use um CSV de 1 byte a 128 KiB."
                    payload = None
        key = (SCHEMA_VERSION, ALGORITHM_VERSION, source, generation, hashlib.sha256(payload).hexdigest()) if payload else None
        sync_selection(key)
        if problem:
            st.error(problem)
        if payload is None:
            if not problem:
                st.info("Selecione o CSV na barra lateral para começar.")
            schema_help()
            return

    if st.sidebar.button("Comparar com o plano", key="_episode_run", type="primary", use_container_width=True):
        st.session_state.pop(RESULT_KEY, None)
        st.session_state.pop("_episode_csv", None)
        try:
            if source == CSV_SOURCE:
                parsed = parse_episode_csv(payload)
                episode = parsed.episode
                st.session_state["_episode_csv"] = (key, parsed)
            result = review_episode(episode)
            st.session_state[RESULT_KEY] = (key, result)
        except CSVValidationError as error:
            st.error(str(error))

    saved = st.session_state.get(RESULT_KEY)
    saved = saved if saved and saved[0] == key else None
    if source == CSV_SOURCE:
        parsed_state = st.session_state.get("_episode_csv")
        if not parsed_state or parsed_state[0] != key:
            st.caption("CSV recebido. Clique em Comparar com o plano para validar o arquivo e gerar o resumo.")
            schema_help()
            return
        parsed = parsed_state[1]
        episode = parsed.episode
        st.caption(f"{episode.episode_id} · {parsed.row_count} registros · origem declarada: {parsed.declared_source}. Conteúdo: {parsed.fingerprint[:12]}.")
    else:
        st.caption("Os planos A e B usam o mesmo histórico e os mesmos registros atuais; muda apenas o plano anterior.")
    st.caption(f"Corte da revisão: {episode.as_of.strftime('%d/%m/%Y às %H:%M:%S UTC')}.")
    render_episode(episode, saved, precise_times=source == CSV_SOURCE)
    if source == CSV_SOURCE:
        schema_help()


if __name__ == "__main__":
    main()
