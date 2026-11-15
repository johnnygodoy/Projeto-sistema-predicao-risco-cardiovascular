"""Página independente, compatível com Streamlit 1.32 (diretório pages)."""

from pathlib import Path
import sys

import altair as alt
import pandas as pd
import streamlit as st

# Também permite AppTest.from_file da página, sem depender do diretório atual.
APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from navigation import render_navigation
from local_inputs import LOCAL_COMMAND, local_uploads_enabled
from bidmc_review_ui import clear_research_state, render_bidmc

from monitoramento_demo import (  # noqa: E402
    ALGORITHM_VERSION, MAX_BYTES, MIN_HISTORY, REVIEW_SCORE, SIGNALS, WINDOW,
    ValidationError, analyze, chart_rows, comparison_rows, parse_csv, preview_rows,
    selection_key, sync_selection,
)


SOURCES = (
    "Escolha uma fonte",
    "Exemplo com variações",
    "Exemplo",
    "Importar CSV de teste",
    "BIDMC · PhysioNet",
)
FIXTURES = {
    SOURCES[1]: "variacoes_ficticias.csv",
    SOURCES[2]: "controle_ficticio.csv",
}


def clear_demo():
    generation = st.session_state.get("_mon_upload_generation", 0) + 1
    for key in list(st.session_state):
        if key.startswith(("_mon_", "_bidmc_")):
            del st.session_state[key]
    st.session_state["_mon_source"] = SOURCES[0]
    st.session_state["_mon_upload_generation"] = generation


def forget_upload():
    # Rotacionar a identidade do widget evita restaurar arquivo do frontend.
    for key in list(st.session_state):
        if key.startswith(("_mon_file_", "_mon_bidmc_file_")):
            del st.session_state[key]
    st.session_state["_mon_upload_generation"] = st.session_state.get("_mon_upload_generation", 0) + 1
    sync_selection(st.session_state, None)
    clear_research_state()


def render_results(results):
    scored = sum(r.score is not None for r in results)
    alerts = tuple(r for r in results if r.review)
    st.subheader("Análise dos sinais")
    a, b, c = st.columns(3)
    a.metric("Medições comparadas", scored)
    b.metric("Variações para revisão", len(alerts))
    c.metric("Sem histórico suficiente", len(results) - scored)
    if not scored:
        st.info("Nenhuma comparação executada: envie mais medições em sequência.")
    elif alerts:
        st.warning(
            "Há variações estatísticas para revisão manual dos dados. "
            "Isso não é um alerta clínico nem uma previsão de evento."
        )
        st.dataframe(pd.DataFrame(comparison_rows(alerts)), hide_index=True, use_container_width=True)
    else:
        st.info("Sem alertas estatísticos neste exemplo. Isso não demonstra segurança clínica.")

    choices = [signal for signal in SIGNALS if any(r.reading.signal == signal for r in results)]
    labels = {f"{SIGNALS[value].label} ({SIGNALS[value].unit})": value for value in choices}
    selected_label = st.selectbox(
        "Sinal para visualizar (uma unidade por gráfico)", list(labels),
        key="_mon_chart_signal",
    )
    signal = labels[selected_label]
    frame = pd.DataFrame(chart_rows(results, signal))
    legend = alt.Color("Série:N", title=None, scale=alt.Scale(
        domain=["Medição", "Mediana anterior", "Revisar variação"],
        range=["#4DA6FF", "#f59e0b", "#ef4444"],
    ))
    x = alt.X("Instante:T", title="Instante (UTC)", scale=alt.Scale(type="utc"))
    y = alt.Y("Valor:Q", title=f"{SIGNALS[signal].label} ({SIGNALS[signal].unit})",
              scale=alt.Scale(zero=False))
    tooltip = [alt.Tooltip("Instante UTC:N", title="Instante (UTC)"),
               alt.Tooltip("Valor:Q"), alt.Tooltip("Unidade:N"),
               alt.Tooltip("Mediana anterior:Q"), alt.Tooltip("Score:Q", format=".2f")]
    measured = alt.Chart(frame.assign(Série="Medição")).mark_line(point=True).encode(
        x=x, y=y, color=legend, detail="Segmento:N", tooltip=tooltip,
    )
    reference = frame.dropna(subset=["Mediana anterior"]).copy()
    reference["Valor"] = reference["Mediana anterior"]
    baseline = alt.Chart(reference.assign(Série="Mediana anterior")).mark_line(strokeDash=[6, 4], point=True).encode(
        x=x, y=y, color=legend, detail="Segmento:N",
        tooltip=[alt.Tooltip("Instante UTC:N"), alt.Tooltip("Valor:Q", title="Mediana anterior"), "Unidade:N"],
    )
    flagged = alt.Chart(frame[frame["Revisar"]].assign(Série="Revisar variação")).mark_point(
        filled=True, shape="diamond", size=110,
    ).encode(x=x, y=y, color=legend, tooltip=tooltip)
    st.altair_chart(measured + baseline + flagged, use_container_width=True)
    st.caption(
        "Azul: medições do CSV; tracejado: referência histórica; losangos vermelhos: "
        "variações para revisão. Sem referência nos pontos iniciais; lacunas >6h "
        "interrompem as linhas. Nenhum valor é preenchido com zero ou interpolado no cálculo. "
        "A cadência de coleta não é presumida."
    )
    with st.expander("Ver todas as comparações e explicar o cálculo"):
        st.dataframe(pd.DataFrame(comparison_rows(results)), hide_index=True, use_container_width=True)
        st.write(
            f"Para cada sinal: mediana das até {WINDOW} medições anteriores, exigindo "
            f"pelo menos {MIN_HISTORY}. Escala = máximo entre 1,4826 × MAD e o piso "
            f"da tabela. Score = |valor − mediana| / escala. Revisar quando score > {REVIEW_SCORE}. "
            "MAD é a mediana dos desvios absolutos em relação à mediana anterior. "
            "A medição atual só entra no histórico depois de ser comparada. "
            "Intervalos maiores que 6 horas reiniciam o histórico."
        )
        st.caption(
            "Parâmetros fixos de demonstração, não calibrados clinicamente. Uma série "
            "constantemente alterada pode não gerar alerta; a referência também se "
            "adapta a mudanças persistentes. Não é um modelo treinado nem diagnóstico."
        )


def main():
    st.set_page_config(page_title="Monitoramento", page_icon="📈", layout="wide")
    render_navigation()
    st.title("📈 Monitoramento")
    st.caption("Uso acadêmico. Não use arquivos com dados identificáveis.")
    st.sidebar.header("Dados de entrada")
    source = st.sidebar.selectbox("Fonte", SOURCES, key="_mon_source", on_change=forget_upload)
    st.sidebar.button("Limpar seleção e resultados", on_click=clear_demo, key="_mon_clear")
    local_enabled = local_uploads_enabled(st.get_option("server.address"))
    generation = st.session_state.get("_mon_upload_generation", 0)

    if source == SOURCES[4]:
        sync_selection(st.session_state, None)
        render_bidmc(local_enabled, generation)
        return
    clear_research_state()
    st.caption(
        "Comparação estatística com o histórico de cada sinal, sem diagnóstico ou "
        "recomendação de tratamento. Este modo não executa o modelo cardiovascular."
    )
    results_container = st.container()
    with st.expander("Formato do CSV e critérios de validação"):
        st.code("timestamp,signal,value,unit\n2026-01-01T12:00:00Z,heart_rate,72,bpm", language="text")
        st.write(
            "CSV UTF-8 separado por vírgula; até 256 KiB e 2.000 medições. "
            "Somente estas quatro colunas, nesta ordem. Valor com ponto decimal "
            "e até 3 casas; instantes com segundos e fuso, em ordem UTC. "
            "Sem células vazias, duplicatas ou colunas adicionais. Um erro rejeita "
            "o arquivo inteiro. Não há conversão de unidade, ordenação automática "
            "ou preenchimento de valores. A validação não identifica a origem "
            "sensível dos dados. Importação disponível apenas na execução local."
        )
        st.dataframe(pd.DataFrame([
            {"signal": name, "Sinal": spec.label, "unit": spec.unit,
             "Mínimo aceito": spec.minimum, "Máximo aceito": spec.maximum,
             "Piso da escala": spec.scale_floor}
            for name, spec in SIGNALS.items()
        ]), hide_index=True, use_container_width=True)
        st.caption("Limites técnicos de entrada e parâmetros acadêmicos, não faixas de normalidade clínica.")

    payload = None
    problem = None
    if source in FIXTURES:
        try:
            payload = (APP_DIR / "monitoramento_exemplos" / FIXTURES[source]).read_bytes()
        except OSError:
            problem = "Exemplo não encontrado. Verifique se a atualização foi aplicada por completo."
        if payload is not None:
            filename = "exemplo-com-variacoes.csv" if source == SOURCES[1] else "exemplo.csv"
            st.sidebar.download_button("Baixar exemplo", payload, filename, "text/csv")
    elif source == SOURCES[3]:
        if not local_enabled:
            sync_selection(st.session_state, None)
            st.info("Importação desativada nesta execução. Inicie localmente, sem túnel ou proxy público:")
            st.code(LOCAL_COMMAND, language="text")
            return
        st.sidebar.caption(
            "CSV de teste processado em memória, sem gravação ou envio a serviços externos. "
            "O Streamlit recebe o arquivo antes da validação de 256 KiB."
        )
        upload = st.sidebar.file_uploader(
            "Importar CSV de teste (até 256 KiB)", type=["csv"], key=f"_mon_file_{generation}"
        )
        if upload is not None:
            if upload.size > MAX_BYTES:
                problem = "Arquivo excede 256 KiB. Escolha um CSV de teste menor."
            else:
                payload = upload.getvalue()

    key = selection_key(source, payload) if payload is not None else None
    sync_selection(st.session_state, key)
    if problem:
        st.error(problem)
    if payload is None:
        st.info("Selecione uma fonte na barra lateral para começar.")
        return
    if st.sidebar.button("Validar e visualizar", key="_mon_validate", use_container_width=True):
        st.session_state.pop("_mon_preview", None)
        st.session_state.pop("_mon_results", None)
        try:
            st.session_state["_mon_preview"] = (key, parse_csv(payload))
        except ValidationError as error:
            st.error(str(error))

    preview = st.session_state.get("_mon_preview")
    if not preview or preview[0] != key:
        return
    dataset = preview[1]
    st.subheader("Prévia e qualidade")
    st.caption(f"Fonte: {source}. Identificação do conteúdo: {dataset.fingerprint[:12]}.")
    st.write(f"{len(dataset.readings)} medições validadas; horário em UTC.")
    st.dataframe(pd.DataFrame(preview_rows(dataset)).head(100), hide_index=True, use_container_width=True)
    if len(dataset.readings) > 100:
        st.caption("Prévia limitada às primeiras 100 medições; a análise usa todas as medições validadas.")
    for notice in dataset.notices:
        st.warning(notice)
    st.sidebar.caption("Confira a prévia na área principal antes de executar.")
    if st.sidebar.button("Executar análise dos sinais", key="_mon_run", type="primary", use_container_width=True):
        st.session_state["_mon_results"] = (key, ALGORITHM_VERSION, analyze(dataset))
    result = st.session_state.get("_mon_results")
    if result and result[:2] == (key, ALGORITHM_VERSION):
        with results_container:
            render_results(result[2])


if __name__ == "__main__":
    main()
