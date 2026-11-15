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

from monitoramento_demo import (  # noqa: E402
    ALGORITHM_VERSION, MAX_BYTES, MIN_HISTORY, REVIEW_SCORE, SIGNALS, WINDOW,
    ValidationError, analyze, chart_rows, comparison_rows, parse_csv, preview_rows,
    selection_key, sync_selection,
)


SOURCES = (
    "Escolha uma fonte fictícia",
    "Exemplo fictício com variações",
    "Controle fictício sem variações injetadas",
    "Enviar CSV fictício",
)
FIXTURES = {
    SOURCES[1]: "variacoes_ficticias.csv",
    SOURCES[2]: "controle_ficticio.csv",
}


def clear_demo():
    generation = st.session_state.get("_mon_upload_generation", 0) + 1
    for key in list(st.session_state):
        if key.startswith("_mon_"):
            del st.session_state[key]
    st.session_state["_mon_source"] = SOURCES[0]
    st.session_state["_mon_declared"] = False
    st.session_state["_mon_upload_generation"] = generation


def forget_upload():
    # Rotacionar a identidade do widget evita restaurar arquivo do frontend.
    for key in list(st.session_state):
        if key.startswith("_mon_file_"):
            del st.session_state[key]
    st.session_state["_mon_upload_generation"] = st.session_state.get("_mon_upload_generation", 0) + 1
    sync_selection(st.session_state, None)


def render_results(results):
    scored = sum(r.score is not None for r in results)
    alerts = tuple(r for r in results if r.review)
    st.subheader("3. Análise dos sinais")
    a, b, c = st.columns(3)
    a.metric("Medições comparadas", scored)
    b.metric("Variações para revisão", len(alerts))
    c.metric("Sem histórico suficiente", len(results) - scored)
    if not scored:
        st.info("Nenhuma comparação executada: envie mais medições fictícias em sequência.")
    elif alerts:
        st.warning(
            "Há variações estatísticas para revisão manual dos dados fictícios. "
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
        domain=["Medição fictícia", "Mediana anterior", "Revisar variação"],
        range=["#4DA6FF", "#f59e0b", "#ef4444"],
    ))
    x = alt.X("Instante:T", title="Instante (UTC)", scale=alt.Scale(type="utc"))
    y = alt.Y("Valor:Q", title=f"{SIGNALS[signal].label} ({SIGNALS[signal].unit})",
              scale=alt.Scale(zero=False))
    tooltip = [alt.Tooltip("Instante UTC:N", title="Instante (UTC)"),
               alt.Tooltip("Valor:Q"), alt.Tooltip("Unidade:N"),
               alt.Tooltip("Mediana anterior:Q"), alt.Tooltip("Score:Q", format=".2f")]
    measured = alt.Chart(frame.assign(Série="Medição fictícia")).mark_line(point=True).encode(
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
            "Parâmetros fixos e didáticos, não calibrados clinicamente. Uma série "
            "constantemente alterada pode não gerar alerta; a referência também se "
            "adapta a mudanças persistentes. Não é um modelo treinado nem diagnóstico."
        )


def main():
    st.set_page_config(page_title="Monitoramento didático", page_icon="📈", layout="wide")
    st.title("📈 Monitoramento didático de sinais vitais")
    st.warning(
        "Protótipo acadêmico, somente com dados fictícios. Não use dados de pacientes, "
        "nomes, documentos ou outros identificadores. Sem diagnóstico, previsão clínica "
        "ou recomendação de tratamento. A predição original permanece na página inicial."
    )
    st.caption(
        "Comparação estatística determinística de uma única série simulada. "
        "Esta página não carrega o modelo cardiovascular e não envia dados a uma IA externa."
    )
    results_container = st.container()
    st.sidebar.header("📋 Dados fictícios")
    st.sidebar.button("Limpar seleção e resultados", on_click=clear_demo, key="_mon_clear")

    with st.expander("Formato do CSV e limites de qualidade"):
        st.code("timestamp,signal,value,unit\n2026-01-01T12:00:00Z,heart_rate,72,bpm", language="text")
        st.write(
            "UTF-8, separado por vírgula; até 256 KiB e 2.000 medições. "
            "Somente as quatro colunas acima, nessa ordem. Valor com ponto decimal "
            "e até 3 casas. Instantes com segundos e fuso explícito, em ordem UTC. "
            "Sem células vazias, duplicatas do mesmo sinal/instante ou colunas pessoais. "
            "Qualquer erro rejeita o arquivo inteiro; não há conversão de unidade, "
            "ordenação automática, interpolação ou preenchimento de valores."
        )
        st.dataframe(pd.DataFrame([
            {"signal": name, "Sinal": spec.label, "unit": spec.unit,
             "Mínimo aceito": spec.minimum, "Máximo aceito": spec.maximum,
             "Piso da escala": spec.scale_floor}
            for name, spec in SIGNALS.items()
        ]), hide_index=True, use_container_width=True)
        st.caption("Faixas de entrada são limites técnicos do exemplo, não faixas clínicas de normalidade.")

    st.sidebar.subheader("1. Escolher a fonte")
    source = st.sidebar.selectbox("Fonte", SOURCES, key="_mon_source")
    payload = None
    problem = None
    if source in FIXTURES:
        try:
            payload = (APP_DIR / "monitoramento_exemplos" / FIXTURES[source]).read_bytes()
        except OSError:
            problem = "Exemplo não encontrado. Verifique se o pacote foi aplicado por completo."
        if payload is not None:
            st.sidebar.download_button("Baixar este CSV fictício", payload, FIXTURES[source], "text/csv")
    elif source == SOURCES[3]:
        st.sidebar.info(
            "Antes de selecionar o arquivo: o upload transfere seu conteúdo ao servidor "
            "que executa o Streamlit. Este protótipo aceita somente conteúdo totalmente "
            "inventado e não verifica sozinho se os dados são fictícios. Ele processa em "
            "memória, sem persistência ou cache de dados no código desta página. "
            "O limite de 256 KiB é validado após a transferência pelo Streamlit."
        )
        declared = st.sidebar.checkbox(
            "Declaro que o CSV contém apenas dados fictícios, sem pessoas reais ou identificadores.",
            key="_mon_declared", on_change=forget_upload,
        )
        if declared:
            # Fora de st.form para que trocar/remover arquivo provoque rerun imediato
            # e esconda o resultado anterior. Este é o único uploader da página.
            generation = st.session_state.get("_mon_upload_generation", 0)
            upload = st.sidebar.file_uploader(
                "CSV fictício (até 256 KiB)", type=["csv"], key=f"_mon_file_{generation}"
            )
            if upload is not None:
                if upload.size > MAX_BYTES:
                    problem = "Arquivo excede 256 KiB. Escolha um CSV fictício menor."
                else:
                    payload = upload.getvalue()

    key = selection_key(source, payload) if payload is not None else None
    sync_selection(st.session_state, key)
    if problem:
        st.error(problem)
    if payload is None:
        st.info("Escolha um exemplo ou declare a origem fictícia antes de selecionar um CSV.")
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
    st.subheader("2. Prévia e qualidade")
    st.caption(f"Seleção: {source}. Identificação do conteúdo: {dataset.fingerprint[:12]}.")
    st.write(f"{len(dataset.readings)} medições validadas; horário convertido para UTC.")
    st.dataframe(pd.DataFrame(preview_rows(dataset)).head(100), hide_index=True, use_container_width=True)
    if len(dataset.readings) > 100:
        st.caption("Prévia limitada às primeiras 100 medições; a análise usa todas as medições validadas.")
    for notice in dataset.notices:
        st.warning(notice)
    st.caption("Cada sinal precisa de 5 medições anteriores. Revise a prévia e execute a análise pela barra lateral.")
    st.sidebar.caption("Prévia disponível na área principal. Confira os dados antes de executar.")
    if st.sidebar.button("Executar análise dos sinais", key="_mon_run", type="primary", use_container_width=True):
        st.session_state["_mon_results"] = (key, ALGORITHM_VERSION, analyze(dataset))
    result = st.session_state.get("_mon_results")
    if result and result[:2] == (key, ALGORITHM_VERSION):
        with results_container:
            render_results(result[2])


if __name__ == "__main__":
    main()
