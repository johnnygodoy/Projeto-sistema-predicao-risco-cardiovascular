"""Cobertura e gráfico da amostra oficial, sem aplicar o detector sintético."""

import hashlib

import altair as alt
import pandas as pd
import streamlit as st

from bidmc_review import (
    CHECKSUM_URL, DOI_URL, EXPECTED_BYTES, EXPECTED_SHA256, FILE_URL, FILENAME,
    LICENSE_URL, MODE_LABEL, SIGNALS, SOURCE_URL, UNITS, SampleError,
    coverage_rows, signal_rows, verify_and_parse,
)
from local_inputs import LOCAL_COMMAND


def clear_research_state():
    st.session_state.pop("_bidmc_verified", None)
    st.session_state.pop("_bidmc_selected", None)


def render_bidmc(local_enabled: bool, generation: int):
    st.subheader(MODE_LABEL)
    st.caption(
        f"Fonte: [BIDMC PPG and Respiration Dataset v1.0.0]({SOURCE_URL}), "
        f"Pimentel, Johnson, Charlton e Clifton. [DOI 10.13026/C2208R]({DOI_URL}). "
        f"Contém informação disponibilizada sob a [ODC-By 1.0]({LICENSE_URL})."
    )
    st.info(
        "Uma série observada de pesquisa, não simulada: 481 instantes de 0 a 480 segundos. "
        "Aqui verificamos arquivo, cobertura e visualização. Não há rótulos de eventos "
        "para medir acurácia de anomalias. Não são executados alertas, diagnóstico ou predição."
    )
    with st.expander("Arquivo autorizado e limites desta fonte"):
        st.write(
            f"Somente [{FILENAME}]({FILE_URL}), exatamente como publicado pelo PhysioNet. "
            "Não salve a amostra dentro do repositório. O aplicativo não baixa, grava, "
            "exporta nem envia esta amostra para serviços externos. O arquivo fica na memória "
            "da sessão local. Nenhum outro arquivo clínico está autorizado neste modo."
        )
        st.caption(f"SHA-256 esperado: {EXPECTED_SHA256}. [Conferir SHA256SUMS oficial]({CHECKSUM_URL}).")
        st.write(
            "HR, SpO₂, RESP e PULSE usam unidades separadas. Há 13 ausências em PULSE "
            "e 13 em SpO₂, preservadas sem zero ou interpolação. O tempo é relativo; "
            "não há data/hora absoluta. Este arquivo não contém pressão arterial, "
            "prescrições, movimento, áudio ou vídeo."
        )
        st.caption(
            "Publicação original: Pimentel et al., Towards a Robust Estimation of Respiratory "
            "Rate from Pulse Oximeters, DOI 10.1109/TBME.2016.2613124. "
            "Infraestrutura: PhysioNet, DOI 10.1038/s44360-026-00096-z."
        )
    if not local_enabled:
        clear_research_state()
        st.info("Importação desativada nesta execução. Inicie localmente, sem túnel ou proxy público:")
        st.code(LOCAL_COMMAND, language="text")
        return

    st.sidebar.caption("Selecione a cópia oficial mantida fora do repositório. Verificação automática por nome e SHA-256.")
    upload = st.sidebar.file_uploader(
        FILENAME, type=["csv"], key=f"_mon_bidmc_file_{generation}"
    )
    payload = None
    issue = None
    if upload is not None:
        if upload.name != FILENAME:
            issue = "Selecione somente bidmc_01_Numerics.csv, sem renomear outros arquivos."
        elif upload.size != EXPECTED_BYTES:
            issue = "O tamanho não corresponde à amostra oficial. Nenhum dado foi analisado."
        else:
            payload = upload.getvalue()
    selected = hashlib.sha256(payload).hexdigest() if payload is not None else None
    if selected != st.session_state.get("_bidmc_selected"):
        clear_research_state()
        st.session_state["_bidmc_selected"] = selected
    if issue:
        clear_research_state()
        st.error(issue)
    if payload is None:
        return
    if st.sidebar.button("Verificar arquivo e visualizar", key="_mon_bidmc_validate", type="primary"):
        st.session_state.pop("_bidmc_verified", None)
        try:
            sample = verify_and_parse(upload.name, payload)
            st.session_state["_bidmc_verified"] = (selected, sample)
        except SampleError as error:
            st.error(str(error))
    saved = st.session_state.get("_bidmc_verified")
    if not saved or saved[0] != selected:
        return
    sample = saved[1]
    st.success("Arquivo oficial verificado por SHA-256. Visualização descritiva, sem avaliação clínica.")
    a, b, c = st.columns(3)
    a.metric("Instantes na amostra", len(sample.seconds))
    b.metric("Tempo relativo", "0–480 s")
    c.metric("Fonte", "BIDMC 1.0.0")
    st.subheader("Cobertura dos sinais")
    st.dataframe(pd.DataFrame(coverage_rows(sample)), hide_index=True, use_container_width=True)
    signal = st.selectbox("Sinal da amostra", SIGNALS, key="_bidmc_signal")
    frame = pd.DataFrame(signal_rows(sample, signal))
    valid = frame.dropna(subset=["Valor"])
    chart = alt.Chart(valid).mark_line(point=True).encode(
        x=alt.X("Segundo relativo:Q", title="Tempo relativo (s)", scale=alt.Scale(domain=[0, 480])),
        y=alt.Y("Valor:Q", title=f"{signal} ({UNITS[signal]})", scale=alt.Scale(zero=False)),
        detail="Trecho:N",
        tooltip=["Segundo relativo:Q", "Sinal:N", "Valor:Q", "Unidade:N"],
    )
    st.altair_chart(chart, use_container_width=True)
    absent = sum(value is None for value in sample.values(signal))
    st.caption(
        f"{signal}: {len(valid)} valores presentes e {absent} ausentes. "
        "A linha interrompe cada lacuna; não há preenchimento, suavização, referência "
        "sintética ou classificação de risco. Isso valida a leitura e a cobertura, não acurácia clínica."
    )
