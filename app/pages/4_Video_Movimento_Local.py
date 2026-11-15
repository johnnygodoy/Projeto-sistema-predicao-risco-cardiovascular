"""Página opcional local para Streamlit 1.32; sem alterações em app.py."""
from pathlib import Path
import hashlib
import json
import os
import sys

import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
from video_movimento_local import (  # noqa: E402
    MAX_BYTES, Options, VideoError, analyze_pair, local_model_path, worker_python, validate_upload, verify_model,
)
from navigation import render_navigation  # noqa: E402
from local_inputs import local_uploads_enabled  # noqa: E402


def clear():
    generation = st.session_state.get("_vid_generation", 0) + 1
    for key in list(st.session_state):
        if key.startswith("_vid_"):
            del st.session_state[key]
    st.session_state["_vid_generation"] = generation


def withdraw():
    for key in list(st.session_state):
        if key.startswith("_vid_file_") or key in ("_vid_result", "_vid_selection"):
            del st.session_state[key]
    st.session_state["_vid_generation"] = st.session_state.get("_vid_generation", 0) + 1


def main():
    st.set_page_config(page_title="Vídeo", page_icon="🎥", layout="wide")
    render_navigation()
    st.title("🎥 Vídeo")
    st.caption("Uso acadêmico. Use apenas vídeos de teste autorizados, sem pacientes ou dados clínicos.")
    st.caption("Uma referência e uma observação do mesmo movimento, alinhadas desde o início. Compara o ângulo 2D escolhido, a 5 amostras por segundo. Inferência local com modelo já baixado; não calcula risco cardiovascular.")
    st.sidebar.header("Entrada de vídeo")
    st.sidebar.button("Limpar vídeos e resultado", key="_vid_clear", on_click=clear)
    local = local_uploads_enabled(st.get_option("server.address"))
    upload_limit = st.get_option("server.maxUploadSize")
    # Limite global compartilhado com áudio (24 MiB); o vídeo continua limitado a 20 MiB.
    limited = isinstance(upload_limit, (int, float)) and 0 < upload_limit <= 24
    if not local or not limited:
        st.info("Esta página só aceita uploads na execução local limitada. Inicie com: python -m streamlit run app/app.py --server.address 127.0.0.1 --server.maxUploadSize 24 --browser.gatherUsageStats false")
        st.session_state.pop("_vid_result", None)
        return
    declared = st.sidebar.checkbox(
        "Alinhei o início dos dois vídeos para a mesma fase do movimento.",
        key="_vid_declared", on_change=withdraw,
    )
    st.sidebar.caption("MP4/H.264, 2–15 s, até 20 MiB cada, até 1280 × 720 (ou vertical), taxa constante de 5–60 fps. Uma pessoa, câmera fixa, braço visível no plano da imagem.")
    joint = st.sidebar.selectbox("Ângulo analisado", ["Cotovelo", "Ombro (abdução no plano da imagem)"], key="_vid_joint")
    side = st.sidebar.selectbox("Lado", ["Direito", "Esquerdo"], key="_vid_side")
    shoulder = joint != "Cotovelo"
    if shoulder:
        st.caption("Ombro: quadril–ombro–cotovelo do mesmo lado, relativo ao tronco. Mantenha os três pontos visíveis. É uma projeção 2D, não uma medida anatômica de abdução ou avaliação clínica.")
    else:
        st.caption("Cotovelo: ombro–cotovelo–punho do mesmo lado, com os três pontos visíveis.")
    with st.expander("Exemplo de teste e preparação dos clipes"):
        st.write("Comece com o mesmo clipe curto nos dois campos para testar a repetibilidade. Depois compare duas execuções do mesmo exercício, mesma câmera, lado e fase inicial. Um resultado sem diferenças no controle não comprova precisão do modelo.")
        st.write("Para REHAB24-6 Ex1 (elevação lateral do braço), selecione Ombro e o lado direito. Use uma fase correspondente da mesma câmera em cada clipe; durações próximas não garantem alinhamento.")
        st.info("Use MP4 de 2 a 15 s, até 720p. Não há conversão automática.")
    tolerance = st.sidebar.slider("Diferença angular para revisão (graus)", 5, 60, 20, key="_vid_tolerance")
    persistence = st.sidebar.slider("Intervalo mínimo entre amostras desviantes (s)", 0.4, 3.0, 0.6, 0.2, key="_vid_persistence")
    options = Options(side="right" if side == "Direito" else "left", tolerance_degrees=float(tolerance), min_event_seconds=float(persistence), joint="shoulder" if shoulder else "elbow")
    st.sidebar.caption("Modelo e ambiente de vídeo são configurados uma vez no servidor local. Não há download automático.")
    if not declared:
        st.session_state.pop("_vid_result", None)
        st.info("Confirme o alinhamento temporal para abrir os dois campos de envio.")
        return
    generation = st.session_state.get("_vid_generation", 0)
    reference = st.sidebar.file_uploader("Vídeo de referência", type=["mp4"], key=f"_vid_file_ref_{generation}")
    observed = st.sidebar.file_uploader("Vídeo observado", type=["mp4"], key=f"_vid_file_obs_{generation}")
    blobs = []
    try:
        for upload in (reference, observed):
            if upload is None:
                blobs.append(None)
            elif upload.size > MAX_BYTES:
                raise VideoError("Cada vídeo deve ter no máximo 20 MiB.")
            else:
                data = upload.getvalue()
                validate_upload(data)
                blobs.append(data)
    except VideoError as exc:
        st.session_state.pop("_vid_result", None)
        st.error(str(exc))
        return
    config_key = hashlib.sha256((os.environ.get("FIAP_POSE_MODEL_PATH", "") + "\0" + os.environ.get("FIAP_VIDEO_PYTHON", "")).encode()).hexdigest()
    selection = (tuple(hashlib.sha256(blob).hexdigest() if blob else None for blob in blobs), options, config_key)
    if selection != st.session_state.get("_vid_selection"):
        st.session_state.pop("_vid_result", None)
        st.session_state["_vid_selection"] = selection
    left, right = st.columns(2)
    for column, label, blob in zip((left, right), ("Referência", "Observado"), blobs):
        column.subheader(label)
        if blob:
            column.video(blob, format="video/mp4")
    model = None
    model_ready = True
    try:
        model = local_model_path()
        verify_model(model)
        worker_python()
    except VideoError as exc:
        st.info(str(exc))
        model_ready = False
        st.session_state.pop("_vid_result", None)
    st.caption("Prévia não valida formato, duração, pose ou alinhamento. A validação completa ocorre ao analisar; erros não geram conclusão.")
    run = st.sidebar.button("Analisar os dois vídeos", key="_vid_run", disabled=not all(blobs) or not model_ready)
    if run:
        st.session_state.pop("_vid_result", None)
        try:
            with st.spinner("Validando e comparando localmente (até 45 s por clipe)..."):
                st.session_state["_vid_result"] = analyze_pair(blobs[0], blobs[1], model, options)
        except VideoError as exc:
            st.error(str(exc))
    report = st.session_state.get("_vid_result")
    if not report:
        return
    st.subheader("Resultado e evidências temporais")
    st.caption(f"Ângulo analisado: {'ombro' if options.joint == 'shoulder' else 'cotovelo'} {'direito' if options.side == 'right' else 'esquerdo'}, no plano da imagem.")
    if report.get("same_input_control"):
        st.info("Controle com o mesmo arquivo nos dois campos. Testa repetibilidade do processamento; não demonstra precisão do modelo nem correção do exercício.")
    a, b, c = st.columns(3)
    a.metric("Pares válidos", report["valid_pairs"])
    b.metric("Cobertura válida", f"{report['valid_coverage']:.0%}")
    c.metric("Diferença na duração dos clipes", f"{report['duration_delta_seconds']:+.2f} s")
    if report["status"] != "analisavel":
        st.warning("Não analisável: exige ao menos 10 pares e 80% de cobertura válida. Ausência de pose, oclusão ou falta de confiança não são desvios de movimento e não significam resultado normal.")
    elif report["events"]:
        st.warning(f"{len(report['events'])} intervalo(s) amostrado(s) diferem da referência. Revise os vídeos nos tempos indicados.")
        st.dataframe(report["events"], hide_index=True, use_container_width=True)
    else:
        st.info("Nenhum intervalo ultrapassou os parâmetros desta comparação. Isso não demonstra correção do movimento ou segurança clínica.")
    st.caption("Tempos em segundos a partir do início de cada clipe. Os intervalos cobrem as amostras observadas, sem interpolar lacunas. Não há sincronização automática de fases.")
    with st.expander("Ângulos, confiança e motivo de exclusão por amostra"):
        st.dataframe(report["comparisons"], hide_index=True, use_container_width=True)
    for limitation in report["limitations"]:
        st.caption(limitation)
    st.download_button(
        "Baixar relatório JSON com evidências", json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        file_name="relatorio_video.json", mime="application/json", key="_vid_download",
    )


if __name__ == "__main__":
    main()
