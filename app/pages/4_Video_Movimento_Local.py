"""Um vídeo observado, referência local fixa e comparação deliberada no clique."""
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
    MAX_BYTES, Options, VideoError, analyze_pair, local_model_path, read_local_reference,
    worker_python, validate_upload, verify_model,
)
from navigation import render_navigation  # noqa: E402
from local_inputs import local_uploads_enabled  # noqa: E402

EXERCISE = "Elevação lateral do braço direito"
OPTIONS = Options(joint="shoulder", side="right", tolerance_degrees=20.0,
                  min_event_seconds=0.6, min_quality=0.7)


def discard_result():
    st.session_state.pop("_vid_result", None)
    st.session_state.pop("_vid_selection", None)


def main():
    st.set_page_config(page_title="Vídeo", page_icon="🎥", layout="wide")
    render_navigation()
    st.title("🎥 Vídeo")
    st.caption("Uso acadêmico. Use apenas vídeos de teste autorizados, sem pacientes ou dados clínicos.")
    st.subheader(EXERCISE)
    st.write("Envie seu vídeo e clique em Analisar vídeo. A comparação usa a referência configurada no servidor local.")
    st.info("Comece na mesma fase da referência, com câmera e enquadramento comparáveis. Uma pessoa, quadril, ombro e cotovelo direitos visíveis. O app não reconhece o exercício nem alinha fases automaticamente.")
    st.caption("MP4 de 2 a 15 s, até 720p e 20 MiB, taxa constante de 5–60 fps. Não há conversão automática.")
    local = local_uploads_enabled(st.get_option("server.address"))
    upload_limit = st.get_option("server.maxUploadSize")
    limited = isinstance(upload_limit, (int, float)) and 0 < upload_limit <= 24
    if not local or not limited:
        discard_result()
        st.info("Execute localmente: python -m streamlit run app/app.py --server.address 127.0.0.1 --server.maxUploadSize 24 --browser.gatherUsageStats false")
        return

    model, reference, setup_error = None, None, None
    try:
        model = local_model_path()
        verify_model(model)
        worker_python()
        reference = read_local_reference()
    except VideoError as exc:
        setup_error = str(exc)
        discard_result()
    if setup_error:
        st.warning("Configuração necessária antes de analisar. " + setup_error)
        st.caption("Configure uma vez, fora desta tela: o modelo local de pose, o Python separado de vídeo e a referência autorizada. A referência padrão fica em models/video-reference/referencia-ombro-direito.mp4; ela deve mostrar este exercício na mesma fase inicial que será usada nos envios.")
    else:
        st.caption("Referência local carregada. A validação completa de formato, duração e pose ocorre ao analisar.")

    upload = st.file_uploader("Vídeo observado", type=["mp4"], key="_vid_file_observed", disabled=setup_error is not None)
    data, upload_error = None, None
    try:
        if upload is not None and setup_error is None:
            if upload.size > MAX_BYTES:
                raise VideoError("O vídeo deve ter no máximo 20 MiB.")
            data = upload.getvalue()
            validate_upload(data)
    except VideoError as exc:
        upload_error = str(exc)
        discard_result()
        st.error(upload_error)

    config = "\0".join(os.environ.get(name, "") for name in (
        "FIAP_POSE_MODEL_PATH", "FIAP_VIDEO_PYTHON", "FIAP_VIDEO_REFERENCE_PATH"))
    selection = (
        hashlib.sha256(data).hexdigest() if data else None,
        hashlib.sha256(reference).hexdigest() if reference else None,
        hashlib.sha256(config.encode()).hexdigest(), OPTIONS,
    )
    if selection != st.session_state.get("_vid_selection"):
        st.session_state.pop("_vid_result", None)
        st.session_state["_vid_selection"] = selection
    ready = data is not None and reference is not None and setup_error is None and upload_error is None
    run = st.button("Analisar vídeo", key="_vid_run", disabled=not ready)
    if run and ready:
        st.session_state.pop("_vid_result", None)
        try:
            with st.spinner("Validando e comparando localmente (até 45 s por clipe)..."):
                report = analyze_pair(reference, data, model, OPTIONS)
            report["workflow"] = {"exercise": EXERCISE, "reference_source": "configured_local",
                                  "same_start_phase_required": True, "automatic_exercise_recognition": False}
            st.session_state["_vid_result"] = report
        except VideoError as exc:
            st.error(str(exc))
    report = st.session_state.get("_vid_result")
    if not report or not ready:
        return

    st.subheader("Relatório da comparação")
    st.caption("Comparação com a referência local de elevação lateral do braço direito, na mesma fase. Mede o ângulo quadril–ombro–cotovelo no plano da imagem; não é avaliação clínica.")
    if report["status"] != "analisavel":
        st.warning(f"Não foi possível comparar com confiança: {report['valid_pairs']} pares válidos e {report['valid_coverage']:.0%} de cobertura. São necessários ao menos 10 pares e 80%. Verifique se há uma só pessoa e se os pontos do braço/tronco estão visíveis. Isso não significa ausência de desvios.")
    elif report["events"]:
        st.warning(f"A comparação encontrou {len(report['events'])} intervalo(s) com diferença angular acima dos limites. Revise os momentos abaixo.")
        for event in report["events"]:
            st.write(f"De {event['start_seconds']:.2f} s a {event['end_seconds']:.2f} s: diferença máxima de {event['max_delta_degrees']:.1f}° em relação à referência.")
    else:
        st.info("Nenhum intervalo ultrapassou os limites desta comparação. Isso não comprova que o exercício está correto ou seguro.")
    if report.get("same_input_control"):
        st.caption("O arquivo enviado é idêntico à referência: este é um controle de repetibilidade, não uma validação do exercício.")
    st.caption("Tempos a partir do início do vídeo observado. A câmera, o ritmo e a mesma fase influenciam a comparação.")
    st.video(data, format="video/mp4")  # Somente após decodificação/validação completa dos dois clipes.
    with st.expander("Detalhes e evidências"):
        st.write(f"Pares válidos: {report['valid_pairs']}. Cobertura: {report['valid_coverage']:.0%}. Diferença de duração: {report['duration_delta_seconds']:+.2f} s.")
        st.caption("Parâmetros fixos: ombro direito, diferença >20°, intervalo amostrado ≥0,6 s, qualidade mínima 0,7 e 5 amostras/s. Sem sincronização automática ou interpolação de lacunas.")
        st.dataframe(report["comparisons"], hide_index=True, use_container_width=True)
        for limitation in report["limitations"]:
            st.caption(limitation)
    st.download_button("Baixar relatório JSON", json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
                       file_name="relatorio_video.json", mime="application/json", key="_vid_download")


if __name__ == "__main__":
    main()
