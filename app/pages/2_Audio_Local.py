"""Optional academic local audio page; Streamlit 1.32 native widgets."""

import os
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from navigation import render_navigation
from local_inputs import LOCAL_COMMAND, local_uploads_enabled
from audio_local import (
    AudioError, MAX_BYTES, selection_key, sync_selection, timestamp,
    transcribe_local, validate_wav,
)
from audio_text import sentiment_summary, term_rows


def clear_audio():
    generation = st.session_state.get("_audio_upload_generation", 0) + 1
    # Keep local model/idiom controls; discard upload and all derived results.
    for name in list(st.session_state):
        if name.startswith("_audio_") and name not in {"_audio_model", "_audio_language"}:
            del st.session_state[name]
    st.session_state["_audio_upload_generation"] = generation


def render_results(result):
    st.subheader("Transcrição local")
    st.caption(f"Motor: {result.engine}. Idioma selecionado: {result.language}. Tempos aproximados.")
    st.caption("Reconhecimento automático pode errar, omitir palavras ou gerar texto inexistente. Confira ouvindo o arquivo.")
    if not result.segments:
        st.info("O modelo não retornou segmentos. Isso não comprova ausência de fala.")
        return
    st.dataframe(pd.DataFrame([
        {"Início": timestamp(segment.start), "Fim": timestamp(segment.end), "Texto": segment.text}
        for segment in result.segments
    ]), hide_index=True, use_container_width=True)
    st.subheader("Termos no texto")
    rows = term_rows(result.segments)
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    else:
        st.info("Nenhum termo do pequeno vocabulário foi encontrado. Isso não indica ausência de sintomas.")
    st.caption(
        "Busca literal em seis grupos de termos. Negação e incerteza usam até cinco palavras "
        "anteriores na mesma oração e segmento. São pistas textuais, não sintomas confirmados "
        "ou negados. Erros de transcrição, pontuação, terceiros e contexto podem mudar a leitura."
    )
    with st.expander("Exemplo opcional: polaridade lexical do texto"):
        data = sentiment_summary(result.text)
        st.write("Palavras positivas encontradas: " + (", ".join(data["positive"]) or "nenhuma"))
        st.write("Palavras negativas encontradas: " + (", ".join(data["negative"]) or "nenhuma"))
        st.write("Excluídas por pista próxima de negação/incerteza: " + (", ".join(data["excluded"]) or "nenhuma"))
        if data["score"] is None:
            st.info("Sem evidência lexical suficiente para calcular o índice.")
        else:
            st.write(f"Índice lexical: {data['score']:.2f}, calculado por (positivas − negativas) / total encontrado.")
        st.caption(
            "Contagem didática de palavras, sem modelo de sentimento validado. Não mede emoção, "
            "estado mental, tom de voz ou risco. Não interpreta ironia ou contexto."
        )


def main():
    st.set_page_config(page_title="Áudio", page_icon="🔊", layout="wide")
    render_navigation()
    st.title("🔊 Áudio")
    st.caption("Uso acadêmico. Use apenas gravações de teste, sem dados identificáveis.")
    st.sidebar.header("Entrada de áudio")
    st.sidebar.button("Limpar áudio e resultados", key="_audio_clear", on_click=clear_audio)
    if not local_uploads_enabled(st.get_option("server.address")):
        clear_audio()
        st.info("Importação desativada nesta execução. Inicie localmente, sem túnel ou proxy público:")
        st.code(LOCAL_COMMAND, language="text")
        return
    model_dir = st.sidebar.text_input(
        "Pasta local do modelo", value=os.environ.get("FIAP_WHISPER_MODEL_DIR", ""),
        key="_audio_model", placeholder="C:\\modelos\\faster-whisper-tiny",
        help="Pasta absoluta de pesos já obtidos. Não aceita URL ou nome como tiny; não baixa modelos.",
    )
    language_label = st.sidebar.selectbox("Idioma do áudio", ["Português", "Inglês"], key="_audio_language")
    language = "pt" if language_label == "Português" else "en"
    generation = st.session_state.get("_audio_upload_generation", 0)
    upload = st.sidebar.file_uploader(
        "Importar WAV (até 60 s / 2 MiB)", type=["wav"], key=f"_audio_file_{generation}",
        help="PCM16 mono a 16.000 Hz. O conteúdo é verificado antes de reproduzir ou transcrever.",
    )
    st.sidebar.caption("Processamento na memória desta sessão, sem gravar áudio/transcrição ou enviar a APIs.")
    with st.expander("Formato e preparação local"):
        st.write(
            "WAV RIFF canônico: PCM16 mono, 16.000 Hz, 0,25–60 s, até 2 MiB, "
            "somente blocos fmt e data. Arquivos comprimidos, estéreo, metadados e "
            "cabeçalhos inconsistentes são rejeitados, sem conversão automática. "
            "O Streamlit recebe o upload antes da nossa validação."
        )
        st.write("Se necessário, converta uma gravação de teste com seu FFmpeg local, fora do aplicativo:")
        st.caption("Escolha antes um trecho com fala, de até 60 s. O comando só converte; não recorta nem sobrescreve arquivos existentes.")
        st.code('ffmpeg -n -i trecho.wav -ac 1 -ar 16000 -c:a pcm_s16le -map_metadata -1 -fflags +bitexact audio_teste.wav', language="text")
        st.write(
            "Use faster-whisper no mesmo Python do Streamlit e pesos CTranslate2 já locais, "
            "incluindo model.bin, config.json e tokenizer.json. Recomenda-se tiny multilíngue "
            "para teste em Português; tiny.en é somente Inglês. Dependências e pesos não são instalados pelo app."
        )
        st.code('python -m pip install "streamlit==1.32.0" "faster-whisper==1.2.1"\npython -m pip check', language="text")
    payload = None
    if upload is not None:
        if upload.size > MAX_BYTES:
            sync_selection(st.session_state, None)
            st.error("Arquivo excede 2 MiB. Escolha um trecho menor.")
            return
        payload = upload.getvalue()
    key = selection_key(payload, model_dir, language) if payload is not None else None
    sync_selection(st.session_state, key)
    if payload is None:
        st.info("Selecione um WAV na barra lateral para validar e ouvir antes de transcrever.")
        return
    if st.sidebar.button("Validar e ouvir", key="_audio_validate", use_container_width=True):
        for name in ("_audio_preview", "_audio_result", "_audio_error"):
            st.session_state.pop(name, None)
        try:
            st.session_state["_audio_preview"] = (key, validate_wav(payload))
        except AudioError as error:
            st.session_state["_audio_error"] = str(error)
    preview = st.session_state.get("_audio_preview")
    if not preview or preview[0] != key:
        if st.session_state.get("_audio_error"):
            st.error(st.session_state["_audio_error"])
        return
    sample = preview[1]
    st.subheader("Prévia do arquivo validado")
    a, b, c = st.columns(3)
    a.metric("Duração", f"{sample.seconds:.2f} s")
    b.metric("Formato", "PCM16 · mono")
    c.metric("Amostragem", "16.000 Hz")
    st.caption(f"Identificação do conteúdo: {sample.fingerprint[:12]}. A validação técnica não verifica a origem do áudio.")
    st.audio(sample.payload, format="audio/wav")
    if sample.digital_silence:
        st.info("Todos os valores PCM são zero. Controle de silêncio digital; transcrição não executada.")
        return
    if st.sidebar.button("Transcrever localmente", key="_audio_run", type="primary", use_container_width=True):
        st.session_state.pop("_audio_result", None)
        st.session_state.pop("_audio_error", None)
        try:
            with st.spinner("Reconhecendo fala no computador local…"):
                st.session_state["_audio_result"] = (key, transcribe_local(payload, model_dir, language))
        except AudioError as error:
            st.session_state["_audio_error"] = str(error)
    if st.session_state.get("_audio_error"):
        st.error(st.session_state["_audio_error"])
    saved = st.session_state.get("_audio_result")
    if saved and saved[0] == key:
        render_results(saved[1])


if __name__ == "__main__":
    main()
