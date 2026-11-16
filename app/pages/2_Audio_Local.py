"""Optional academic local audio page; Streamlit 1.32 native widgets."""

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
    AudioError, MAX_BYTES, sync_selection, timestamp,
    configured_model_dir, local_asr_setup_present, transcribe_local,
)
from audio_text import sentiment_summary, term_rows
from audio_inputs import input_selection_key, prepare_audio, source_format


def clear_audio():
    generation = st.session_state.get("_audio_upload_generation", 0) + 1
    # Keep the language control; discard upload and all derived results.
    for name in list(st.session_state):
        if name.startswith("_audio_") and name != "_audio_language":
            del st.session_state[name]
    st.session_state["_audio_upload_generation"] = generation


def render_results(result):
    st.subheader("Transcrição local")
    st.caption(f"Motor: {result.engine}. Idioma selecionado: {result.language}. Tempos aproximados.")
    st.caption("Reconhecimento automático pode errar, omitir palavras ou gerar texto inexistente. Confira ouvindo o arquivo.")
    if result.vad_seconds is not None:
        st.caption(
            f"Filtro local de fala: {result.vad_seconds:.2f} s retidos. "
            "Falas baixas, curtas ou atípicas podem ser perdidas. Um segmento pode abranger pausas."
        )
    if result.vad_seconds == 0:
        st.info("Nenhum trecho de fala detectado pelo filtro local. Isso não comprova ausência de fala.")
        return
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
    model_dir = configured_model_dir()
    setup_present = local_asr_setup_present(model_dir)
    if not setup_present:
        st.session_state.pop("_audio_result", None)
        st.sidebar.info("Transcrição indisponível: prepare o motor e os pesos locais. Veja Formato e preparação local.")
    language_label = st.sidebar.selectbox("Idioma do áudio", ["Português", "Inglês"], key="_audio_language")
    language = "pt" if language_label == "Português" else "en"
    generation = st.session_state.get("_audio_upload_generation", 0)
    upload = st.sidebar.file_uploader(
        "Importar WAV ou MP3 (até 12 min / 24 MiB)", type=["wav", "mp3"], key=f"_audio_file_{generation}",
        help="WAV PCM16 mono a 16.000 Hz ou MP3. MP3 é convertido localmente antes da reprodução, sem alterar o original.",
    )
    st.sidebar.caption("Processamento na memória desta sessão, sem gravar áudio/transcrição ou enviar a APIs.")
    with st.expander("Formato e preparação local"):
        st.write(
            "WAV RIFF canônico: PCM16 mono, 16.000 Hz, 0,25 s a 12 min, até 24 MiB, "
            "um bloco fmt PCM de 16 bytes e um bloco data. Metadados LIST/INFO e "
            "padding JUNK/PAD de até 64 KiB são validados e descartados. Arquivos "
            "comprimidos, estéreo, blocos desconhecidos e cabeçalhos inconsistentes "
            "são rejeitados no caminho WAV. "
            "O Streamlit recebe o upload antes da nossa validação."
        )
        st.write("MP3: conversão local para PCM16 mono a 16.000 Hz, usando FFmpeg já disponível no PATH. O original não é alterado. Outros formatos ainda não são aceitos.")
        st.caption("A conversão MP3 aguarda até 60 s, mais o encerramento do processo, e só entrega ao player o áudio decodificado e validado. Dependendo da codificação, o MP3 pode ser recusado.")
        st.write("Se necessário, converta uma gravação de teste com seu FFmpeg local, fora do aplicativo:")
        st.caption("Gravações dentro do limite decodificado de 12 min podem ser usadas inteiras. O comando só converte; não recorta nem sobrescreve arquivos existentes.")
        st.code('ffmpeg -n -i entrada.wav -ac 1 -ar 16000 -c:a pcm_s16le -map_metadata -1 -fflags +bitexact audio_teste.wav', language="text")
        st.write(
            "O modelo é encontrado automaticamente em models/whisper-tiny na raiz do projeto. "
            "Copie para lá os pesos locais já obtidos: model.bin, config.json e tokenizer.json. "
            "Use faster-whisper no mesmo Python do Streamlit. Consulte docs/AUDIO_LOCAL.md "
            "para a preparação sem sobrescrever arquivos. Dependências e pesos não são instalados pelo app."
        )
        st.code('python -m pip install "streamlit==1.32.0" "faster-whisper==1.2.1"\npython -m pip check', language="text")
    payload = None
    filename = getattr(upload, "name", "audio.wav") if upload is not None else "audio.wav"
    if upload is not None:
        if upload.size > MAX_BYTES:
            sync_selection(st.session_state, None)
            st.error("Arquivo excede 24 MiB. Escolha um trecho menor.")
            return
        payload = upload.getvalue()
    key = input_selection_key(payload, filename, model_dir, language, generation) if payload is not None else None
    sync_selection(st.session_state, key)
    if payload is None:
        st.info("Selecione um WAV ou MP3 na barra lateral para validar e ouvir antes de transcrever.")
        return
    if st.sidebar.button("Validar e ouvir", key="_audio_validate", use_container_width=True):
        for name in ("_audio_preview", "_audio_result", "_audio_error"):
            st.session_state.pop(name, None)
        try:
            with st.spinner("Validando o áudio localmente…"):
                prepared = prepare_audio(payload, filename)
                if st.session_state.get("_audio_selection") == key:
                    st.session_state["_audio_preview"] = (key, prepared)
        except AudioError as error:
            if st.session_state.get("_audio_selection") == key:
                st.session_state["_audio_error"] = str(error)
    preview = st.session_state.get("_audio_preview")
    if not preview or preview[0] != key or st.session_state.get("_audio_selection") != key:
        if st.session_state.get("_audio_error"):
            st.error(st.session_state["_audio_error"])
        return
    sample = preview[1]
    st.subheader("Prévia do arquivo validado")
    a, b, c = st.columns(3)
    a.metric("Duração", f"{sample.seconds:.2f} s")
    b.metric("Formato", "PCM16 · mono")
    c.metric("Amostragem", "16.000 Hz")
    st.caption(f"Identificação do arquivo de origem: {sample.fingerprint[:12]}. A validação técnica não verifica a origem do áudio.")
    if source_format(filename) == "mp3":
        st.caption("MP3 convertido localmente para WAV PCM16 mono a 16.000 Hz. Original preservado; player e transcrição usam a versão normalizada.")
    if sample.metadata_discarded:
        st.caption("Metadados opcionais descartados antes da reprodução e da transcrição.")
    st.audio(sample.payload, format="audio/wav")
    if sample.digital_silence:
        st.info("Todos os valores PCM são zero. Controle de silêncio digital; transcrição não executada.")
        return
    if sample.seconds > 60:
        st.info(
            "Áudio longo: a transcrição local pode levar vários minutos e usar mais memória. "
            "O arquivo inteiro será enviado ao motor local, sem corte automático."
        )
    st.sidebar.caption(
        "Execução síncrona. Orçamento de 15 min verificado entre etapas; uma chamada "
        "nativa em andamento pode ultrapassá-lo. Não há cancelamento imediato."
    )
    if st.sidebar.button("Transcrever localmente", key="_audio_run", type="primary", use_container_width=True,
                         disabled=not setup_present) and setup_present:
        st.session_state.pop("_audio_result", None)
        st.session_state.pop("_audio_error", None)
        meter = st.progress(0.0, text="Carregando o modelo local…")

        def update_progress(stage, position, total):
            if stage == "loading":
                text = "Carregando o modelo local…"
            elif stage == "transcribing":
                text = "Detectando fala e transcrevendo: aguardando o primeiro segmento…"
            elif stage == "complete":
                text = "Processamento concluído."
            else:
                text = f"Posição do último segmento retornado: {timestamp(position)} / {timestamp(total)}"
            meter.progress(min(position / total, 1.0), text=text)

        try:
            with st.spinner("Reconhecendo fala no computador local…"):
                result = transcribe_local(sample.payload, model_dir, language, progress=update_progress)
                if st.session_state.get("_audio_selection") == key:
                    st.session_state["_audio_result"] = (key, result)
        except AudioError as error:
            if st.session_state.get("_audio_selection") == key:
                st.session_state["_audio_error"] = str(error)
        finally:
            meter.empty()
    if st.session_state.get("_audio_error"):
        st.error(st.session_state["_audio_error"])
    saved = st.session_state.get("_audio_result")
    if saved and saved[0] == key and st.session_state.get("_audio_selection") == key:
        render_results(saved[1])


if __name__ == "__main__":
    main()
