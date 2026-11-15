"""WAV validation and optional offline ASR. No downloads, subprocesses or storage."""

from dataclasses import dataclass
import hashlib
import importlib
import json
import math
from pathlib import Path
import re
import struct

MAX_BYTES = 2 * 1024 * 1024
MAX_SECONDS = 60
SAMPLE_RATE = 16000
VERSION = "audio-local-v1"
MAX_SEGMENTS = 200
MAX_TEXT = 20000
LFS_POINTER = re.compile(
    rb"version https://git-lfs\.github\.com/spec/v1\r?\n"
    rb"oid sha256:[0-9a-f]{64}\r?\nsize [0-9]{1,20}\r?\n?"
)


class AudioError(ValueError):
    """A safe, actionable message for an invalid input or unavailable local ASR."""


@dataclass(frozen=True)
class WavSample:
    payload: bytes
    pcm: bytes
    frames: int
    fingerprint: str

    @property
    def seconds(self):
        return self.frames / SAMPLE_RATE

    @property
    def digital_silence(self):
        return not any(self.pcm)


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class Transcript:
    segments: tuple[Segment, ...]
    language: str
    fingerprint: str
    engine: str = "faster-whisper / CPU int8 / local"

    @property
    def text(self):
        return " ".join(segment.text for segment in self.segments)


def validate_wav(payload: bytes) -> WavSample:
    """Parse RIFF directly before any playback/ASR; accept canonical PCM only.

    Two chunks are accepted: fmt (16 bytes) then data. Reject metadata, extended
    fmt, compressed/audio-video containers, truncated files and trailing bytes.
    Deliberately strict: conversion of a test recording is an explicit local step.
    """
    if not isinstance(payload, bytes) or len(payload) > MAX_BYTES:
        raise AudioError("Use um WAV de até 2 MiB.")
    if LFS_POINTER.fullmatch(payload):
        raise AudioError(
            "Este arquivo é um ponteiro Git LFS, não contém áudio. "
            "Obtenha o WAV completo na fonte original e tente novamente."
        )
    if len(payload) < 44 or payload[:4] != b"RIFF" or payload[8:12] != b"WAVE":
        raise AudioError("O conteúdo não é um WAV RIFF válido.")
    if struct.unpack_from("<I", payload, 4)[0] != len(payload) - 8:
        raise AudioError("WAV truncado ou com dados adicionais após o RIFF.")
    if payload[12:16] != b"fmt " or struct.unpack_from("<I", payload, 16)[0] != 16:
        raise AudioError("Use WAV PCM canônico, sem metadados ou cabeçalho estendido.")
    fmt, channels, rate, byte_rate, alignment, bits = struct.unpack_from("<HHIIHH", payload, 20)
    if (fmt, channels, rate, bits) != (1, 1, SAMPLE_RATE, 16):
        raise AudioError("Formato exigido: PCM16 sem compressão, mono, 16.000 Hz.")
    if (byte_rate, alignment) != (SAMPLE_RATE * 2, 2):
        raise AudioError("Taxa de bytes ou alinhamento PCM inconsistente.")
    if payload[36:40] != b"data":
        raise AudioError("Use WAV com apenas os blocos fmt e data, sem metadados.")
    data_size = struct.unpack_from("<I", payload, 40)[0]
    if data_size != len(payload) - 44 or data_size % 2:
        raise AudioError("Tamanho de dados PCM inconsistente ou arquivo truncado.")
    frames = data_size // 2
    if not SAMPLE_RATE // 4 <= frames <= SAMPLE_RATE * MAX_SECONDS:
        raise AudioError("A duração aceita é de 0,25 a 60 segundos.")
    return WavSample(payload, payload[44:], frames, hashlib.sha256(payload).hexdigest())


def validate_model_dir(value: str) -> Path:
    """An explicit existing directory is mandatory; model names/URLs are rejected.

    tokenizer.json is mandatory: faster-whisper otherwise has an online tokenizer
    fallback even when model weights are local. No generic model name is passed.
    """
    if not isinstance(value, str) or not value.strip():
        raise AudioError("Informe a pasta absoluta dos pesos locais do faster-whisper.")
    path = Path(value.strip())
    # A UNC/network path is not local storage; do not probe it on Windows.
    if value.strip().startswith(("\\\\", "//")) or "://" in value or not path.is_absolute():
        raise AudioError("Use uma pasta absoluta local, não URL, nome de modelo ou caminho de rede.")
    try:
        if not path.is_dir():
            raise AudioError("Pasta do modelo não encontrada. Nenhum download foi iniciado.")
        for name in ("model.bin", "config.json", "tokenizer.json"):
            item = path / name
            if not item.is_file() or item.stat().st_size == 0:
                raise AudioError(f"Modelo local incompleto: falta {name}. Nenhum download foi iniciado.")
        config = path / "preprocessor_config.json"
        if config.exists():
            if config.stat().st_size > 65536:
                raise AudioError("Configuração local do modelo excede o limite técnico.")
            config_data = json.loads(config.read_text(encoding="utf-8"))
            if not isinstance(config_data, dict) or config_data.get("sampling_rate", SAMPLE_RATE) != SAMPLE_RATE:
                raise AudioError("O modelo deve trabalhar com áudio de 16.000 Hz.")
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AudioError("Não foi possível ler a configuração do modelo local.") from error
    return path.resolve()


def transcribe_local(payload: bytes, model_dir: str, language: str = "pt") -> Transcript:
    """Real ASR only. Optional dependency is imported after strict validation.

    The PCM array is passed directly to Whisper, avoiding general media decoding.
    No prompts/hotwords, no remote fallback, no cache of user audio or transcript.
    """
    sample = validate_wav(payload)
    if language not in {"pt", "en"}:
        raise AudioError("Selecione Português ou Inglês.")
    if sample.digital_silence:
        raise AudioError("PCM inteiramente zerado: a transcrição não foi executada.")
    path = validate_model_dir(model_dir)
    try:
        fw = importlib.import_module("faster_whisper")
        np = importlib.import_module("numpy")
    except (ImportError, OSError) as error:
        raise AudioError(
            "faster-whisper indisponível neste Python. Instale a dependência opcional "
            "no mesmo ambiente que executa o Streamlit."
        ) from error
    try:
        model = fw.WhisperModel(
            str(path), device="cpu", compute_type="int8", cpu_threads=2,
            num_workers=1, local_files_only=True,
        )
        if language not in model.supported_languages:
            raise AudioError("O modelo selecionado não suporta esse idioma. Para Português, use pesos multilíngues.")
        if model.feature_extractor.sampling_rate != SAMPLE_RATE:
            raise AudioError("O modelo carregado não usa 16.000 Hz.")
        waveform = np.frombuffer(sample.pcm, dtype="<i2").astype(np.float32) / 32768.0
        source, _info = model.transcribe(
            waveform, language=language, task="transcribe", beam_size=1,
            best_of=1, temperature=0.0, condition_on_previous_text=False,
            vad_filter=False, word_timestamps=False, without_timestamps=False,
            max_new_tokens=128,
        )
        segments = []
        count = 0
        previous_start = 0.0
        for index, item in enumerate(source):  # Inference runs while iterating.
            if index >= MAX_SEGMENTS:
                raise AudioError("Transcrição excedeu o limite técnico de segmentos; resultado descartado.")
            start, end = float(item.start), float(item.end)
            if (not math.isfinite(start) or not math.isfinite(end) or
                    start < 0 or start < previous_start or end <= start or
                    end > sample.seconds + 0.1):
                raise AudioError("O modelo retornou tempos inconsistentes; resultado descartado.")
            if not isinstance(item.text, str):
                raise AudioError("O modelo retornou texto inválido; resultado descartado.")
            text = item.text.strip()
            count += len(text)
            if count > MAX_TEXT:
                raise AudioError("Transcrição excedeu o limite técnico de texto; resultado descartado.")
            previous_start = start
            if text:
                segments.append(Segment(start, end, text))
        return Transcript(tuple(segments), language, sample.fingerprint)
    except AudioError:
        raise
    except Exception as error:
        # Do not expose stack traces, local filenames or transcript fragments in UI.
        raise AudioError(
            "A transcrição local falhou. Verifique pesos compatíveis, dependências e memória. "
            "Nenhum resultado parcial foi mantido e não há fallback para a nuvem."
        ) from error


def selection_key(payload: bytes, model_dir: str, language: str) -> str:
    digest = hashlib.sha256(payload).hexdigest()
    return hashlib.sha256(json.dumps([VERSION, digest, model_dir.strip(), language]).encode()).hexdigest()


def sync_selection(state, key):
    if state.get("_audio_selection") != key:
        for name in ("_audio_preview", "_audio_result", "_audio_error"):
            state.pop(name, None)
        state["_audio_selection"] = key


def timestamp(seconds: float) -> str:
    milliseconds = round(seconds * 1000)
    minutes, remainder = divmod(milliseconds, 60000)
    second, millisecond = divmod(remainder, 1000)
    return f"{minutes:02d}:{second:02d}.{millisecond:03d}"
