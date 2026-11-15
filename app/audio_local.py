"""WAV validation and optional offline ASR. No downloads, subprocesses or storage."""

from dataclasses import dataclass
import hashlib
import importlib
import json
import math
from pathlib import Path
import re
import struct
import time
from typing import Callable, Optional

MAX_BYTES = 24 * 1024 * 1024
MAX_SECONDS = 12 * 60
SAMPLE_RATE = 16000
VERSION = "audio-local-v3-long-vad"
MAX_SEGMENTS = 1000
MAX_TEXT = 60000
MAX_INFERENCE_SECONDS = 15 * 60
MAX_METADATA_BYTES = 64 * 1024
MAX_RIFF_CHUNKS = 16
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
    metadata_discarded: bool = False

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
    engine: str = "faster-whisper / CPU int8 / Silero VAD local"
    vad_seconds: Optional[float] = None

    @property
    def text(self):
        return " ".join(segment.text for segment in self.segments)


def _validate_info(chunk: bytes):
    """Validate bounded LIST/INFO structure, never decode or return its values."""
    if chunk[:4] != b"INFO":
        raise AudioError("Somente metadados LIST/INFO são aceitos neste WAV.")
    position = 4
    count = 0
    while position < len(chunk):
        count += 1
        if count > 64 or len(chunk) - position < 8:
            raise AudioError("Estrutura de metadados WAV inválida.")
        size = struct.unpack_from("<I", chunk, position + 4)[0]
        end = position + 8 + size
        padded_end = end + (size % 2)
        if padded_end > len(chunk) or (size % 2 and chunk[end] != 0):
            raise AudioError("Tamanho ou padding dos metadados WAV inválido.")
        position = padded_end


def validate_wav(payload: bytes) -> WavSample:
    """Parse RIFF directly before playback/ASR; accept bounded PCM16 mono.

    Exactly one fmt (16 bytes) before one data chunk. Optional LIST/INFO, JUNK and
    PAD chunks are bounded, structurally checked and discarded. The returned WAV
    is canonical and contains no metadata; the fingerprint identifies the source.
    Reject unknown/extended chunks, compression, truncation and trailing bytes.
    """
    if not isinstance(payload, bytes) or len(payload) > MAX_BYTES:
        raise AudioError("Use um WAV de até 24 MiB.")
    if LFS_POINTER.fullmatch(payload):
        raise AudioError(
            "Este arquivo é um ponteiro Git LFS, não contém áudio. "
            "Obtenha o WAV completo na fonte original e tente novamente."
        )
    if len(payload) < 44 or payload[:4] != b"RIFF" or payload[8:12] != b"WAVE":
        raise AudioError("O conteúdo não é um WAV RIFF válido.")
    if struct.unpack_from("<I", payload, 4)[0] != len(payload) - 8:
        raise AudioError("WAV truncado ou com dados adicionais após o RIFF.")
    position = 12
    fmt_chunk = None
    pcm = None
    count = 0
    metadata_bytes = 0
    while position < len(payload):
        count += 1
        if count > MAX_RIFF_CHUNKS or len(payload) - position < 8:
            raise AudioError("Quantidade ou estrutura de blocos WAV inválida.")
        kind = payload[position:position + 4]
        size = struct.unpack_from("<I", payload, position + 4)[0]
        start = position + 8
        end = start + size
        padded_end = end + (size % 2)
        if padded_end > len(payload) or (size % 2 and payload[end] != 0):
            raise AudioError("Tamanho ou padding dos blocos WAV inválido.")
        if kind == b"fmt ":
            if fmt_chunk is not None or pcm is not None or size != 16:
                raise AudioError("Use um único bloco fmt PCM de 16 bytes, antes dos dados.")
            fmt_chunk = payload[start:end]
        elif kind == b"data":
            if pcm is not None or fmt_chunk is None:
                raise AudioError("Use um único bloco data após o formato PCM.")
            pcm = payload[start:end]
        elif kind in {b"LIST", b"JUNK", b"PAD "}:
            metadata_bytes += padded_end - position
            if metadata_bytes > MAX_METADATA_BYTES:
                raise AudioError("Metadados WAV excedem o limite de 64 KiB.")
            if kind == b"LIST":
                _validate_info(payload[start:end])
        else:
            raise AudioError("Bloco WAV não suportado. Converta para WAV PCM16 mono a 16.000 Hz.")
        position = padded_end
    if fmt_chunk is None or pcm is None:
        raise AudioError("WAV sem os blocos obrigatórios de formato e dados.")
    fmt, channels, rate, byte_rate, alignment, bits = struct.unpack("<HHIIHH", fmt_chunk)
    if (fmt, channels, rate, bits) != (1, 1, SAMPLE_RATE, 16):
        raise AudioError("Formato exigido: PCM16 sem compressão, mono, 16.000 Hz.")
    if (byte_rate, alignment) != (SAMPLE_RATE * 2, 2):
        raise AudioError("Taxa de bytes ou alinhamento PCM inconsistente.")
    data_size = len(pcm)
    if data_size % 2:
        raise AudioError("Tamanho de dados PCM inconsistente ou arquivo truncado.")
    frames = data_size // 2
    if not SAMPLE_RATE // 4 <= frames <= SAMPLE_RATE * MAX_SECONDS:
        raise AudioError("A duração aceita é de 0,25 segundo a 12 minutos.")
    canonical = payload if not metadata_bytes else (
        b"RIFF" + struct.pack("<I", 36 + data_size) + b"WAVEfmt " +
        struct.pack("<I", 16) + fmt_chunk + b"data" + struct.pack("<I", data_size) + pcm
    )
    return WavSample(canonical, pcm, frames, hashlib.sha256(payload).hexdigest(), bool(metadata_bytes))


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


def transcribe_local(
    payload: bytes, model_dir: str, language: str = "pt", *,
    progress: Optional[Callable[[str, float, float], None]] = None,
) -> Transcript:
    """Real ASR only. Optional dependency is imported after strict validation.

    The PCM array is passed directly to Whisper, avoiding general media decoding.
    Bundled Silero VAD selects speech; native Whisper windows and word alignment
    restore file-relative timestamps. No app-level cuts, overlaps, deduplication,
    string filtering, prompt or remote fallback.

    The 15-minute budget is cooperative, checked around native calls and after
    returned segments. It cannot preempt a running C++/NumPy call. The optional
    callback reports the last returned segment end, NOT completed audio coverage.
    """
    sample = validate_wav(payload)
    if language not in {"pt", "en"}:
        raise AudioError("Selecione Português ou Inglês.")
    if sample.digital_silence:
        raise AudioError("PCM inteiramente zerado: a transcrição não foi executada.")
    path = validate_model_dir(model_dir)
    started = time.monotonic()

    def check_budget():
        if time.monotonic() - started > MAX_INFERENCE_SECONDS:
            raise AudioError(
                "O processamento excedeu o orçamento de 15 minutos. "
                "Resultado descartado; tente um trecho menor ou um modelo local menor."
            )

    def report(stage, position=0.0):
        if progress is not None:
            progress(stage, position, sample.seconds)

    try:
        fw = importlib.import_module("faster_whisper")
        np = importlib.import_module("numpy")
        ort = importlib.import_module("onnxruntime")
        ort.disable_telemetry_events()
    except (ImportError, OSError) as error:
        raise AudioError(
            "Motor de fala ou dependências opcionais indisponíveis neste Python. Instale faster-whisper "
            "no mesmo ambiente que executa o Streamlit."
        ) from error
    try:
        report("loading")
        check_budget()
        model = fw.WhisperModel(
            str(path), device="cpu", compute_type="int8", cpu_threads=2,
            num_workers=1, local_files_only=True,
        )
        check_budget()
        if language not in model.supported_languages:
            raise AudioError("O modelo selecionado não suporta esse idioma. Para Português, use pesos multilíngues.")
        if model.feature_extractor.sampling_rate != SAMPLE_RATE:
            raise AudioError("O modelo carregado não usa 16.000 Hz.")
        waveform = np.frombuffer(sample.pcm, dtype="<i2").astype(np.float32)
        waveform /= 32768.0
        report("transcribing")
        source, info = model.transcribe(
            waveform, language=language, task="transcribe", beam_size=1,
            best_of=1, temperature=0.0, condition_on_previous_text=False,
            vad_filter=True, word_timestamps=True, without_timestamps=False,
        )
        check_budget()
        vad_seconds = float(info.duration_after_vad)
        if not math.isfinite(vad_seconds) or not 0 <= vad_seconds <= sample.seconds + 0.1:
            raise AudioError("O filtro local retornou duração inválida; resultado descartado.")
        if vad_seconds == 0:
            report("complete", sample.seconds)
            return Transcript((), language, sample.fingerprint, vad_seconds=0.0)
        segments = []
        count = 0
        previous_start = 0.0
        latest_end = 0.0
        for index, item in enumerate(source):  # Inference runs while iterating.
            check_budget()
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
            latest_end = max(latest_end, min(end, sample.seconds))
            report("segment", latest_end)
        check_budget()
        report("complete", sample.seconds)
        return Transcript(tuple(segments), language, sample.fingerprint, vad_seconds=vad_seconds)
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
