"""Bounded WAV/MP3 admission. MP3 is decoded locally, never played unvalidated."""
from dataclasses import dataclass, replace
import hashlib
import os
import shutil
import struct
import subprocess
import threading

from audio_local import AudioError, MAX_BYTES, MAX_SECONDS, SAMPLE_RATE, selection_key, validate_wav

VERSION = "audio-input-v1-mp3"
MAX_ID3_BYTES = 64 * 1024
MAX_DECODE_SECONDS = 60
MAX_PCM_BYTES = MAX_SECONDS * SAMPLE_RATE * 2
# MPEG encoder delay/padding and an optional Xing frame are not decoded duration.
MAX_ENCODED_SECONDS = MAX_SECONDS + 0.25
MPEG1_BITRATES = (0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320)
MPEG2_BITRATES = (0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160)


@dataclass(frozen=True)
class Mp3Frames:
    payload: bytes
    frame_count: int
    sample_rate: int
    channels: int
    encoded_seconds: float
    metadata_discarded: bool


def source_format(filename):
    return filename.rsplit('.', 1)[-1].lower() if isinstance(filename, str) and '.' in filename else ''


def input_selection_key(payload, filename, model_dir, language, generation=0):
    key = selection_key(payload, model_dir, language)
    return hashlib.sha256(f"{VERSION}|{generation}|{source_format(filename)}|{key}".encode()).hexdigest()


def inspect_mp3(payload: bytes) -> Mp3Frames:
    """Walk complete Layer III frames; no sync-search or generic format sniffing.

    Accept MPEG1/2/2.5 mono/stereo at a constant source rate (8–48 kHz), including
    variable bitrates. Bounded leading ID3v2.3/v2.4 (without footer) and trailing
    ID3v1 are discarded without parsing or exposing their values. Unsupported
    metadata, free-format streams, other layers, truncation and mixed rates fail.
    """
    if not isinstance(payload, bytes) or len(payload) > MAX_BYTES:
        raise AudioError('Use um áudio de até 24 MiB.')
    start, stop = 0, len(payload)
    metadata_bytes = 0
    if payload[:3] == b'ID3':
        if len(payload) < 10 or payload[3] not in (3, 4) or payload[4] != 0:
            raise AudioError('Cabeçalho ID3 do MP3 não suportado.')
        allowed_flags = 0xE0 if payload[3] == 3 else 0xF0
        if payload[5] & ~allowed_flags or payload[5] & 0x10 or any(v & 0x80 for v in payload[6:10]):
            raise AudioError('Metadados ID3 do MP3 inválidos ou não suportados.')
        size = 0
        for value in payload[6:10]:
            size = (size << 7) | value
        start = 10 + size
        metadata_bytes = start
        if start > stop or metadata_bytes > MAX_ID3_BYTES:
            raise AudioError('Metadados MP3 truncados ou maiores que 64 KiB.')
    if stop - start >= 128 and payload[stop - 128:stop - 125] == b'TAG':
        stop -= 128
        metadata_bytes += 128
    if metadata_bytes > MAX_ID3_BYTES:
        raise AudioError('Metadados MP3 maiores que 64 KiB.')
    position = start
    count = 0
    signature = None
    while position < stop:
        if stop - position < 4:
            raise AudioError('MP3 truncado ou com dados adicionais.')
        header = int.from_bytes(payload[position:position + 4], 'big')
        version = (header >> 19) & 3
        layer = (header >> 17) & 3
        bitrate_index = (header >> 12) & 15
        rate_index = (header >> 10) & 3
        if (header >> 21 != 0x7FF or version == 1 or layer != 1 or
                bitrate_index in (0, 15) or rate_index == 3 or (header & 3) == 2):
            raise AudioError('O conteúdo não é um MP3 Layer III suportado. Renomear a extensão não converte áudio.')
        rate = (44100, 48000, 32000)[rate_index] // (1 if version == 3 else 2 if version == 2 else 4)
        channels = 1 if (header >> 6) & 3 == 3 else 2
        current = (version, rate, channels)
        if signature is not None and current != signature:
            raise AudioError('MP3 com formato ou taxa variável entre quadros não suportado.')
        signature = current
        bitrate = (MPEG1_BITRATES if version == 3 else MPEG2_BITRATES)[bitrate_index]
        frame_size = (144000 if version == 3 else 72000) * bitrate // rate + ((header >> 9) & 1)
        if position + frame_size > stop:
            raise AudioError('Quadro MP3 truncado.')
        count += 1
        samples_per_frame = 1152 if version == 3 else 576
        if count * samples_per_frame / rate > MAX_ENCODED_SECONDS:
            raise AudioError('MP3 excede o limite de 12 minutos.')
        position += frame_size
    if count < 2 or signature is None:
        raise AudioError('MP3 sem quadros de áudio suficientes.')
    return Mp3Frames(payload[start:stop], count, rate, channels,
                     count * samples_per_frame / rate, bool(metadata_bytes))


def _decode_mp3(payload: bytes) -> bytes:
    """Isolated optional FFmpeg; memory-only pipes, bounded output and deadline.

    No uploaded filename, file URL, shell, network protocol or temporary media is
    passed to FFmpeg. Decoder and demuxer are forced; pipe readers cap retained
    PCM and discard stderr. This is process isolation, not an OS security sandbox.
    """
    executable = shutil.which('ffmpeg')
    if not executable:
        raise AudioError('MP3 precisa do FFmpeg local disponível no PATH. WAV continua funcionando sem ele.')
    command = [
        executable, '-nostdin', '-hide_banner', '-loglevel', 'error', '-xerror',
        '-max_alloc', '67108864', '-threads', '1', '-protocol_whitelist', 'pipe',
        '-f', 'mp3', '-codec:a', 'mp3', '-err_detect', 'explode', '-i', 'pipe:0',
        '-map', '0:a:0', '-vn', '-sn', '-dn', '-map_metadata', '-1',
        '-ac', '1', '-ar', str(SAMPLE_RATE), '-codec:a', 'pcm_s16le',
        '-t', str(MAX_SECONDS + 1), '-f', 's16le', 'pipe:1',
    ]
    environment = os.environ.copy()
    environment.pop('FFREPORT', None)  # Do not let inherited FFmpeg logging write files.
    try:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, shell=False, bufsize=0, env=environment,
                                   creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    except OSError as error:
        raise AudioError('Não foi possível iniciar o FFmpeg local para este MP3.') from error
    output = bytearray()
    overflow = threading.Event()
    failed_io = threading.Event()

    def kill():
        try:
            process.kill()
        except OSError:
            pass

    def write_input():
        try:
            view = memoryview(payload)
            offset = 0
            while offset < len(view):
                written = process.stdin.write(view[offset:offset + 65536])
                if not written:
                    raise OSError('Incomplete input')
                offset += written
        except (OSError, ValueError):
            failed_io.set()
        finally:
            try:
                process.stdin.close()
            except OSError:
                pass

    def read_output():
        try:
            while chunk := process.stdout.read(65536):
                if len(output) + len(chunk) > MAX_PCM_BYTES:
                    overflow.set()
                    kill()
                    break
                output.extend(chunk)
        except (OSError, ValueError):
            failed_io.set()

    def drain_errors():
        try:
            while process.stderr.read(4096):
                pass
        except (OSError, ValueError):
            failed_io.set()

    workers = [threading.Thread(target=fn, daemon=True) for fn in (write_input, read_output, drain_errors)]
    timed_out = False
    lifecycle_failed = False
    returncode = None
    try:
        for worker in workers:
            worker.start()
        returncode = process.wait(timeout=MAX_DECODE_SECONDS)
    except subprocess.TimeoutExpired:
        timed_out = True
    except Exception:
        lifecycle_failed = True
    finally:
        # Covers startup failures and cancellation as well as the normal deadline.
        if returncode is None:
            kill()
            try:
                returncode = process.wait(timeout=5)
            except Exception:
                lifecycle_failed = True
        for worker in workers:
            try:
                if worker.is_alive():
                    worker.join(timeout=2)
            except Exception:
                lifecycle_failed = True
        for stream in (process.stdin, process.stdout, process.stderr):
            try:
                stream.close()
            except (OSError, ValueError):
                lifecycle_failed = True
        for worker in workers:
            if worker.is_alive():
                kill()
                worker.join(timeout=1)
    if timed_out or any(worker.is_alive() for worker in workers):
        raise AudioError('A conversão MP3 excedeu 60 segundos ou não encerrou corretamente; resultado descartado.')
    if overflow.is_set():
        raise AudioError('O áudio MP3 decodificado excede 12 minutos; resultado descartado, sem corte.')
    if returncode != 0 or failed_io.is_set() or lifecycle_failed:
        raise AudioError('Não foi possível decodificar este MP3. Arquivo inválido ou formato não suportado.')
    if not output or len(output) % 2:
        raise AudioError('O MP3 não gerou amostras PCM válidas.')
    return bytes(output)


def prepare_audio(payload: bytes, filename: str):
    """Return canonical validated PCM WAV; fingerprint identifies original bytes."""
    kind = source_format(filename)
    if kind == 'wav':
        return validate_wav(payload)
    if kind != 'mp3':
        raise AudioError('Selecione WAV ou MP3. Outros formatos ainda não são aceitos.')
    frames = inspect_mp3(payload)
    pcm = _decode_mp3(frames.payload)
    if len(pcm) > MAX_PCM_BYTES:
        raise AudioError('O áudio decodificado excede 12 minutos.')
    header = (b'RIFF' + struct.pack('<I', 36 + len(pcm)) + b'WAVEfmt ' +
              struct.pack('<IHHIIHH', 16, 1, 1, SAMPLE_RATE, SAMPLE_RATE * 2, 2, 16) +
              b'data' + struct.pack('<I', len(pcm)))
    sample = validate_wav(header + pcm)
    return replace(sample, fingerprint=hashlib.sha256(payload).hexdigest(),
                   metadata_discarded=frames.metadata_discarded)
