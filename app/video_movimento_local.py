"""Comparação didática de dois vídeos locais. Núcleo sem dependências de ML.

Não importa MediaPipe/OpenCV ao abrir a página e não contém chamadas de rede.
Isso não substitui a validação de tráfego das dependências nativas.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

ALGORITHM_VERSION = "membro-superior-2d-alinhado-0.2"
MODEL_FILENAME = "pose_landmarker_lite.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
MODEL_SHA256 = "59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a"
MODEL_BYTES = 5_777_746
MAX_BYTES = 20 * 1024 * 1024
MIN_SECONDS = 2.0
MAX_SECONDS = 15.0
MAX_PIXELS = 1280 * 720
MAX_SIDE = 1280
MAX_FPS = 60.0
SAMPLE_HZ = 5
MAX_DECODE_FRAMES = 901
MAX_SAMPLES = 76
PROCESS_TIMEOUT = 45
MAX_RESULT_BYTES = 256 * 1024
WORKER_PACKAGES = {
    "mediapipe": "0.10.21", "numpy": "1.26.4",
    "opencv-contrib-python": "4.11.0.86", "jax": "0.4.38",
    "jaxlib": "0.4.38", "protobuf": "4.25.9",
}
# Código fixo. Não recebe nomes de pacotes/comandos/código do upload ou da UI.
WORKER_PROBE = """import sys,json,struct,importlib.metadata as m
names=('mediapipe','numpy','opencv-contrib-python','jax','jaxlib','protobuf')
conflicts=[]
for name in ('opencv-python','opencv-python-headless','opencv-contrib-python-headless'):
 try:m.version(name);conflicts.append(name)
 except m.PackageNotFoundError:pass
print(json.dumps({'implementation':sys.implementation.name,'version':list(sys.version_info[:2]),'bits':struct.calcsize('P')*8,'venv':sys.prefix!=sys.base_prefix,'conflicts':conflicts,'packages':{name:m.version(name) for name in names}}))
"""


class VideoError(ValueError):
    """Erro seguro para apresentar ao usuário, sem caminhos locais ou stderr."""


def worker_python():
    """Configuração confiável do servidor; nunca é um controle de navegador.

    Não resolve o symlink retornado: isso apagaria a identidade do venv no Linux.
    Ausência da variável mantém o caminho original já testado.
    """
    raw = os.environ.get("FIAP_VIDEO_PYTHON", "")
    if not raw:
        return sys.executable, False
    message = "Configuração do worker de vídeo inválida. Use um Python local absoluto em ambiente separado."
    if len(raw) > 4096 or any(c in raw for c in ("\0", "\n", "\r")) or raw.startswith(("\\\\", "//")) or "://" in raw:
        raise VideoError(message)
    path = Path(raw)
    try:
        if not path.is_absolute() or not path.is_file() or str(path.resolve()).startswith(("\\\\", "//")):
            raise VideoError(message)
        if not re.fullmatch(r"python(?:3(?:\.\d+)?)?(?:\.exe)?", path.name, re.IGNORECASE):
            raise VideoError(message)
        if os.name == "nt" and path.suffix.lower() != ".exe":
            raise VideoError(message)
        if os.name != "nt" and not os.access(path, os.X_OK):
            raise VideoError(message)
    except (OSError, RuntimeError) as exc:
        raise VideoError(message) from exc
    return str(path), True


def verify_worker_python(executable: str):
    """Confere o ambiente configurado sem importar as bibliotecas nativas."""
    message = "O ambiente separado de vídeo não está pronto. Confira requirements-video-worker.txt e execute pip check nele."
    try:
        done = subprocess.run(
            [executable, "-I", "-c", WORKER_PROBE],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            shell=False, timeout=10, check=False,
        )
        if done.returncode != 0 or len(done.stdout) > 4096:
            raise VideoError(message)
        info = json.loads(done.stdout)
        if (info.get("implementation") != "cpython" or info.get("version") not in ([3, 11], [3, 12])
                or info.get("bits") != 64 or info.get("conflicts") != []
                or info.get("venv") is not True or info.get("packages") != WORKER_PACKAGES):
            raise VideoError(message)
    except (OSError, subprocess.TimeoutExpired, ValueError, TypeError, AttributeError) as exc:
        raise VideoError(message) from exc


def local_model_path():
    """Peso configurado no servidor ou no diretório relativo ignorado pelo Git."""
    raw = os.environ.get("FIAP_POSE_MODEL_PATH", "")
    if not raw:
        return Path(__file__).resolve().parents[1] / "models" / "pose-landmarker-lite" / MODEL_FILENAME
    path = Path(raw)
    if (len(raw) > 4096 or any(c in raw for c in ("\0", "\n", "\r"))
            or raw.startswith(("\\\\", "//")) or "://" in raw or not path.is_absolute()):
        raise VideoError("Configuração do modelo local inválida. Confira o caminho configurado no servidor.")
    try:
        if str(path.resolve()).startswith(("\\\\", "//")):
            raise VideoError("Configuração do modelo local inválida. Confira o caminho configurado no servidor.")
    except (OSError, RuntimeError) as exc:
        raise VideoError("Configuração do modelo local inválida. Confira o caminho configurado no servidor.") from exc
    return path


@dataclass(frozen=True)
class Options:
    side: str = "right"
    tolerance_degrees: float = 20.0
    min_event_seconds: float = 0.6
    min_quality: float = 0.7
    joint: str = "elbow"

    def validate(self):
        if self.side not in ("right", "left"):
            raise VideoError("Escolha o lado direito ou esquerdo.")
        if self.joint not in ("elbow", "shoulder"):
            raise VideoError("Escolha o ângulo do cotovelo ou do ombro.")
        for value, low, high in (
            (self.tolerance_degrees, 5, 60),
            (self.min_event_seconds, 0.4, 3),
            (self.min_quality, 0.5, 0.95),
        ):
            if not math.isfinite(value) or not low <= value <= high:
                raise VideoError("Parâmetros de análise fora dos limites.")


def validate_upload(data: bytes):
    if not isinstance(data, bytes) or not data:
        raise VideoError("Envie um arquivo MP4 não vazio.")
    if len(data) > MAX_BYTES:
        raise VideoError("Cada vídeo deve ter no máximo 20 MiB.")
    # Filtro inicial apenas. O decodificador confirma a estrutura posteriormente.
    if len(data) < 24 or data[4:8] != b"ftyp":
        raise VideoError("O conteúdo não tem o cabeçalho MP4 esperado.")
    box_size = int.from_bytes(data[:4], "big")
    if box_size < 16 or box_size > len(data):
        raise VideoError("Cabeçalho MP4 inválido.")


def verify_model(model_path: Path):
    model_path = Path(model_path)
    if not model_path.is_file():
        raise VideoError("Modelo local ausente. Configure o peso no servidor local; não há download automático.")
    if model_path.stat().st_size != MODEL_BYTES:
        raise VideoError("Modelo local com tamanho diferente do modelo aprovado.")
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if digest != MODEL_SHA256:
        raise VideoError("Modelo local não corresponde ao SHA-256 esperado.")


def validate_metadata(meta: dict):
    try:
        width, height = int(meta["width"]), int(meta["height"])
        fps, frames = float(meta["fps"]), int(meta["frames"])
    except (TypeError, ValueError, KeyError, OverflowError) as exc:
        raise VideoError("Metadados do vídeo inválidos.") from exc
    if width <= 0 or height <= 0 or max(width, height) > MAX_SIDE or width * height > MAX_PIXELS:
        raise VideoError("Use até 1280 × 720 pixels, ou 720 × 1280 na vertical.")
    if not math.isfinite(fps) or not SAMPLE_HZ <= fps <= MAX_FPS:
        raise VideoError("Use vídeo de taxa constante entre 5 e 60 quadros por segundo.")
    duration = frames / fps
    if frames <= 0 or frames >= MAX_DECODE_FRAMES or not MIN_SECONDS <= duration <= MAX_SECONDS:
        raise VideoError("Use um clipe completo de 2 a 15 segundos.")
    return {"width": width, "height": height, "fps": fps, "frames": frames, "duration_seconds": duration}


def image_plane_angle(points, width: int, height: int):
    """Ângulo 2D em pixels; x/y normalizados NÃO têm a mesma escala."""
    if len(points) != 3:
        return None
    try:
        coords = [(float(p[0]) * width, float(p[1]) * height) for p in points]
        if not all(math.isfinite(v) for point in coords for v in point):
            return None
        a, b, c = coords
        u, v = (a[0] - b[0], a[1] - b[1]), (c[0] - b[0], c[1] - b[1])
        length = math.hypot(*u) * math.hypot(*v)
        if length < 1e-8:
            return None
        cosine = max(-1.0, min(1.0, (u[0] * v[0] + u[1] * v[1]) / length))
        return math.degrees(math.acos(cosine))
    except (TypeError, ValueError, IndexError):
        return None


# Nome anterior mantido para consumidores do primeiro candidato.
elbow_angle = image_plane_angle


def landmark_indexes(options: Options):
    """Três pontos do mesmo lado, com a articulação no vértice central."""
    options.validate()
    if options.joint == "shoulder":
        return (24, 12, 14) if options.side == "right" else (23, 11, 13)
    return (12, 14, 16) if options.side == "right" else (11, 13, 15)


def angle_definition(options: Options):
    return {
        "joint": options.joint,
        "side": options.side,
        "landmark_indexes": list(landmark_indexes(options)),
        "point_order": ["hip", "shoulder", "elbow"] if options.joint == "shoulder" else ["shoulder", "elbow", "wrist"],
        "space": "image_plane_pixels",
        "range_degrees": [0, 180],
    }


def pose_sample(result, timestamp: float, width: int, height: int, options: Options):
    """Extrai um ângulo somente se exatamente uma pose e três pontos forem utilizáveis."""
    row = {"time_seconds": round(timestamp, 6), "angle_degrees": None, "quality_min": None, "reason": None}
    poses = result.pose_landmarks
    if len(poses) != 1:
        row["reason"] = "sem_pose" if not poses else "multiplas_poses"
        return row
    indexes = landmark_indexes(options)
    try:
        points = [poses[0][i] for i in indexes]
        scores = [float(getattr(p, field, 0) or 0) for p in points for field in ("visibility", "presence")]
        if not all(math.isfinite(score) and 0 <= score <= 1 for score in scores):
            raise ValueError("quality")
        quality = min(scores)
        row["quality_min"] = round(quality, 6)
        if quality < options.min_quality or not all(0 <= p.x <= 1 and 0 <= p.y <= 1 for p in points):
            row["reason"] = "baixa_confianca_ou_oclusao"
            return row
        angle = image_plane_angle([(p.x, p.y) for p in points], width, height)
        if angle is None:
            row["reason"] = "geometria_invalida"
        else:
            row["angle_degrees"] = round(angle, 4)
    except (IndexError, TypeError, ValueError, AttributeError):
        row["reason"] = "pontos_invalidos"
    return row


def compare(reference: dict, observed: dict, options: Options):
    options.validate()
    rmeta, ometa = reference["metadata"], observed["metadata"]
    duration_delta = ometa["duration_seconds"] - rmeta["duration_seconds"]
    if abs(duration_delta) > 0.5:
        raise VideoError("Clipes diferem mais de 0,5 s. Recorte e alinhe o mesmo movimento antes de comparar.")
    ref, obs = reference["samples"], observed["samples"]
    if not ref or not obs:
        raise VideoError("Sem amostras decodificadas para comparar.")
    # Correspondência por tempo real, sem normalizar duração ou esconder dessincronia.
    rows = []
    used = set()
    for r in ref:
        idx = min(range(len(obs)), key=lambda j: abs(obs[j]["time_seconds"] - r["time_seconds"]))
        o = obs[idx]
        gap = abs(o["time_seconds"] - r["time_seconds"])
        usable = idx not in used and gap <= 0.12 and r["angle_degrees"] is not None and o["angle_degrees"] is not None
        used.add(idx)
        delta = abs(o["angle_degrees"] - r["angle_degrees"]) if usable else None
        rows.append({
            "reference_seconds": r["time_seconds"], "observed_seconds": o["time_seconds"],
            "reference_angle": r["angle_degrees"], "observed_angle": o["angle_degrees"],
            "delta_degrees": round(delta, 4) if delta is not None else None,
            "reference_quality": r["quality_min"], "observed_quality": o["quality_min"],
            "reason": None if usable else (r["reason"] or o["reason"] or "tempo_sem_correspondencia"),
        })
    valid = sum(row["delta_degrees"] is not None for row in rows)
    coverage = valid / max(len(ref), len(obs))
    analyzable = valid >= 10 and coverage >= 0.8
    events, run = [], []

    def finish():
        if run:
            span = run[-1]["observed_seconds"] - run[0]["observed_seconds"]
            if span + 1e-6 >= options.min_event_seconds:
                events.append({
                    "start_seconds": run[0]["observed_seconds"],
                    "end_seconds": run[-1]["observed_seconds"],
                    "sampled_span_seconds": round(span, 6),
                    "max_delta_degrees": max(row["delta_degrees"] for row in run),
                    "samples": len(run),
                })
            run.clear()

    if analyzable:
        for row in rows:
            if row["delta_degrees"] is not None and row["delta_degrees"] > options.tolerance_degrees:
                if run and row["observed_seconds"] - run[-1]["observed_seconds"] > 0.3:
                    finish()
                run.append(row)
            else:
                finish()
        finish()
    return {
        "algorithm_version": ALGORITHM_VERSION, "model_sha256": MODEL_SHA256,
        "status": "analisavel" if analyzable else "nao_analisavel",
        "angle_definition": angle_definition(options),
        "options": asdict(options), "valid_pairs": valid, "valid_coverage": round(coverage, 4),
        "duration_delta_seconds": round(duration_delta, 6),
        "reference_metadata": rmeta, "observed_metadata": ometa,
        "events": events, "comparisons": rows,
        "limitations": [
            ("Ombro: ângulo quadril–ombro–cotovelo no plano da imagem, relativo ao tronco. Não mede abdução anatômica 3D nem distingue flexão de abdução."
             if options.joint == "shoulder" else "Cotovelo: ângulo ombro–cotovelo–punho no plano da imagem."),
            "A comparação angular não é avaliação médica nem identifica execução correta de exercício.",
            "Inícios e fases devem ser alinhados pelo usuário. Diferenças de câmera, ritmo ou oclusão confundem o resultado.",
            "Amostragem de 5 Hz não observa todos os instantes. Durações são intervalos entre amostras, não eventos clínicos contínuos.",
            "Visibility/presence são filtros do estimador, não probabilidade de acerto nem prova de ausência de oclusão.",
            "Sem desvios detectados não significa movimento correto, segurança ou saúde.",
        ],
    }


def analyze_pair(reference_bytes: bytes, observed_bytes: bytes, model_path: Path, options=Options()):
    """Cada decodificação/inferência ocorre em processo descartável com limite de tempo.

    Somente caminhos criados pela aplicação entram em argv; nomes enviados são ignorados.
    Não é um sandbox de segurança para arquivos deliberadamente hostis.
    """
    options.validate()
    validate_upload(reference_bytes)
    validate_upload(observed_bytes)
    verify_model(model_path)
    executable, separate = worker_python()
    if separate:
        verify_worker_python(executable)
    worker = Path(__file__).with_name("video_movimento_worker.py")
    results = []
    with tempfile.TemporaryDirectory(prefix="video-local-") as directory:
        root = Path(directory)
        # Cópia pequena verificada: evita mudanças no modelo entre validação e inferência.
        local_model = root / MODEL_FILENAME
        local_model.write_bytes(Path(model_path).read_bytes())
        verify_model(local_model)
        (root / "options.json").write_text(json.dumps(asdict(options)), encoding="utf-8")
        for tag, data in (("reference", reference_bytes), ("observed", observed_bytes)):
            (root / "input.mp4").write_bytes(data)
            output = root / "result.json"
            output.unlink(missing_ok=True)
            try:
                done = subprocess.run(
                    [executable, "-I", str(worker), str(root)],
                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    shell=False, timeout=PROCESS_TIMEOUT, check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise VideoError("Tempo limite de 45 s excedido em um clipe; análise interrompida.") from exc
            except OSError as exc:
                raise VideoError("Não foi possível iniciar a análise local.") from exc
            if not output.is_file() or output.stat().st_size > MAX_RESULT_BYTES:
                raise VideoError("O processo de vídeo não retornou um resultado válido.")
            try:
                decoded = json.loads(output.read_text(encoding="utf-8"))
            except (ValueError, OSError) as exc:
                raise VideoError("Resposta do processo de vídeo inválida.") from exc
            if done.returncode != 0 or "error" in decoded:
                raise VideoError(decoded.get("error", "Falha na análise local de vídeo."))
            results.append(decoded)
        report = compare(results[0], results[1], options)
        report["reference_sha256"] = hashlib.sha256(reference_bytes).hexdigest()
        report["observed_sha256"] = hashlib.sha256(observed_bytes).hexdigest()
        report["same_input_control"] = report["reference_sha256"] == report["observed_sha256"]
        report["runtime"] = results[0].get("runtime", {})
        report["runtime"]["interpreter_mode"] = "separate" if separate else "current"
        return report
