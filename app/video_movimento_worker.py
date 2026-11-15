"""Processo auxiliar fixo: não aceita URLs, nomes originais ou comandos do upload."""
from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError, version
import math
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from video_movimento_local import (  # noqa: E402
    MAX_DECODE_FRAMES, MAX_SAMPLES, MODEL_FILENAME, Options, SAMPLE_HZ,
    VideoError, pose_sample, validate_metadata, verify_model,
)


def decode(root: Path):
    # Cache de bibliotecas também fica no diretório descartável.
    os.environ["MPLCONFIGDIR"] = str(root / "mpl-cache")
    os.environ["XDG_CACHE_HOME"] = str(root / "cache")
    try:
        if version("mediapipe") != "0.10.21":
            raise VideoError("Este candidato exige MediaPipe 0.10.21. Versões posteriores não foram aprovadas para este fluxo local.")
    except PackageNotFoundError as exc:
        raise VideoError("Dependências de vídeo ausentes. Configure o ambiente separado conforme requirements-video-worker.txt.") from exc
    try:
        import cv2
        import mediapipe as mp
        import numpy as np
    except ImportError as exc:
        raise VideoError("Dependências de vídeo ausentes. Configure o ambiente separado conforme requirements-video-worker.txt.") from exc
    options = Options(**json.loads((root / "options.json").read_text(encoding="utf-8")))
    options.validate()
    model = root / MODEL_FILENAME
    verify_model(model)
    cv2.setNumThreads(1)
    capture = cv2.VideoCapture(str(root / "input.mp4"), cv2.CAP_FFMPEG)
    if not capture.isOpened():
        capture.release()
        raise VideoError("MP4 não pôde ser decodificado pelo backend FFmpeg local. Prefira MP4/H.264.")
    try:
        meta = validate_metadata({
            "width": capture.get(cv2.CAP_PROP_FRAME_WIDTH), "height": capture.get(cv2.CAP_PROP_FRAME_HEIGHT),
            "fps": capture.get(cv2.CAP_PROP_FPS), "frames": capture.get(cv2.CAP_PROP_FRAME_COUNT),
        })
        configuration = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model), delegate=mp.tasks.BaseOptions.Delegate.CPU),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_poses=2, min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5, min_tracking_confidence=0.5,
            output_segmentation_masks=False,
        )
        samples, previous, origin, target = [], None, None, 0.0
        decoded_frames, started = 0, time.monotonic()
        with mp.tasks.vision.PoseLandmarker.create_from_options(configuration) as landmarker:
            for _ in range(MAX_DECODE_FRAMES):
                if time.monotonic() - started > 40:
                    raise VideoError("Limite de processamento atingido. Use um clipe menor.")
                ok, frame = capture.read()
                if not ok:
                    break
                decoded_frames += 1
                if frame.shape[:2] != (meta["height"], meta["width"]):
                    raise VideoError("Resolução decodificada não corresponde aos metadados.")
                pts = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                if not math.isfinite(pts) or pts < 0:
                    raise VideoError("Vídeo sem timestamps utilizáveis.")
                if origin is None:
                    origin = pts
                timestamp = pts - origin
                if previous is not None:
                    step = timestamp - previous
                    if step <= 0 or abs(step - 1 / meta["fps"]) > 0.2 / meta["fps"]:
                        raise VideoError("Use vídeo de taxa constante com timestamps crescentes; taxa variável não é suportada.")
                previous = timestamp
                if timestamp > 15.0:
                    raise VideoError("Conteúdo decodificado ultrapassa 15 segundos.")
                if timestamp + 1e-6 >= target:
                    if len(samples) >= MAX_SAMPLES:
                        raise VideoError("Limite de amostras excedido.")
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    result = landmarker.detect_for_video(image, int(round(timestamp * 1000)))
                    samples.append(pose_sample(result, timestamp, meta["width"], meta["height"], options))
                    target += 1 / SAMPLE_HZ
            else:
                raise VideoError("Limite de quadros atingido; arquivo rejeitado, sem análise parcial.")
        if decoded_frames != meta["frames"] or previous is None:
            raise VideoError("Vídeo truncado ou contagem de quadros inconsistente; nenhuma conclusão gerada.")
        actual_duration = previous + 1 / meta["fps"]
        if abs(actual_duration - meta["duration_seconds"]) > 1 / meta["fps"] + 1e-4:
            raise VideoError("Duração e timestamps do vídeo são inconsistentes.")
        meta["duration_seconds"] = round(actual_duration, 6)
        return {"metadata": meta, "samples": samples,
                "runtime": {"python": sys.version.split()[0], "mediapipe": mp.__version__, "opencv": cv2.__version__, "numpy": np.__version__}}
    finally:
        capture.release()


def main():
    if len(sys.argv) != 2:
        return 2
    root = Path(sys.argv[1])
    try:
        result = decode(root)
        code = 0
    except VideoError as exc:
        result, code = {"error": str(exc)}, 1
    except Exception:
        result, code = {"error": "Falha no decodificador ou modelo local. Nenhuma conclusão foi gerada."}, 1
    (root / "result.json").write_text(json.dumps(result, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
