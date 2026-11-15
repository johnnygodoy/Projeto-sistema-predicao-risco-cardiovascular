"""Opt-in: modelo real sobre frames pretos, sem qualquer validação humana.

RUN_VIDEO_MODEL_TESTS=1 habilita; exige dependências e peso local documentados.
"""
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
from video_movimento_local import analyze_pair, VideoError
MODEL = Path(os.environ.get("FIAP_POSE_MODEL_PATH", str(ROOT / "models/pose_landmarker_lite.task")))


@unittest.skipUnless(os.getenv("RUN_VIDEO_MODEL_TESTS") == "1", "Smoke nativo opcional; não valida movimento humano.")
class NativeSmokeTests(unittest.TestCase):
    def test_black_video_is_not_analyzable(self):
        import cv2
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "black.mp4"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (320, 240))
            self.assertTrue(writer.isOpened())
            try:
                for _ in range(60): writer.write(np.zeros((240, 320, 3), dtype=np.uint8))
            finally:
                writer.release()
            data = path.read_bytes()
            report = analyze_pair(data, data, MODEL)
        self.assertEqual(report["status"], "nao_analisavel")
        self.assertEqual(report["valid_pairs"], 0)
        self.assertEqual(report["events"], [])
        self.assertEqual(report["runtime"]["mediapipe"], "0.10.21")

    def test_corrupt_mp4_not_treated_as_normal(self):
        data = (24).to_bytes(4, "big") + b"ftypisom" + b"\0" * 12
        with self.assertRaisesRegex(VideoError, "decodificado"):
            analyze_pair(data, data, MODEL)


if __name__ == "__main__": unittest.main()
