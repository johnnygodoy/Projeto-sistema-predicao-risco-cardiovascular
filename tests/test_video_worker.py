"""Guardas antes da importação das bibliotecas opcionais."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import video_movimento_worker as worker
from video_movimento_local import VideoError


class WorkerGuardTests(unittest.TestCase):
    def test_rejects_newer_sdk_before_inference(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(worker, "version", return_value="1.2.3"):
            with self.assertRaisesRegex(VideoError, "0.10.21"):
                worker.decode(Path(directory))

    def test_missing_optional_sdk(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(worker, "version", side_effect=worker.PackageNotFoundError):
            with self.assertRaisesRegex(VideoError, "Dependências de vídeo"):
                worker.decode(Path(directory))


if __name__ == "__main__": unittest.main()
