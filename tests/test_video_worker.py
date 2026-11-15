"""Guardas antes da importação das bibliotecas opcionais."""
import builtins
import json
from types import SimpleNamespace
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

    def test_long_input_rejected_before_frames_or_mediapipe_import(self):
        from unittest.mock import Mock
        capture=Mock();capture.isOpened.return_value=True
        values={1:1080,2:1920,3:30,4:1304}
        capture.get.side_effect=lambda key:values[key]
        cv2=SimpleNamespace(setNumThreads=Mock(),VideoCapture=Mock(return_value=capture),CAP_FFMPEG=100,
                            CAP_PROP_FRAME_WIDTH=1,CAP_PROP_FRAME_HEIGHT=2,CAP_PROP_FPS=3,CAP_PROP_FRAME_COUNT=4)
        original_import=builtins.__import__;imports=[]
        def guarded_import(name,*args,**kwargs):
            if name=='mediapipe':
                imports.append(name);raise AssertionError('Model must not load for long input')
            return original_import(name,*args,**kwargs)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'options.json').write_text('{}')
            with patch.object(worker,'version',return_value='0.10.21'),patch.dict(sys.modules,{'cv2':cv2,'numpy':SimpleNamespace()}),patch('builtins.__import__',side_effect=guarded_import),patch.object(worker,'verify_model') as verify:
                with self.assertRaisesRegex(VideoError,'43,47 s'):
                    worker.decode(root)
            verify.assert_not_called()
        capture.read.assert_not_called();capture.release.assert_called_once();self.assertEqual(imports,[])

    def test_missing_optional_sdk(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(worker, "version", side_effect=worker.PackageNotFoundError):
            with self.assertRaisesRegex(VideoError, "Dependências de vídeo"):
                worker.decode(Path(directory))


if __name__ == "__main__": unittest.main()
