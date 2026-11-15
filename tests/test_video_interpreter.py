"""Configuração somente de servidor, sem executar código fornecido por upload."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import video_movimento_local as video
from test_video_movimento_local import MP4_HEADER, clip


class InterpreterTests(unittest.TestCase):
    def test_unset_retains_current_and_existing_symlink_preserves_venv(self):
        with patch.dict(os.environ, {"FIAP_VIDEO_PYTHON": ""}):
            self.assertEqual(video.worker_python(), (sys.executable, False))
        with patch.dict(os.environ, {"FIAP_VIDEO_PYTHON": sys.executable}):
            self.assertEqual(video.worker_python(), (sys.executable, True))

    def test_relative_unc_url_missing_and_invalid_file_rejected(self):
        for value in ("python", "./python", r"\\server\share\python.exe", "//server/share/python", "https://host/python", "/missing/python"):
            with self.subTest(value=value), patch.dict(os.environ, {"FIAP_VIDEO_PYTHON": value}):
                with self.assertRaises(video.VideoError) as error: video.worker_python()
                self.assertNotIn(value, str(error.exception))
        with patch.object(video.os, "environ", {"FIAP_VIDEO_PYTHON": "a\0b"}):
            with self.assertRaises(video.VideoError): video.worker_python()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"python"
            path.write_text("not an executable")
            with patch.dict(os.environ, {"FIAP_VIDEO_PYTHON": str(path)}), patch.object(video.os,"access", return_value=False):
                with self.assertRaises(video.VideoError): video.worker_python()
            invalid=Path(directory)/"other-command";invalid.write_text("not Python");invalid.chmod(0o700)
            with patch.dict(os.environ, {"FIAP_VIDEO_PYTHON": str(invalid)}):
                with self.assertRaises(video.VideoError): video.worker_python()

    def test_probe_pins_protocol_timeout_and_malformed_output(self):
        good={"implementation":"cpython","version":[3,11],"bits":64,"conflicts":[],"venv":True,"packages":video.WORKER_PACKAGES}
        with patch.object(video.subprocess,"run",return_value=SimpleNamespace(returncode=0,stdout=json.dumps(good).encode())) as run:
            video.verify_worker_python(sys.executable)
            args,kwargs=run.call_args
            self.assertEqual(args[0], [sys.executable,"-I","-c",video.WORKER_PROBE])
            self.assertFalse(kwargs["shell"]);self.assertEqual(kwargs["timeout"],10)
        bad=[good|{"version":[3,14]},good|{"bits":32},good|{"conflicts":["opencv-python"]},good|{"venv":False},good|{"packages":video.WORKER_PACKAGES|{"numpy":"2.4.4"}},[],None]
        for item in bad:
            with patch.object(video.subprocess,"run",return_value=SimpleNamespace(returncode=0,stdout=json.dumps(item).encode())):
                with self.assertRaises(video.VideoError): video.verify_worker_python(sys.executable)
        for output in (b"bad",b"x"*4097):
            with patch.object(video.subprocess,"run",return_value=SimpleNamespace(returncode=0,stdout=output)):
                with self.assertRaises(video.VideoError): video.verify_worker_python(sys.executable)
        with patch.object(video.subprocess,"run",side_effect=subprocess.TimeoutExpired([sys.executable],10)):
            with self.assertRaises(video.VideoError): video.verify_worker_python(sys.executable)
        with patch.object(video.subprocess,"run",side_effect=OSError("private executable detail")):
            with self.assertRaises(video.VideoError) as error: video.verify_worker_python(sys.executable)
            self.assertNotIn("private executable detail",str(error.exception))
        with patch.object(video.subprocess,"run",return_value=SimpleNamespace(returncode=1,stdout=b"")):
            with self.assertRaises(video.VideoError): video.verify_worker_python(sys.executable)

    def test_default_model_and_invalid_server_paths(self):
        with patch.dict(os.environ,{"FIAP_POSE_MODEL_PATH":""}):
            path=video.local_model_path()
            self.assertEqual(path.parts[-3:],("models","pose-landmarker-lite",video.MODEL_FILENAME))
        for raw in ("model.task","https://host/model.task",r"\\server\share\model.task"):
            with patch.dict(os.environ,{"FIAP_POSE_MODEL_PATH":raw}):
                with self.assertRaises(video.VideoError):video.local_model_path()

    def test_selected_interpreter_is_used_and_no_paths_in_report(self):
        def run(args,**kwargs):
            self.assertEqual(args[:2],[sys.executable,"-I"])
            self.assertFalse(kwargs['shell']);self.assertEqual(kwargs['timeout'],45)
            (Path(args[-1])/"result.json").write_text(json.dumps(clip()))
            return SimpleNamespace(returncode=0)
        with tempfile.TemporaryDirectory() as directory:
            model=Path(directory)/"model.task";model.write_bytes(b"test")
            with patch.dict(os.environ,{"FIAP_VIDEO_PYTHON":sys.executable}), patch.object(video,"verify_model"), patch.object(video,"verify_worker_python") as probe, patch.object(video.subprocess,"run",side_effect=run):
                report=video.analyze_pair(MP4_HEADER,MP4_HEADER,model)
                probe.assert_called_once_with(sys.executable)
                self.assertEqual(report['runtime']['interpreter_mode'],'separate')
                self.assertNotIn(directory,json.dumps(report));self.assertNotIn(sys.executable,json.dumps(report))


if __name__ == "__main__": unittest.main()
