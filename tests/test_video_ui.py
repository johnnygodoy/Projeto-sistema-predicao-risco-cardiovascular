"""AppTest com uploads/ML simulados só na fronteira; não valida vídeo humano."""
from contextlib import ExitStack
import io
import os
from pathlib import Path
import sys
import unicodedata
import unittest
from unittest.mock import patch

try:
    from streamlit.testing.v1 import AppTest
except ImportError:
    AppTest = None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
import video_movimento_local as video
from test_video_movimento_local import MP4_HEADER, clip


class Upload(io.BytesIO):
    @property
    def size(self): return len(self.getbuffer())


@unittest.skipIf(AppTest is None, "Streamlit/AppTest opcional não instalado.")
class PageTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.stack.enter_context(patch.dict(os.environ, {"FIAP_VIDEO_PYTHON": "", "FIAP_POSE_MODEL_PATH": ""}))
        import streamlit
        original = streamlit.get_option
        def config(name):
            return {"server.address": "127.0.0.1", "server.maxUploadSize": 20}.get(name, original(name))
        self.stack.enter_context(patch("streamlit.get_option", side_effect=config))

    def tearDown(self): self.stack.close()

    def page(self):
        app = AppTest.from_file(str(ROOT / "app/pages/4_Video_Movimento_Local.py"), default_timeout=10).run()
        self.assertFalse(app.exception)
        return app

    def declared(self):
        app = self.page()
        app.checkbox(key="_vid_declared").check().run()
        self.assertFalse(app.exception)
        return app

    def uploads(self, value=None):
        return patch("streamlit.sidebar.file_uploader", side_effect=lambda *a, **k: Upload(MP4_HEADER) if value is None else value)

    def test_initial_and_model_absent(self):
        app = self.page()
        self.assertEqual(len(app.get("file_uploader")), 0)
        app.checkbox(key="_vid_declared").check().run()
        self.assertEqual(len(app.get("file_uploader")), 2)
        self.assertTrue(app.button(key="_vid_run").disabled)
        self.assertFalse(app.exception)

    def test_native_video_navigation_and_clean_copy(self):
        from streamlit.source_util import get_pages
        page = ROOT / "app/pages/4_Video_Movimento_Local.py"
        pages = get_pages(str(ROOT / "app/app.py"))
        pages = dict(sorted(pages.items(), key=lambda item: item[1]["script_path"] != str(page)))
        with patch("streamlit.elements.widgets.button.get_main_script_directory", return_value=str(ROOT / "app")), \
             patch("streamlit.elements.widgets.button.source_util.get_pages", return_value=pages):
            app = self.page()
        links = app.sidebar.get("page_link")
        self.assertEqual([item.proto.label for item in links], ["Predição de risco", "Monitoramento", "Áudio", "Vídeo"])
        self.assertEqual([item.proto.icon for item in links], ["🫀", "📈", "🔊", "🎥"])
        self.assertEqual([item.proto.page for item in links], ["app", "Monitoramento_Didatico", "Audio_Local", "Video_Movimento_Local"])
        self.assertEqual(app.title[0].value, "🎥 Vídeo")
        parts = [str(item.value) for kind in ("title", "header", "subheader", "caption", "info", "warning", "error", "markdown", "code") for item in app.get(kind)]
        for kind in ("checkbox", "text_input", "file_uploader", "button", "selectbox", "slider"):
            for item in app.get(kind):
                parts.extend([str(getattr(item.proto, "label", "")), str(getattr(item.proto, "help", ""))])
        text = "".join(c for c in unicodedata.normalize("NFKD", " ".join(parts).lower()) if not unicodedata.combining(c))
        self.assertNotRegex(text, r"fictici\w*")
        self.assertEqual(text.count("uso academico"), 1)

    def test_model_change_and_bad_model_path_clear_results(self):
        report = video.compare(clip(), clip(), video.Options())
        with self.uploads(), patch.object(video, "verify_model"), patch.object(video, "analyze_pair", return_value=report):
            app = self.declared()
            app.button(key="_vid_run").click().run()
            self.assertEqual(len(app.metric), 3)
            self.assertEqual(len(app.text_input), 0)
            with patch.dict(os.environ, {"FIAP_POSE_MODEL_PATH": "https://example.invalid/model.task"}):
                app.run()
                self.assertEqual(len(app.metric), 0)
                self.assertTrue(app.button(key="_vid_run").disabled)
                self.assertTrue(any("Configuração do modelo local inválida" in item.value for item in app.info))
                self.assertFalse(any("example.invalid" in item.value for item in app.info))
            self.assertFalse(app.exception)

    def test_server_worker_change_clears_without_showing_path(self):
        report = video.compare(clip(), clip(), video.Options())
        with self.uploads(), patch.object(video, "verify_model"), patch.object(video, "analyze_pair", return_value=report):
            app = self.declared()
            app.button(key="_vid_run").click().run()
            self.assertEqual(len(app.metric), 3)
            with patch.dict(os.environ, {"FIAP_VIDEO_PYTHON": "/private/example/not-python"}):
                app.run()
                self.assertEqual(len(app.metric), 0)
                self.assertTrue(app.button(key="_vid_run").disabled)
                self.assertFalse(any("/private/" in item.value for item in app.info))
                self.assertEqual(len(app.text_input), 0)
                self.assertFalse(app.exception)

    def test_joint_and_side_change_clear_and_reach_analysis(self):
        report = video.compare(clip(), clip(), video.Options())
        report["same_input_control"] = True
        with self.uploads(), patch.object(video, "verify_model"), patch.object(video, "analyze_pair", return_value=report) as analyze:
            app = self.declared()
            self.assertEqual(app.selectbox(key="_vid_joint").value, "Cotovelo")
            app.button(key="_vid_run").click().run()
            self.assertTrue(any("Controle com o mesmo arquivo" in item.value for item in app.info))
            app.selectbox(key="_vid_joint").set_value("Ombro (abdução no plano da imagem)").run()
            self.assertEqual(len(app.metric), 0)
            app.button(key="_vid_run").click().run()
            self.assertEqual(analyze.call_args.args[3].joint, "shoulder")
            app.selectbox(key="_vid_side").set_value("Esquerdo").run()
            self.assertEqual(len(app.metric), 0)
            app.button(key="_vid_run").click().run()
            self.assertEqual(analyze.call_args.args[3].side, "left")
            self.assertFalse(app.exception)

    def test_short_clips_limit_and_control_instructions_visible(self):
        app = self.page()
        text = " ".join(str(item.value) for kind in ("markdown", "caption", "info") for item in app.get(kind))
        self.assertIn("Use MP4 de 2 a 15 s, até 720p.", text)
        self.assertIn("Não há conversão automática.", text)
        self.assertNotIn("docs/VIDEO_LOCAL.md", text)
        self.assertNotIn("43,5", text)
        self.assertIn("REHAB24-6 Ex1", text)
        self.assertIn("mesmo clipe curto", text)

    def test_refuse_remote_or_unbounded_upload(self):
        import streamlit
        original = streamlit.get_option
        with patch("streamlit.get_option", side_effect=lambda name: "0.0.0.0" if name == "server.address" else original(name)):
            app = self.page()
            self.assertEqual(len(app.get("file_uploader")), 0)
            self.assertEqual(len(app.checkbox), 0)

        with patch("streamlit.get_option", side_effect=lambda name: 200 if name == "server.maxUploadSize" else original(name)):
            app = self.page()
            self.assertEqual(len(app.get("file_uploader")), 0)
            self.assertEqual(len(app.checkbox), 0)

    def test_shared_audio_upload_cap_preserves_video_cap(self):
        import streamlit
        original = streamlit.get_option
        with patch("streamlit.get_option", side_effect=lambda name: 24 if name == "server.maxUploadSize" else original(name)):
            app = self.declared()
            self.assertEqual(len(app.get("file_uploader")), 2)
            class Oversize:
                size = video.MAX_BYTES + 1
                def getvalue(self): raise AssertionError("Oversize must be rejected before copying")
            with self.uploads(Oversize()), patch.object(video, "analyze_pair") as analyze:
                app.run()
                self.assertTrue(any("20 MiB" in item.value for item in app.error))
                analyze.assert_not_called()

    def test_success_repeat_and_option_change_clear(self):
        report = video.compare(clip(), clip(125), video.Options())
        with self.uploads(), patch.object(video, "verify_model"), patch.object(video, "analyze_pair", return_value=report) as analyze:
            app = self.declared()
            app.button(key="_vid_run").click().run()
            self.assertEqual(len(app.metric), 3)
            self.assertFalse(app.exception)
            app.button(key="_vid_run").click().run()
            self.assertEqual(analyze.call_count, 2)
            app.slider(key="_vid_tolerance").set_value(25).run()
            self.assertEqual(len(app.metric), 0)

    def test_withdraw_clear_and_replace_file_remove_results(self):
        report = video.compare(clip(), clip(), video.Options())
        with self.uploads(), patch.object(video, "verify_model"), patch.object(video, "analyze_pair", return_value=report):
            app = self.declared()
            app.button(key="_vid_run").click().run()
            self.assertEqual(len(app.metric), 3)
            with self.uploads(Upload(MP4_HEADER + b"changed")):
                app.run()
                self.assertEqual(len(app.metric), 0)
            app.button(key="_vid_run").click().run()
            app.checkbox(key="_vid_declared").uncheck().run()
            self.assertEqual(len(app.metric), 0)
            self.assertEqual(len(app.get("file_uploader")), 0)
            app.button(key="_vid_clear").click().run()
            self.assertFalse(app.checkbox(key="_vid_declared").value)
            self.assertFalse(app.exception)

    def test_failure_and_no_pose_not_success(self):
        with self.uploads(), patch.object(video, "verify_model"), patch.object(video, "analyze_pair", side_effect=video.VideoError("Tempo limite.")):
            app = self.declared()
            app.button(key="_vid_run").click().run()
            self.assertEqual(len(app.error), 1)
            self.assertEqual(len(app.metric), 0)
            self.assertFalse(app.exception)
        report = video.compare(clip(), clip(invalid=tuple(range(15))), video.Options())
        with self.uploads(), patch.object(video, "verify_model"), patch.object(video, "analyze_pair", return_value=report):
            app = self.declared()
            app.button(key="_vid_run").click().run()
            self.assertTrue(any("Não analisável" in item.value for item in app.warning))
            self.assertFalse(app.exception)

    def test_invalid_upload_no_analysis(self):
        with self.uploads(Upload(b"invalid")):
            app = self.declared()
            self.assertEqual(len(app.error), 1)
            self.assertEqual(len(app.metric), 0)
            self.assertFalse(app.exception)


if __name__ == "__main__": unittest.main()
