"""Streamlit 1.32 AppTest; upload boundary injected, ASR explicitly mocked.

Native browser upload and genuine spoken-audio inference remain separate checks.
"""

import io
from pathlib import Path
import sys
import unicodedata
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from audio_local import AudioError, Segment, Transcript, validate_wav
from test_audio_local import wav_bytes

try:
    from streamlit.testing.v1 import AppTest
except ImportError:
    AppTest = None

PAGE = Path(__file__).resolve().parents[1] / "app/pages/2_Audio_Local.py"


class Upload(io.BytesIO):
    @property
    def size(self):
        return len(self.getbuffer())


@unittest.skipIf(AppTest is None, "Streamlit 1.32 não instalado.")
class AudioPageTests(unittest.TestCase):
    def setUp(self):
        import streamlit as st
        original = st.get_option
        guard = patch("streamlit.get_option", side_effect=lambda name: "127.0.0.1" if name == "server.address" else original(name))
        guard.start()
        self.addCleanup(guard.stop)
        self.payload = wav_bytes()
        self.result = Transcript((Segment(0, 0.9, "Sem dor no peito. Estou bem."),), "pt", validate_wav(self.payload).fingerprint)

    def app(self):
        app = AppTest.from_file(str(PAGE), default_timeout=10).run()
        self.assertFalse(app.exception)
        return app

    def validate(self, app):
        app.sidebar.button(key="_audio_validate").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.main.metric), 3)
        self.assertEqual(len(app.sidebar.metric), 0)
        return app

    def run_mock(self, app):
        with patch("audio_local.transcribe_local", return_value=self.result) as inference:
            app.button(key="_audio_run").click().run()
            inference.assert_called_once()
        self.assertFalse(app.exception)
        self.assertTrue(any(item.value == "Transcrição local" for item in app.subheader))
        return app

    def test_initial_local_file_input_no_fabricated_results(self):
        app = self.app()
        self.assertEqual(len(app.get("file_uploader")), 1)
        self.assertEqual(len(app.metric), 0)
        self.assertFalse(any(item.value == "Transcrição local" for item in app.subheader))
        self.assertFalse(any(button.key == "_audio_run" for button in app.button))

    def test_visible_copy_help_and_safe_conversion_command(self):
        def assert_copy(app):
            parts = []
            for kind in ["title", "header", "subheader", "caption", "info", "warning", "error", "markdown", "code"]:
                parts.extend(str(element.value) for element in app.get(kind))
            for kind in ["file_uploader", "text_input", "selectbox", "button"]:
                for element in app.get(kind):
                    parts.append(str(getattr(element.proto, "help", "")))
                    parts.append(str(getattr(element.proto, "label", "")))
            normalized = "".join(character for character in unicodedata.normalize("NFKD", " ".join(parts).lower())
                                 if not unicodedata.combining(character))
            self.assertNotRegex(normalized, r"fictici\w*")
            self.assertTrue(any(item.value == "Uso acadêmico. Use apenas gravações de teste, sem dados identificáveis."
                                for item in app.caption))
            command = next(item.value for item in app.code if item.value.startswith("ffmpeg "))
            self.assertIn("ffmpeg -n -i trecho.wav", command)
            self.assertNotIn(" -t ", command)
            self.assertNotIn(" -y", command)

        app = self.app()
        assert_copy(app)
        with patch("streamlit.sidebar.file_uploader", return_value=Upload(self.payload)):
            app.run()
            self.validate(app)
            self.run_mock(app)
            assert_copy(app)

    def test_nonlocal_and_unset_bind_hide_uploader(self):
        for bind in ["0.0.0.0", None, "localhost", "192.168.1.2"]:
            with self.subTest(bind=bind), patch("streamlit.get_option", return_value=bind):
                app = self.app()
                self.assertEqual(len(app.get("file_uploader")), 0)
                self.assertFalse(any(button.key == "_audio_run" for button in app.button))

    def test_valid_preview_before_mock_inference_sidebar_layout(self):
        with patch("streamlit.sidebar.file_uploader", return_value=Upload(self.payload)):
            app = self.validate(self.app())
            self.assertEqual(len(app.get("audio")), 1)
            self.assertEqual(len(app.main.button), 0)
            self.run_mock(app)
            self.assertEqual(len(app.dataframe), 2)

    def test_invalid_upload_no_audio_or_inference(self):
        with patch("streamlit.sidebar.file_uploader", return_value=Upload(b"ID3-not-wav")):
            app = self.app()
            app.button(key="_audio_validate").click().run()
            self.assertEqual(len(app.error), 1)
            self.assertEqual(len(app.get("audio")), 0)
            self.assertFalse(any(button.key == "_audio_run" for button in app.button))
            self.assertFalse(app.exception)

    def test_lfs_pointer_is_explained_without_preview_or_inference(self):
        pointer = (b"version https://git-lfs.github.com/spec/v1\n"
                   b"oid sha256:" + b"a" * 64 + b"\nsize 12345678\n")
        with patch("streamlit.sidebar.file_uploader", return_value=Upload(pointer)):
            app = self.app()
            app.button(key="_audio_validate").click().run()
            self.assertEqual(len(app.error), 1)
            self.assertIn("ponteiro Git LFS", app.error[0].value)
            self.assertEqual(len(app.get("audio")), 0)
            self.assertFalse(any(button.key == "_audio_run" for button in app.button))
            self.assertFalse(app.exception)

    def test_audio_navigation_uses_shared_native_menu(self):
        from streamlit.source_util import get_pages
        root = PAGE.parents[1]
        pages = get_pages(str(root / "app.py"))
        pages = dict(sorted(pages.items(), key=lambda item: item[1]["script_path"] != str(PAGE)))
        with patch("streamlit.elements.widgets.button.get_main_script_directory", return_value=str(root)), \
             patch("streamlit.elements.widgets.button.source_util.get_pages", return_value=pages):
            app = self.app()
        links = app.sidebar.get("page_link")
        self.assertEqual([link.proto.label for link in links], ["Predição de risco", "Monitoramento", "Áudio"])
        self.assertEqual([link.proto.icon for link in links], ["🫀", "📈", "🔊"])
        self.assertEqual([link.proto.page for link in links], ["app", "Monitoramento_Didatico", "Audio_Local"])
        self.assertEqual(app.title[0].value, "🔊 Áudio")

    def test_oversize_rejected_without_getvalue(self):
        class TooLarge:
            size = 2 * 1024 * 1024 + 1

            def getvalue(self):
                raise AssertionError("Oversize content must not be read")

        with patch("streamlit.sidebar.file_uploader", return_value=TooLarge()):
            app = self.app()
            self.assertEqual(len(app.error), 1)
            self.assertEqual(len(app.metric), 0)

    def test_digital_silence_control_is_not_asr(self):
        with patch("streamlit.sidebar.file_uploader", return_value=Upload(wav_bytes(value=0))):
            app = self.validate(self.app())
            self.assertTrue(any("silêncio digital" in item.value for item in app.info))
            self.assertFalse(any(button.key == "_audio_run" for button in app.button))

    def test_missing_model_honest_error_with_no_fake_text(self):
        with patch("streamlit.sidebar.file_uploader", return_value=Upload(self.payload)):
            app = self.validate(self.app())
            app.text_input(key="_audio_model").set_value("/missing-audio-model").run()
            self.validate(app)
            app.button(key="_audio_run").click().run()
            self.assertEqual(len(app.error), 1)
            self.assertTrue("não encontrada" in app.error[0].value)
            self.assertNotIn("_audio_result", app.session_state)
            self.assertFalse(app.exception)

    def test_repeated_run_revalidation_clear_and_remove(self):
        with patch("streamlit.sidebar.file_uploader", return_value=Upload(self.payload)):
            app = self.validate(self.app())
            self.run_mock(app)
            self.run_mock(app)
            app.button(key="_audio_validate").click().run()
            self.assertNotIn("_audio_result", app.session_state)
            self.run_mock(app)
        with patch("streamlit.sidebar.file_uploader", return_value=None):
            app.run()
            self.assertNotIn("_audio_result", app.session_state)
            self.assertEqual(len(app.metric), 0)
        with patch("streamlit.sidebar.file_uploader", return_value=Upload(self.payload)):
            app.run()
            self.validate(app)
            self.run_mock(app)
            app.button(key="_audio_clear").click().run()
            self.assertNotIn("_audio_result", app.session_state)
            self.assertNotIn("_audio_preview", app.session_state)
            self.assertEqual(len(app.metric), 0)

    def test_model_language_and_content_changes_invalidate(self):
        with patch("streamlit.sidebar.file_uploader", return_value=Upload(self.payload)):
            app = self.validate(self.app())
            self.run_mock(app)
            app.text_input(key="_audio_model").set_value("/other-model").run()
            self.assertNotIn("_audio_result", app.session_state)
            self.assertEqual(len(app.metric), 0)
            self.validate(app)
            self.run_mock(app)
            app.selectbox(key="_audio_language").select("Inglês").run()
            self.assertNotIn("_audio_result", app.session_state)
            self.assertEqual(len(app.metric), 0)
            self.validate(app)
            self.run_mock(app)
        with patch("streamlit.sidebar.file_uploader", return_value=Upload(wav_bytes(value=200))):
            app.run()
            self.assertNotIn("_audio_result", app.session_state)
            self.assertEqual(len(app.metric), 0)

    def test_invalid_replaces_prior_valid_without_stale_results(self):
        with patch("streamlit.sidebar.file_uploader", return_value=Upload(self.payload)):
            app = self.validate(self.app())
            self.run_mock(app)
        with patch("streamlit.sidebar.file_uploader", return_value=Upload(b"not wav")):
            app.run()
            self.assertNotIn("_audio_result", app.session_state)
            self.assertEqual(len(app.metric), 0)
            app.button(key="_audio_validate").click().run()
            self.assertEqual(len(app.error), 1)

    def test_failed_repeat_discards_old_result_success_clears_error(self):
        with patch("streamlit.sidebar.file_uploader", return_value=Upload(self.payload)):
            app = self.validate(self.app())
            self.run_mock(app)
            with patch("audio_local.transcribe_local", side_effect=AudioError("Falha controlada de teste")):
                app.button(key="_audio_run").click().run()
            self.assertNotIn("_audio_result", app.session_state)
            self.assertEqual(len(app.error), 1)
            self.run_mock(app)
            self.assertEqual(len(app.error), 0)

    def test_bind_change_clears_prior_upload_and_result(self):
        with patch("streamlit.sidebar.file_uploader", return_value=Upload(self.payload)):
            app = self.validate(self.app())
            self.run_mock(app)
            with patch("streamlit.get_option", return_value="0.0.0.0"):
                app.run()
                self.assertNotIn("_audio_result", app.session_state)
                self.assertNotIn("_audio_preview", app.session_state)
                self.assertEqual(len(app.get("audio")), 0)


if __name__ == "__main__":
    unittest.main()
