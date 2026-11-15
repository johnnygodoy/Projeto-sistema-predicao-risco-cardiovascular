"""AppTest 1.32: fixtures e estado reais; upload injetado apenas na fronteira do widget.

AppTest 1.32 não fornece upload nativo. O teste manual no navegador continua necessário.
"""

import hashlib
import io
from pathlib import Path
import unittest
from unittest.mock import patch

try:
    from streamlit.testing.v1 import AppTest
except ImportError:
    AppTest = None

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "app/pages/1_Monitoramento_Didatico.py"
SAMPLE = (ROOT / "app/monitoramento_exemplos/variacoes_ficticias.csv").read_bytes()
CONTROL = (ROOT / "app/monitoramento_exemplos/controle_ficticio.csv").read_bytes()


class FakeUpload(io.BytesIO):
    @property
    def size(self):
        return len(self.getbuffer())


@unittest.skipIf(AppTest is None, "Streamlit/AppTest não instalado; testes puros são independentes.")
class PageTests(unittest.TestCase):
    def app(self):
        app = AppTest.from_file(str(PAGE), default_timeout=10).run()
        self.assertFalse(app.exception)
        return app

    def select_sample(self, app, label="Exemplo fictício com variações"):
        app.selectbox(key="_mon_source").select(label).run()
        app.button(key="_mon_validate").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.metric), 0)
        return app

    def test_initial_no_selection_upload_or_result(self):
        app = self.app()
        self.assertEqual(app.selectbox(key="_mon_source").value, "Escolha uma fonte fictícia")
        self.assertEqual(len(app.get("file_uploader")), 0)
        self.assertEqual(len(app.metric), 0)
        self.assertEqual(app.sidebar.selectbox(key="_mon_source").value, "Escolha uma fonte fictícia")

    def test_sidebar_actions_and_main_results_keep_native_layout(self):
        app = self.select_sample(self.app())
        self.assertEqual(len(app.main.button), 0)
        app.sidebar.button(key="_mon_run").click().run()
        self.assertEqual(len(app.main.metric), 3)
        self.assertEqual(len(app.sidebar.metric), 0)
        self.assertTrue(any(header.value == "3. Análise dos sinais" for header in app.main.subheader))
        self.assertFalse(app.exception)

    def test_preview_run_repeat_and_chart_change(self):
        app = self.select_sample(self.app())
        app.button(key="_mon_run").click().run()
        self.assertFalse(app.exception)
        self.assertEqual([m.value for m in app.metric], ["65", "4", "25"])
        app.button(key="_mon_run").click().run()
        self.assertEqual([m.value for m in app.metric], ["65", "4", "25"])
        app.selectbox(key="_mon_chart_signal").select("SpO₂ (%)").run()
        self.assertFalse(app.exception)
        self.assertEqual([m.value for m in app.metric], ["65", "4", "25"])

    def test_source_switch_clears_then_control_has_no_alert(self):
        app = self.select_sample(self.app())
        app.button(key="_mon_run").click().run()
        app.selectbox(key="_mon_source").select("Controle fictício sem variações injetadas").run()
        self.assertEqual(len(app.metric), 0)
        self.assertNotIn("_mon_results", app.session_state)
        app.button(key="_mon_validate").click().run()
        app.button(key="_mon_run").click().run()
        self.assertEqual([m.value for m in app.metric], ["65", "0", "25"])

    def test_native_uploader_only_after_declaration_and_removed_after_withdrawal(self):
        app = self.app()
        app.selectbox(key="_mon_source").select("Enviar CSV fictício").run()
        self.assertEqual(len(app.get("file_uploader")), 0)
        app.checkbox(key="_mon_declared").check().run()
        self.assertEqual(len(app.get("file_uploader")), 1)
        first_generation = app.session_state["_mon_upload_generation"]
        app.checkbox(key="_mon_declared").uncheck().run()
        self.assertEqual(len(app.get("file_uploader")), 0)
        app.checkbox(key="_mon_declared").check().run()
        self.assertGreater(app.session_state["_mon_upload_generation"], first_generation)
        self.assertFalse(app.exception)

    def test_reset_and_revalidation_remove_results(self):
        app = self.select_sample(self.app())
        app.button(key="_mon_run").click().run()
        app.button(key="_mon_validate").click().run()
        self.assertEqual(len(app.metric), 0)
        app.button(key="_mon_run").click().run()
        app.button(key="_mon_clear").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.metric), 0)
        self.assertEqual(app.selectbox(key="_mon_source").value, "Escolha uma fonte fictícia")

    def upload_app(self):
        app = self.app()
        app.selectbox(key="_mon_source").select("Enviar CSV fictício").run()
        app.checkbox(key="_mon_declared").check().run()
        return app

    def test_injected_upload_valid_preview_run_change_remove_and_withdraw(self):
        app = self.upload_app()
        with patch("streamlit.sidebar.file_uploader", return_value=FakeUpload(SAMPLE)):
            app.run()
            app.button(key="_mon_validate").click().run()
            self.assertEqual(len(app.metric), 0)
            app.button(key="_mon_run").click().run()
            self.assertEqual([m.value for m in app.metric], ["65", "4", "25"])
        with patch("streamlit.sidebar.file_uploader", return_value=FakeUpload(CONTROL)):
            app.run()
            self.assertEqual(len(app.metric), 0)
            app.button(key="_mon_validate").click().run()
            app.button(key="_mon_run").click().run()
            self.assertEqual([m.value for m in app.metric], ["65", "0", "25"])
        with patch("streamlit.sidebar.file_uploader", return_value=None):
            app.run()
            self.assertEqual(len(app.metric), 0)
        with patch("streamlit.sidebar.file_uploader", return_value=FakeUpload(SAMPLE)):
            app.run()
            app.button(key="_mon_validate").click().run()
            app.button(key="_mon_run").click().run()
            app.checkbox(key="_mon_declared").uncheck().run()
            self.assertEqual(len(app.metric), 0)
            self.assertNotIn("_mon_results", app.session_state)
        self.assertFalse(app.exception)

    def test_injected_invalid_csv_rejects_without_run(self):
        app = self.upload_app()
        with patch("streamlit.sidebar.file_uploader", return_value=FakeUpload(b"name,value\nPERSON,1\n")):
            app.run()
            app.button(key="_mon_validate").click().run()
            self.assertEqual(len(app.error), 1)
            self.assertEqual(len(app.metric), 0)
            self.assertFalse(any(b.key == "_mon_run" for b in app.button))
        self.assertFalse(app.exception)

    def test_injected_invalid_replaces_valid_results(self):
        app = self.upload_app()
        with patch("streamlit.sidebar.file_uploader", return_value=FakeUpload(SAMPLE)):
            app.run()
            app.button(key="_mon_validate").click().run()
            app.button(key="_mon_run").click().run()
            self.assertEqual(len(app.metric), 3)
        with patch("streamlit.sidebar.file_uploader", return_value=FakeUpload(b"bad csv")):
            app.run()
            self.assertEqual(len(app.metric), 0)
            app.button(key="_mon_validate").click().run()
            self.assertEqual(len(app.error), 1)
            self.assertNotIn("_mon_results", app.session_state)
        self.assertFalse(app.exception)

    def test_injected_insufficient_history_is_explicit(self):
        app = self.upload_app()
        tiny = b"timestamp,signal,value,unit\n2026-01-01T12:00:00Z,heart_rate,72,bpm\n"
        with patch("streamlit.sidebar.file_uploader", return_value=FakeUpload(tiny)):
            app.run()
            app.button(key="_mon_validate").click().run()
            self.assertTrue(any("histórico insuficiente" in warning.value for warning in app.warning))
            app.button(key="_mon_run").click().run()
            self.assertEqual([m.value for m in app.metric], ["0", "0", "1"])
            self.assertTrue(any("Nenhuma comparação" in info.value for info in app.info))
        self.assertFalse(app.exception)

    def test_injected_oversize_rejected_before_reading_bytes(self):
        class Oversize:
            size = 256 * 1024 + 1

            def getvalue(self):
                raise AssertionError("Não deveria ler bytes excessivos")

        app = self.upload_app()
        with patch("streamlit.sidebar.file_uploader", return_value=Oversize()):
            app.run()
            self.assertEqual(len(app.error), 1)
            self.assertFalse(app.exception)
            self.assertFalse(any(b.key == "_mon_validate" for b in app.button))


class OriginalAppTests(unittest.TestCase):
    def test_original_prediction_source_unmodified(self):
        # Git pode converter LF em CRLF no checkout Windows. Comparar conteúdo canônico.
        payload = (ROOT / "app/app.py").read_bytes().replace(b"\r\n", b"\n")
        blob = b"blob " + str(len(payload)).encode() + b"\0" + payload
        self.assertEqual(hashlib.sha1(blob).hexdigest(), "90cb26669213b89e986def0b636b985b068176e8")

    @unittest.skipIf(AppTest is None, "Streamlit não instalado")
    def test_streamlit_132_discovers_page_next_to_original_app(self):
        from streamlit.source_util import get_pages
        pages = get_pages(str(ROOT / "app/app.py"))
        paths = {Path(page["script_path"]).resolve() for page in pages.values()}
        self.assertIn(PAGE.resolve(), paths)
        self.assertIn((ROOT / "app/app.py").resolve(), paths)


if __name__ == "__main__":
    unittest.main()
