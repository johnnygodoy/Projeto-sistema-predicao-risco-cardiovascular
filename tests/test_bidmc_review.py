"""Sem linhas BIDMC embutidas. Integração opcional lê a cópia oficial externa.

Defina FIAP_BIDMC_SAMPLE com o caminho externo a este repositório para executar
os testes da amostra real. Não há download automático nem upload nos testes.
"""

import csv
import builtins
import hashlib
import io
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
from bidmc_review import (
    EXPECTED_BYTES, EXPECTED_SHA256, FILENAME, HEADER, SIGNALS, SampleError,
    _parse_verified_bytes, coverage_rows, signal_rows, verify_and_parse,
)
from local_inputs import local_uploads_enabled

try:
    from streamlit.testing.v1 import AppTest
except ImportError:
    AppTest = None

PAGE = ROOT / "app/pages/1_Monitoramento_Didatico.py"
EXTERNAL_SAMPLE = os.environ.get("FIAP_BIDMC_SAMPLE")


def invented_source_shape():
    """Valores inventados com casas decimais, não copiados da fonte observada."""
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(HEADER)
    for second in range(481):
        missing = 100 <= second < 113
        writer.writerow([second, 180.111, "NaN" if missing else 139.222,
                         29.333, "NaN" if missing else 89.444])
    return stream.getvalue().encode()


class AuthenticationTests(unittest.TestCase):
    def test_exact_official_fingerprint_is_pinned(self):
        self.assertEqual(EXPECTED_SHA256, "27d43794cf2e7150db10e8bfe61d2bfdb26873a8f1629ffc5fbee5c9a4eb6ef5")
        self.assertEqual(EXPECTED_BYTES, 7644)

    def test_name_size_and_hash_checked_before_parser(self):
        for filename, payload in [("other.csv", b"x" * EXPECTED_BYTES),
                                  (FILENAME, b""), (FILENAME, b"x" * EXPECTED_BYTES),
                                  (FILENAME, invented_source_shape())]:
            with self.subTest(filename=filename), patch("bidmc_review._parse_verified_bytes") as parse:
                with self.assertRaises(SampleError):
                    verify_and_parse(filename, payload)
                parse.assert_not_called()

    def test_arbitrary_filename_not_echoed(self):
        with self.assertRaises(SampleError) as error:
            verify_and_parse("SENSITIVE_FILENAME.csv", b"x")
        self.assertNotIn("SENSITIVE_FILENAME", str(error.exception))

    def test_guard_only_trusts_literal_loopback_bind(self):
        for value in [None, "", "0.0.0.0", "::", "localhost", "127.0.0.1.evil", "https://127.0.0.1",
                      "192.168.1.1", {"origin": "http://127.0.0.1"}, True]:
            self.assertFalse(local_uploads_enabled(value))
        self.assertTrue(local_uploads_enabled("127.0.0.1"))
        self.assertTrue(local_uploads_enabled("::1"))


class ShapeAndChartTests(unittest.TestCase):
    def test_relative_seconds_and_coverage(self):
        sample = _parse_verified_bytes(invented_source_shape())
        self.assertEqual(sample.seconds, tuple(range(481)))
        self.assertEqual({row["Sinal"]: row["Ausentes"] for row in coverage_rows(sample)},
                         {"HR": 0, "SpO2": 13, "RESP": 0, "PULSE": 13})
        self.assertEqual(sample.values("HR"), (180.111,) * 481)

    def test_chart_preserves_gaps_without_zero_or_interpolation(self):
        sample = _parse_verified_bytes(invented_source_shape())
        rows = signal_rows(sample, "SpO2")
        self.assertEqual(len(rows), 481)
        self.assertEqual(sum(row["Valor"] is None for row in rows), 13)
        self.assertNotEqual(rows[99]["Trecho"], rows[113]["Trecho"])
        self.assertTrue(all(row["Valor"] is None for row in rows[100:113]))
        self.assertEqual({row["Valor"] for row in rows if row["Valor"] is not None}, {89.444})
        self.assertEqual({row["Unidade"] for row in rows}, {"%"})

    def test_invalid_shape_time_missing_count_and_cells(self):
        data = invented_source_shape()
        cases = [b"\xff", data.replace(b"Time [s]", b"timestamp", 1),
                 b"\n".join(data.splitlines()[:-1]), data + b"481,1,2,3,4\n",
                 data.replace(b"180.111", b"inf", 1), data.replace(b"NaN", b"0", 1),
                 data.replace(b"0,180.111", b"2,180.111", 1),
                 data.replace(b"180.111", b"=1+2", 1), b"x" * 32769]
        for payload in cases:
            with self.subTest(size=len(payload)), self.assertRaises(SampleError):
                _parse_verified_bytes(payload)


class Upload(io.BytesIO):
    name = FILENAME

    @property
    def size(self):
        return len(self.getbuffer())


@unittest.skipIf(AppTest is None, "Streamlit não instalado")
class ReviewUiTests(unittest.TestCase):
    def setUp(self):
        import streamlit as st
        original = st.get_option
        guard = patch("streamlit.get_option", side_effect=lambda name: "127.0.0.1" if name == "server.address" else original(name))
        guard.start()
        self.addCleanup(guard.stop)

    def app(self):
        app = AppTest.from_file(str(PAGE), default_timeout=10).run()
        app.selectbox(key="_mon_source").select("BIDMC · PhysioNet").run()
        self.assertFalse(app.exception)
        return app

    def test_no_checkbox_and_no_upload_on_nonlocal_bind(self):
        app = self.app()
        self.assertEqual(len(app.checkbox), 0)
        self.assertEqual(len(app.get("file_uploader")), 1)
        with patch("streamlit.get_option", return_value=None):
            app.run()
            self.assertEqual(len(app.get("file_uploader")), 0)
            self.assertEqual(len(app.metric), 0)

    def test_mismatched_file_rejected_without_parsing(self):
        app = self.app()
        with patch("streamlit.sidebar.file_uploader", return_value=Upload(b"x" * EXPECTED_BYTES)), \
             patch("bidmc_review._parse_verified_bytes") as parse:
            app.run()
            app.button(key="_mon_bidmc_validate").click().run()
            self.assertEqual(len(app.error), 1)
            self.assertEqual(len(app.metric), 0)
            parse.assert_not_called()
        self.assertFalse(app.exception)

    def test_wrong_filename_size_stop_before_bytes(self):
        class NoRead:
            name = "wrong.csv"
            size = EXPECTED_BYTES

            def getvalue(self):
                raise AssertionError("No read allowed")

        app = self.app()
        with patch("streamlit.sidebar.file_uploader", return_value=NoRead()):
            app.run()
            self.assertEqual(len(app.error), 1)
            self.assertFalse(app.exception)
        wrong_size = NoRead()
        wrong_size.name = FILENAME
        wrong_size.size = EXPECTED_BYTES + 1
        with patch("streamlit.sidebar.file_uploader", return_value=wrong_size):
            app.run()
            self.assertEqual(len(app.error), 1)
            self.assertFalse(app.exception)

    def test_native_navigation_labels_icons_and_unchanged_route(self):
        from streamlit.source_util import get_pages
        pages = get_pages(str(ROOT / "app/app.py"))
        # AppTest 1.32 executa a primeira página registrada. Preservar ambas,
        # escolhendo a de monitoramento como entrada deste teste isolado.
        pages = dict(sorted(pages.items(), key=lambda item: item[1]["script_path"] != str(PAGE)))
        with patch("streamlit.elements.widgets.button.get_main_script_directory", return_value=str(ROOT / "app")), \
             patch("streamlit.elements.widgets.button.source_util.get_pages", return_value=pages):
            app = self.app()
        links = app.sidebar.get("page_link")
        self.assertEqual([link.proto.label for link in links], ["Predição de risco", "Monitoramento"])
        self.assertEqual([link.proto.icon for link in links], ["🫀", "📈"])
        self.assertEqual(links[1].proto.page, "Monitoramento_Didatico")
        self.assertFalse(app.exception)

    def test_native_navigation_in_actual_entrypoint_with_model_io_stub(self):
        # Valida navegação no app real, não o modelo treinado. Nenhum pickle é lido.
        original_open = builtins.open

        def open_for_test(file, *args, **kwargs):
            if str(file) == str(ROOT / "model/model.pkl"):
                return io.BytesIO(b"model-io-stub")
            return original_open(file, *args, **kwargs)

        with patch("builtins.open", side_effect=open_for_test), patch("pickle.load", return_value=object()):
            app = AppTest.from_file(str(ROOT / "app/app.py"), default_timeout=10).run()
        self.assertFalse(app.exception)
        links = app.sidebar.get("page_link")
        self.assertEqual([link.proto.label for link in links], ["Predição de risco", "Monitoramento"])
        self.assertEqual([link.proto.icon for link in links], ["🫀", "📈"])
        self.assertEqual([link.proto.page for link in links], ["app", "Monitoramento_Didatico"])

    def test_default_menu_config_changes_no_launch_or_theme_setting(self):
        import tomllib
        config = tomllib.loads((ROOT / ".streamlit/config.toml").read_text())
        self.assertEqual(config, {"client": {"showSidebarNavigation": False}})

    def test_visible_copy_has_no_removed_word_or_checkbox(self):
        app = self.app()
        for source in ["Exemplo com variações", "Exemplo", "Importar CSV de teste", "BIDMC · PhysioNet"]:
            app.selectbox(key="_mon_source").select(source).run()
            self.assertEqual(len(app.checkbox), 0)
            if source in ["Exemplo", "Exemplo com variações"]:
                app.button(key="_mon_validate").click().run()
                app.button(key="_mon_run").click().run()
            text = " ".join(str(element.value) for kind in ["title", "header", "subheader", "caption", "info", "warning", "error", "markdown"] for element in app.get(kind))
            self.assertNotRegex(text.lower(), "fict[ií]")
            self.assertFalse(app.exception)

    def test_individual_page_has_clear_launch_fallback(self):
        app = self.app()
        self.assertTrue(any("página principal" in item.value for item in app.sidebar.info))
        self.assertFalse(app.exception)


@unittest.skipUnless(EXTERNAL_SAMPLE, "Defina FIAP_BIDMC_SAMPLE para a integração com cópia externa")
class OfficialSampleIntegrationTests(unittest.TestCase):
    def sample_bytes(self):
        path = Path(EXTERNAL_SAMPLE).resolve()
        self.assertFalse(path.is_relative_to(ROOT), "A amostra deve permanecer fora do repositório")
        self.assertEqual(path.name, FILENAME)
        return path.read_bytes()

    def test_authentic_source_coverage_and_charts(self):
        raw = self.sample_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), EXPECTED_SHA256)
        sample = verify_and_parse(FILENAME, raw)
        coverage = coverage_rows(sample)
        self.assertEqual([r["Presentes"] for r in coverage], [481, 468, 481, 468])
        self.assertEqual([r["Ausentes"] for r in coverage], [0, 13, 0, 13])
        for signal in SIGNALS:
            rows = signal_rows(sample, signal)
            self.assertEqual([r["Valor"] for r in rows], list(sample.values(signal)))
            self.assertEqual([r["Segundo relativo"] for r in rows], list(range(481)))

    @unittest.skipIf(AppTest is None, "Streamlit não instalado")
    def test_official_upload_ui_switch_remove_clear_and_invalidate(self):
        import streamlit as st
        original = st.get_option
        raw = self.sample_bytes()
        with patch("streamlit.get_option", side_effect=lambda name: "127.0.0.1" if name == "server.address" else original(name)):
            app = AppTest.from_file(str(PAGE), default_timeout=10).run()
            app.selectbox(key="_mon_source").select("BIDMC · PhysioNet").run()
            with patch("streamlit.sidebar.file_uploader", return_value=Upload(raw)):
                app.run()
                self.assertEqual(len(app.metric), 0)
                app.button(key="_mon_bidmc_validate").click().run()
                self.assertEqual([m.value for m in app.metric], ["481", "0–480 s", "BIDMC 1.0.0"])
                app.selectbox(key="_bidmc_signal").select("SpO2").run()
                self.assertTrue(any("13 ausentes" in c.value for c in app.caption))
                self.assertFalse(any(b.key == "_mon_run" for b in app.button))
            with patch("streamlit.sidebar.file_uploader", return_value=Upload(b"x" * EXPECTED_BYTES)):
                app.run()
                self.assertEqual(len(app.metric), 0)
            with patch("streamlit.sidebar.file_uploader", return_value=Upload(raw)):
                app.run()
                app.button(key="_mon_bidmc_validate").click().run()
                with patch("streamlit.get_option", return_value="0.0.0.0"):
                    app.run()
                    self.assertEqual(len(app.metric), 0)
                    self.assertNotIn("_bidmc_verified", app.session_state)
                app.run()
                app.button(key="_mon_bidmc_validate").click().run()
                app.selectbox(key="_mon_source").select("Exemplo").run()
                self.assertNotIn("_bidmc_verified", app.session_state)
                self.assertEqual(len(app.metric), 0)
                app.button(key="_mon_validate").click().run()
                app.button(key="_mon_run").click().run()
                self.assertEqual([m.value for m in app.metric], ["65", "0", "25"])
                app.selectbox(key="_mon_source").select("BIDMC · PhysioNet").run()
                self.assertNotIn("_mon_results", app.session_state)
                self.assertEqual(len(app.metric), 0)
                app.button(key="_mon_bidmc_validate").click().run()
                app.button(key="_mon_clear").click().run()
                self.assertEqual(len(app.metric), 0)
            self.assertFalse(app.exception)


if __name__ == "__main__":
    unittest.main()
