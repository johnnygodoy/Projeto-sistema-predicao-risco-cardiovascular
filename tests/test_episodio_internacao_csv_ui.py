"""AppTest: upload boundary injected; parser and deterministic rules are real."""

import csv
import hashlib
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import streamlit as st
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
from episodio_internacao_csv import HEADER, MAX_BYTES, parse_episode_csv
from episodio_internacao_demo import Scenario

PAGE = ROOT / "app/pages/3_Internacao.py"
ALERTS = (ROOT / "Datasets/csv/internacao_alertas.csv").read_bytes()
CONTROL = (ROOT / "Datasets/csv/internacao_controle.csv").read_bytes()


class Upload:
    def __init__(self, payload):
        self.payload = payload
        self.size = len(payload)

    def getvalue(self):
        return self.payload


def without(kind):
    reader = csv.DictReader(io.StringIO(ALERTS.decode()))
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=HEADER)
    writer.writeheader()
    writer.writerows(row for row in reader if row["record_type"] != kind)
    return output.getvalue().encode()


class EpisodeCSVPageTests(unittest.TestCase):
    def setUp(self):
        self.upload = Upload(ALERTS)
        self.address = "127.0.0.1"
        original_option = st.get_option
        option = patch("streamlit.get_option", side_effect=lambda name:
                       self.address if name == "server.address" else original_option(name))
        uploader = patch.object(st.sidebar, "file_uploader", side_effect=lambda *args, **kwargs: self.upload)
        parser = patch("episodio_internacao_csv.parse_episode_csv", wraps=parse_episode_csv)
        self.get_option = option.start()
        self.uploader = uploader.start()
        self.parser = parser.start()
        self.addCleanup(option.stop)
        self.addCleanup(uploader.stop)
        self.addCleanup(parser.stop)

    def app(self):
        app = AppTest.from_file(str(PAGE), default_timeout=10).run()
        app.selectbox(key="_episode_source").select("Upload CSV").run()
        self.assertFalse(app.exception)
        return app

    def compare(self, app):
        app.button(key="_episode_run").click().run()
        self.assertFalse(app.exception)
        return app

    def values(self, app):
        return [item.value for item in app.metric]

    def text(self, app):
        return " ".join(str(item.value) for group in
                        (app.caption, app.info, app.warning, app.error, app.subheader, app.markdown)
                        for item in group)

    def assert_empty(self, app):
        self.assertFalse(app.exception)
        self.assertEqual(self.values(app), [])
        self.assertNotIn("_episode_review", app.session_state)
        self.assertNotIn("_episode_csv", app.session_state)
        self.assertNotIn("Resumo da revisão", self.text(app))
        self.assertNotIn("Próxima conferência", self.text(app))

    def test_parse_only_on_explicit_comparison_and_repeat_is_deterministic(self):
        app = self.app()
        self.assertEqual(self.parser.call_count, 0)
        self.assert_empty(app)
        self.compare(app)
        self.assertEqual(self.parser.call_count, 1)
        self.assertEqual(self.values(app), ["2", "0", "0"])
        self.assertEqual(app.subheader[0].value, "Resumo da revisão")
        app.run()
        self.assertEqual(self.parser.call_count, 1)
        self.assertEqual(self.values(app), ["2", "0", "0"])
        self.compare(app)
        self.assertEqual(self.parser.call_count, 2)
        self.assertEqual(self.values(app), ["2", "0", "0"])
        self.assertEqual(len(app.main.button), 0)

    def test_control_and_missing_plans_have_separate_states(self):
        for data, expected, state in ((CONTROL, ["0", "2", "0"], "Sem divergências."),
                                      (without("plan"), ["0", "0", "2"], "Dados insuficientes.")):
            with self.subTest(expected=expected):
                self.upload = Upload(data)
                app = self.compare(self.app())
                self.assertEqual(self.values(app), expected)
                self.assertTrue(any(item.value.startswith(state) for item in app.info))
                self.assertEqual(len(app.warning), 0)

    def test_missing_history_or_current_is_not_promoted_or_normal(self):
        for kind in ("history", "observation"):
            with self.subTest(kind=kind):
                self.upload = Upload(without(kind))
                app = self.compare(self.app())
                self.assertEqual(self.values(app), ["0", "0", "2"])
                self.assertIn("Não foi possível comparar", self.text(app))

    def test_changed_bytes_invalidate_before_new_compare(self):
        app = self.compare(self.app())
        self.upload = Upload(CONTROL)
        app.run()
        self.assert_empty(app)
        self.assertEqual(self.parser.call_count, 1)
        self.compare(app)
        self.assertEqual(self.values(app), ["0", "2", "0"])
        self.assertIn(hashlib.sha256(CONTROL).hexdigest()[:12], self.text(app))
        self.assertNotIn(hashlib.sha256(ALERTS).hexdigest()[:12], self.text(app))

    def test_removing_upload_then_reselecting_requires_compare(self):
        app = self.compare(self.app())
        self.upload = None
        app.run()
        self.assert_empty(app)
        self.assertFalse(any(button.key == "_episode_run" for button in app.button))
        self.upload = Upload(ALERTS)
        app.run()
        self.assert_empty(app)
        self.compare(app)
        self.assertEqual(self.values(app), ["2", "0", "0"])

    def test_source_switch_clears_upload_result_and_scenario_result(self):
        app = self.compare(self.app())
        app.selectbox(key="_episode_source").select("Cenários").run()
        self.assert_empty(app)
        app.selectbox(key="_episode_scenario").select(Scenario.EXPECTED.value).run()
        self.compare(app)
        self.assertEqual(self.values(app), ["0", "2", "0"])
        app.selectbox(key="_episode_source").select("Upload CSV").run()
        self.assert_empty(app)
        self.compare(app)
        self.assertEqual(self.values(app), ["2", "0", "0"])

    def test_clear_resets_source_and_upload_generation(self):
        app = self.compare(self.app())
        generation = app.session_state["_episode_upload_generation"]
        app.button(key="_episode_clear").click().run()
        self.assert_empty(app)
        self.assertEqual(app.selectbox(key="_episode_source").value, "Cenários")
        self.assertEqual(app.selectbox(key="_episode_scenario").value, "Selecione um cenário")
        self.assertGreater(app.session_state["_episode_upload_generation"], generation)
        app.selectbox(key="_episode_source").select("Upload CSV").run()
        self.assert_empty(app)

    def test_invalid_replacement_has_short_error_without_echo_or_stale_result(self):
        app = self.compare(self.app())
        self.upload = Upload(ALERTS.replace(b"EPISODIO-EXEMPLO-01", b"PRIVATE_NAME"))
        app.run()
        self.assert_empty(app)
        self.compare(app)
        self.assert_empty(app)
        self.assertEqual(len(app.error), 1)
        self.assertLess(len(app.error[0].value), 180)
        self.assertNotIn("PRIVATE_NAME", self.text(app))
        self.assertEqual(len(app.dataframe), 0)

    def test_oversize_declared_size_does_not_read_bytes_or_parse(self):
        class TooBig:
            size = MAX_BYTES + 1
            def getvalue(self):
                raise AssertionError("must not read oversize upload")
        self.upload = TooBig()
        app = self.app()
        self.assert_empty(app)
        self.assertEqual(self.parser.call_count, 0)
        self.assertIn("128 KiB", app.error[0].value)
        self.assertFalse(any(button.key == "_episode_run" for button in app.button))

    def test_actual_bytes_size_is_checked_even_if_declared_size_is_wrong(self):
        self.upload = Upload(b"x" * (MAX_BYTES + 1))
        self.upload.size = 1
        app = self.app()
        self.assert_empty(app)
        self.assertEqual(self.parser.call_count, 0)
        self.assertEqual(len(app.error), 1)

    def test_remote_and_wildcard_execution_never_construct_uploader(self):
        for address in ("0.0.0.0", "", "localhost", "::", "example.org", None):
            with self.subTest(address=address):
                self.address = address
                self.uploader.reset_mock()
                app = self.app()
                self.assert_empty(app)
                self.uploader.assert_not_called()
                self.assertEqual(self.parser.call_count, 0)
                self.assertFalse(any(button.key == "_episode_run" for button in app.button))
                self.assertIn("Upload CSV desativado", self.text(app))

    def test_losing_local_guard_clears_and_reenable_requires_new_comparison(self):
        app = self.compare(self.app())
        self.address = "0.0.0.0"
        self.uploader.reset_mock()
        app.run()
        self.assert_empty(app)
        self.uploader.assert_not_called()
        self.address = "::1"
        app.run()
        self.assert_empty(app)
        self.compare(app)
        self.assertEqual(self.values(app), ["2", "0", "0"])

    def test_csv_summary_preserves_full_dates_and_seconds(self):
        data = ALERTS.replace(b"2026-01-15T08:00:00Z", b"2026-01-14T08:00:00Z")
        data = data.replace(b"2026-01-15T09:00:00Z", b"2026-01-14T09:00:17Z")
        self.upload = Upload(data)
        app = self.compare(self.app())
        self.assertEqual(self.values(app), ["2", "0", "0"])
        plans = [item.value for item in app.markdown if item.value.startswith("**Plano:**")]
        records = [item.value for item in app.markdown if item.value.startswith("**Registro:**")]
        self.assertTrue(all("14/01/2026 09:00:17 UTC" in text for text in plans))
        self.assertTrue(all("15/01/2026 10:00:00 UTC" in text for text in plans + records))

    def test_provenance_schema_limits_and_review_only_actions(self):
        app = self.compare(self.app())
        text = self.text(app)
        for expected in ("EPISODIO-EXEMPLO-01", "8 registros", "origem declarada: simulado",
                         hashlib.sha256(ALERTS).hexdigest()[:12], "15/01/2026 às 10:00:00 UTC",
                         "PLANO-ITEM", "Regra C02", "INTERNACAO_SCHEMA.md", "500 registros",
                         "64 caracteres", "Não são inferidos de vídeo", "não significa normalidade clínica",
                         "Confira se existe atualização da prescrição registrada",
                         "Confira a confirmação de movimentação"):
            self.assertIn(expected, text)
        self.assertNotIn("Projeto FIAP ·", text)
        self.assertNotIn("João Roberto", text)
        self.assertEqual([item.label for item in app.metric],
                         ["Alertas", "Sem divergências", "Dados insuficientes"])


if __name__ == "__main__":
    unittest.main()
