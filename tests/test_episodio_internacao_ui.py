"""AppTest real: resumo determinístico, navegação e invalidação de resultados."""

from pathlib import Path
import sys
import unicodedata
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
from episodio_internacao_demo import Scenario

PAGE = ROOT / "app/pages/3_Internacao.py"


class EpisodePageTests(unittest.TestCase):
    def app(self):
        app = AppTest.from_file(str(PAGE), default_timeout=10).run()
        self.assertFalse(app.exception)
        return app

    def select(self, app, scenario):
        app.selectbox(key="_episode_scenario").select(scenario.value).run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.metric), 0)
        return app

    def run_comparison(self, app):
        app.button(key="_episode_run").click().run()
        self.assertFalse(app.exception)
        return app

    def metrics(self, app):
        return [metric.value for metric in app.metric]

    def text(self, app):
        return " ".join(str(item.value) for group in
                        (app.title, app.subheader, app.caption, app.info, app.warning, app.markdown)
                        for item in group)

    def test_discovered_page_and_shared_native_menu_order(self):
        from streamlit.source_util import get_pages
        root = ROOT / "app"
        pages = get_pages(str(root / "app.py"))
        self.assertIn(PAGE.resolve(), {Path(page["script_path"]).resolve() for page in pages.values()})
        pages = dict(sorted(pages.items(), key=lambda item: item[1]["script_path"] != str(PAGE)))
        with patch("streamlit.elements.widgets.button.get_main_script_directory", return_value=str(root)), \
             patch("streamlit.elements.widgets.button.source_util.get_pages", return_value=pages):
            app = self.app()
        links = app.sidebar.get("page_link")
        self.assertEqual([link.proto.label for link in links],
                         ["Predição de risco", "Monitoramento", "Internação", "Áudio", "Vídeo"])
        self.assertEqual([link.proto.icon for link in links], ["🫀", "📈", "🏥", "🔊", "🎥"])
        self.assertEqual([link.proto.page for link in links],
                         ["app", "Monitoramento_Didatico", "Internacao", "Audio_Local", "Video_Movimento_Local"])
        self.assertEqual(app.title[0].value, "Internação")

    def test_initial_has_no_result_upload_chat_or_free_text(self):
        app = self.app()
        self.assertEqual(len(app.metric), 0)
        for kind in ("file_uploader", "chat_input", "text_input", "text_area"):
            self.assertEqual(len(app.get(kind)), 0)
        self.assertEqual(app.selectbox(key="_episode_scenario").value, "Selecione um cenário")
        self.assertTrue(any("Exemplo simulado" in item.value for item in app.info))
        self.assertFalse(any(button.key == "_episode_run" for button in app.button))

    def test_one_simulated_label_native_layout_and_no_individual_branding(self):
        app = self.run_comparison(self.select(self.app(), Scenario.UNEXPECTED))
        text = self.text(app) + " ".join(frame.value.to_string() for frame in app.dataframe)
        normalized = "".join(c for c in unicodedata.normalize("NFKD", text.lower())
                             if not unicodedata.combining(c))
        self.assertEqual(normalized.count("exemplo simulado"), 1)
        self.assertNotRegex(normalized, r"fictici\w*")
        self.assertIn("sem uso clinico", normalized)
        self.assertNotIn("João Roberto Godoy Rodrigues", self.text(app))
        self.assertNotIn("Projeto FIAP ·", self.text(app))
        source = PAGE.read_text()
        self.assertNotIn("unsafe_allow_html", source)
        self.assertNotIn("<style", source)

    def test_preview_has_identical_records_and_distinct_prior_plans(self):
        app = self.select(self.app(), Scenario.EXPECTED)
        history = app.dataframe[0].value.copy()
        plan = app.dataframe[1].value.copy()
        self.select(app, Scenario.UNEXPECTED)
        self.assertTrue(history.equals(app.dataframe[0].value))
        self.assertFalse(plan.equals(app.dataframe[1].value))
        self.assertEqual(len(app.metric), 0)

    def test_expected_change_has_clear_control_summary_and_review_actions(self):
        app = self.run_comparison(self.select(self.app(), Scenario.EXPECTED))
        self.assertEqual(self.metrics(app), ["0", "2", "0"])
        self.assertTrue(any(item.value.startswith("Sem divergências.") for item in app.info))
        self.assertTrue(any("prevista no plano" in item.value for item in app.markdown))
        self.assertIn("Confira a versão e o horário da prescrição", self.text(app))
        self.assertIn("Confira o horário e a origem do registro de movimentação", self.text(app))
        self.assertIn("não significa normalidade clínica", self.text(app))
        self.assertEqual(len(app.warning), 0)
        self.assertEqual(len(app.sidebar.metric), 0)
        self.assertEqual(len(app.main.button), 0)

    def test_alert_summary_is_before_details_with_four_readable_fields_per_channel(self):
        app = self.run_comparison(self.select(self.app(), Scenario.UNEXPECTED))
        self.assertEqual(self.metrics(app), ["2", "0", "0"])
        self.assertTrue(app.warning[0].value.startswith("Alertas."))
        self.assertEqual(app.subheader[0].value, "Resumo da revisão")
        self.assertEqual([item.value for item in app.subheader[:3]], ["Resumo da revisão", "Prescrição", "Movimento"])
        for label in ("**Plano:**", "**Registro:**", "**Diferença:**", "**Próxima conferência:**"):
            self.assertEqual(sum(item.value.startswith(label) for item in app.markdown), 2)
        self.assertIn("Confira se existe atualização da prescrição registrada", self.text(app))
        self.assertIn("Confira a confirmação de movimentação", self.text(app))
        self.assertIn("Regra C02", self.text(app))
        self.assertIn("PLANO-B-ITEM", self.text(app))
        self.assertTrue(any("Nada é enviado" in item.value for item in app.sidebar.caption))
        self.run_comparison(app)
        self.assertEqual(self.metrics(app), ["2", "0", "0"])

    def test_unavailable_is_not_compatible_and_requests_missing_records(self):
        app = self.run_comparison(self.select(self.app(), Scenario.UNAVAILABLE))
        self.assertEqual(self.metrics(app), ["0", "0", "2"])
        self.assertTrue(any(item.value.startswith("Dados insuficientes.") for item in app.info))
        self.assertIn("registro de prescrição válido", self.text(app))
        self.assertIn("registro de movimentação válido", self.text(app))
        self.assertTrue(all(value == "Indisponível" for value in app.dataframe[-1].value["Observado"]))
        self.assertEqual(len(app.warning), 0)

    def test_no_plan_is_unevaluable_and_requests_plan_evidence(self):
        app = self.run_comparison(self.select(self.app(), Scenario.NO_PLAN))
        self.assertEqual(self.metrics(app), ["0", "0", "2"])
        self.assertTrue(all(value == "P01" for value in app.dataframe[-1].value["Regra"]))
        self.assertIn("Plano prospectivo indisponível.", self.text(app))
        self.assertIn("vigência do plano de prescrição", self.text(app))
        self.assertTrue(any(item.value.startswith("Dados insuficientes.") for item in app.info))

    def test_switch_clears_previous_summary_actions_and_results(self):
        app = self.run_comparison(self.select(self.app(), Scenario.UNEXPECTED))
        for scenario, expected in ((Scenario.EXPECTED, ["0", "2", "0"]),
                                   (Scenario.UNAVAILABLE, ["0", "0", "2"]),
                                   (Scenario.UNEXPECTED, ["2", "0", "0"])):
            self.select(app, scenario)
            self.assertNotIn("_episode_review", app.session_state)
            self.assertNotIn("Próxima conferência", self.text(app))
            self.assertNotIn("Resumo da revisão", self.text(app))
            self.run_comparison(app)
            self.assertEqual(self.metrics(app), expected)

    def test_clear_and_reselect_does_not_restore_old_summary(self):
        app = self.run_comparison(self.select(self.app(), Scenario.UNEXPECTED))
        app.button(key="_episode_clear").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.metric), 0)
        self.assertEqual(len(app.dataframe), 0)
        self.assertNotIn("_episode_review", app.session_state)
        self.assertNotIn("Próxima conferência", self.text(app))
        self.assertEqual(app.selectbox(key="_episode_scenario").value, "Selecione um cenário")
        self.select(app, Scenario.UNEXPECTED)
        self.assertNotIn("Resumo da revisão", self.text(app))
        self.run_comparison(app)
        self.assertEqual(self.metrics(app), ["2", "0", "0"])

    def test_empty_selection_after_run_clears_summary_and_run_button(self):
        app = self.run_comparison(self.select(self.app(), Scenario.EXPECTED))
        app.selectbox(key="_episode_scenario").select("Selecione um cenário").run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.metric), 0)
        self.assertNotIn("_episode_review", app.session_state)
        self.assertFalse(any(button.key == "_episode_run" for button in app.button))

    def test_algorithm_version_change_discards_previous_result(self):
        app = self.run_comparison(self.select(self.app(), Scenario.UNEXPECTED))
        app.session_state["_episode_selected"] = ("old-version", Scenario.UNEXPECTED.value)
        app.run()
        self.assertFalse(app.exception)
        self.assertNotIn("_episode_review", app.session_state)
        self.assertEqual(len(app.metric), 0)
        self.assertNotIn("Próxima conferência", self.text(app))

    def test_movement_independence_and_only_three_output_states(self):
        app = self.run_comparison(self.select(self.app(), Scenario.UNEXPECTED))
        self.assertTrue(any("Não são inferidos de vídeo" in item.value for item in app.caption))
        self.assertTrue(any("não classificam queda ou perigo" in item.value for item in app.caption))
        self.assertEqual([item.label for item in app.metric], ["Alertas", "Sem divergências", "Dados insuficientes"])
        self.assertIn("não há chatbot", self.text(app))


if __name__ == "__main__":
    unittest.main()
