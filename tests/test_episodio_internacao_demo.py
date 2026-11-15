"""Regras determinísticas, sem serviços externos nem dados de pacientes."""

from dataclasses import FrozenInstanceError, replace
from datetime import timedelta
from pathlib import Path
import ast
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
from episodio_internacao_demo import (
    MAX_OBSERVATION_AGE, MovementEvent, Observation, Scenario, Status,
    TreatmentState, build_demo, plan_rows, review_episode, review_rows, summarize_review, timeline_rows,
)


class EpisodeTests(unittest.TestCase):
    def setUp(self):
        self.a = build_demo(Scenario.EXPECTED)
        self.b = build_demo(Scenario.UNEXPECTED)

    def test_expected_and_unexpected_share_exactly_the_same_evidence(self):
        for field in ("episode_id", "as_of", "treatment_history", "treatment_current",
                      "movement_history", "movement_current"):
            self.assertEqual(getattr(self.a, field), getattr(self.b, field))
        self.assertEqual(timeline_rows(self.a), timeline_rows(self.b))
        self.assertNotEqual(self.a.treatment_plan.expected, self.b.treatment_plan.expected)
        self.assertNotEqual(self.a.movement_plan.expected, self.b.movement_plan.expected)

    def test_both_plans_are_explicit_and_prospective(self):
        for episode in (self.a, self.b):
            for plan, observed in ((episode.treatment_plan, episode.treatment_current),
                                   (episode.movement_plan, episode.movement_current)):
                self.assertLess(plan.recorded_at, observed.at)
                self.assertLessEqual(plan.recorded_at, plan.valid_from)
                self.assertLessEqual(plan.valid_from, observed.at)
                self.assertLess(observed.at, plan.valid_until)

    def test_same_change_matches_plan_a_and_alerts_only_for_plan_b(self):
        a, b = review_episode(self.a), review_episode(self.b)
        self.assertEqual((a.alert_count, a.unavailable_count), (0, 0))
        self.assertEqual((b.alert_count, b.unavailable_count), (2, 0))
        self.assertTrue(all(item.changed for item in a.assessments + b.assessments))
        self.assertEqual([item.rule for item in a.assessments], ["C01", "C01"])
        self.assertEqual([item.rule for item in b.assessments], ["C02", "C02"])

    def test_missing_data_is_not_compatible_or_an_unexpected_change(self):
        result = review_episode(build_demo(Scenario.UNAVAILABLE))
        self.assertEqual((result.alert_count, result.unavailable_count), (0, 2))
        self.assertTrue(all(item.status is Status.UNAVAILABLE for item in result.assessments))
        self.assertTrue(all(item.rule == "D01" and item.changed is None for item in result.assessments))

    def test_no_plan_is_unevaluable_not_an_unexpected_change(self):
        result = review_episode(build_demo(Scenario.NO_PLAN))
        self.assertEqual((result.alert_count, result.unavailable_count), (0, 2))
        self.assertEqual([item.rule for item in result.assessments], ["P01", "P01"])

    def test_each_channel_is_independent(self):
        result = review_episode(replace(self.b, treatment_current=None))
        self.assertEqual((result.alert_count, result.unavailable_count), (1, 1))
        self.assertFalse(result.treatment.alert)
        self.assertTrue(result.movement.alert)
        result = review_episode(replace(self.b, movement_plan=None))
        self.assertTrue(result.treatment.alert)
        self.assertEqual(result.movement.rule, "P01")

    def test_no_change_matching_plan_b_is_compatible(self):
        episode = replace(self.b,
            treatment_current=replace(self.b.treatment_current, value=TreatmentState.ACTIVE),
            movement_current=replace(self.b.movement_current, value=MovementEvent.IN_BED))
        result = review_episode(episode)
        self.assertEqual(result.alert_count, 0)
        self.assertTrue(all(item.changed is False for item in result.assessments))
        self.assertTrue(all(item.status is Status.ALIGNED for item in result.assessments))

    def test_missing_expected_change_also_is_a_mismatch(self):
        episode = replace(self.a, treatment_current=replace(self.a.treatment_current, value=TreatmentState.ACTIVE))
        result = review_episode(episode)
        self.assertTrue(result.treatment.alert)
        self.assertFalse(result.treatment.changed)
        self.assertFalse(result.movement.alert)

    def test_plan_wrong_scope_retroactive_and_expired_are_not_alerts(self):
        p = self.b.treatment_plan
        variants = {
            "wrong episode": (replace(p, episode_id="OTHER-SIMULATION"), "P02"),
            "wrong item": (replace(p, subject_id="OTHER-FICTIONAL-ITEM"), "P02"),
            "wrong type": (replace(p, expected=MovementEvent.LEFT_BED), "P03"),
            "naive date": (replace(p, recorded_at=p.recorded_at.replace(tzinfo=None)), "P04"),
            "simultaneous": (replace(p, recorded_at=self.b.as_of), "P05"),
            "retroactive": (replace(p, recorded_at=self.b.as_of + timedelta(minutes=1)), "P05"),
            "recorded after start": (replace(p, recorded_at=p.valid_from + timedelta(seconds=1)), "P05"),
            "reversed interval": (replace(p, valid_until=p.valid_from), "P05"),
            "not yet valid": (replace(p, valid_from=self.b.as_of + timedelta(minutes=1)), "P06"),
            "expired at boundary": (replace(p, valid_from=p.recorded_at, valid_until=self.b.as_of), "P06"),
        }
        for label, (plan, rule) in variants.items():
            with self.subTest(label=label):
                result = review_episode(replace(self.b, treatment_plan=plan)).treatment
                self.assertEqual(result.rule, rule)
                self.assertIs(result.status, Status.UNAVAILABLE)
                self.assertFalse(result.alert)

    def test_movement_plan_has_same_missing_and_temporal_guards(self):
        p = self.b.movement_plan
        for plan in (None, replace(p, recorded_at=self.b.as_of),
                     replace(p, subject_id="WRONG"), replace(p, expected=TreatmentState.ACTIVE)):
            with self.subTest(plan=plan):
                result = review_episode(replace(self.b, movement_plan=plan)).movement
                self.assertIs(result.status, Status.UNAVAILABLE)
                self.assertFalse(result.alert)

    def test_future_and_stale_data_are_unavailable_with_boundary_inclusive(self):
        for offset in (-timedelta(seconds=1), MAX_OBSERVATION_AGE + timedelta(seconds=1)):
            result = review_episode(replace(self.a, as_of=self.a.as_of + offset))
            self.assertEqual([item.rule for item in result.assessments], ["D04", "D04"])
            self.assertEqual(result.alert_count, 0)
        result = review_episode(replace(self.a, as_of=self.a.as_of + MAX_OBSERVATION_AGE))
        self.assertTrue(all(item.status is Status.ALIGNED for item in result.assessments))

    def test_invalid_history_types_order_duplicates_and_missing_values(self):
        h = self.a.treatment_history
        variants = [
            ((), "D05"),
            ((replace(h[0], value=None), h[1]), "D05"),
            ((replace(h[0], value=MovementEvent.IN_BED), h[1]), "D05"),
            ((replace(h[0], at=h[0].at.replace(tzinfo=None)), h[1]), "D05"),
            (tuple(reversed(h)), "D06"),
            ((h[1], h[1]), "D06"),
            ((h[0], replace(h[1], at=self.a.as_of)), "D06"),
        ]
        for history, rule in variants:
            with self.subTest(history=history):
                result = review_episode(replace(self.a, treatment_history=history)).treatment
                self.assertEqual(result.rule, rule)
                self.assertFalse(result.alert)

    def test_current_type_and_naive_timestamps_are_unavailable(self):
        current = self.a.treatment_current
        variants = [(replace(current, value=MovementEvent.LEFT_BED), "D02"),
                    (replace(current, at=current.at.replace(tzinfo=None)), "D03")]
        for observed, rule in variants:
            with self.subTest(observed=observed):
                self.assertEqual(review_episode(replace(self.a, treatment_current=observed)).treatment.rule, rule)
        result = review_episode(replace(self.a, as_of=self.a.as_of.replace(tzinfo=None)))
        self.assertEqual([item.rule for item in result.assessments], ["D03", "D03"])

    def test_impossible_movement_sequence_is_unavailable_not_reassuring(self):
        episode = replace(self.a, movement_current=replace(self.a.movement_current, value=MovementEvent.RETURNED_TO_BED))
        result = review_episode(episode).movement
        self.assertEqual(result.rule, "D07")
        self.assertFalse(result.alert)

    def test_valid_return_sequence_is_supported_by_typed_timeline(self):
        current = replace(self.a.movement_current, value=MovementEvent.RETURNED_TO_BED)
        history = (self.a.movement_history[0],
                   replace(self.a.movement_history[1], value=MovementEvent.LEFT_BED))
        plan = replace(self.a.movement_plan, expected=MovementEvent.RETURNED_TO_BED)
        result = review_episode(replace(self.a, movement_history=history,
                                       movement_current=current, movement_plan=plan)).movement
        self.assertIs(result.status, Status.ALIGNED)

    def test_deterministic_repeat_does_not_mutate_episode_or_accumulate_alerts(self):
        self.assertEqual(review_episode(self.b), review_episode(self.b))
        for _ in range(5):
            self.assertEqual(review_episode(self.b).alert_count, 2)
        with self.assertRaises(FrozenInstanceError):
            self.a.episode_id = "changed"

    def test_labels_and_timeline_are_explicit_and_ordered(self):
        rows = timeline_rows(self.a)
        self.assertEqual(len(rows), 6)
        self.assertEqual([r["Instante (UTC)"] for r in rows], sorted(r["Instante (UTC)"] for r in rows))
        self.assertEqual(sum(r["Origem"] == "Atual" for r in rows), 2)
        self.assertEqual(len(plan_rows(self.a)), 2)
        rows = review_rows(review_episode(build_demo(Scenario.UNAVAILABLE)))
        self.assertTrue(all(r["Observado"] == "Indisponível" for r in rows))
        self.assertTrue(all(r["Resultado"] == Status.UNAVAILABLE.value for r in rows))

    def test_unknown_scenario_is_rejected(self):
        with self.assertRaises(ValueError):
            build_demo("real patient")

    def test_summary_has_exactly_three_states_and_deterministic_sentences(self):
        self.assertEqual({item.value for item in Status}, {"Alertas", "Sem divergências", "Dados insuficientes"})
        expected = {Scenario.EXPECTED: Status.ALIGNED, Scenario.UNEXPECTED: Status.REVIEW,
                    Scenario.UNAVAILABLE: Status.UNAVAILABLE, Scenario.NO_PLAN: Status.UNAVAILABLE}
        for scenario, status in expected.items():
            with self.subTest(scenario=scenario):
                episode = build_demo(scenario)
                summary = summarize_review(episode, review_episode(episode))
                self.assertIs(summary.status, status)
                self.assertEqual(summary, summarize_review(episode, review_episode(episode)))
                self.assertEqual(len(summary.items), 2)
                self.assertTrue(all(item.next_action.startswith("Confira ") for item in summary.items))

    def test_alert_summary_names_plan_record_difference_and_evidence_action(self):
        summary = summarize_review(self.b, review_episode(self.b))
        treatment, movement = summary.items
        self.assertIn("Ativo", treatment.plan)
        self.assertIn("Pausado", treatment.record)
        self.assertIn("previa Ativo", treatment.difference)
        self.assertEqual(treatment.next_action, "Confira se existe atualização da prescrição registrada para esse horário.")
        self.assertEqual(movement.next_action, "Confira a confirmação de movimentação e o horário do registro.")
        self.assertIn("PLANO-B-ITEM", treatment.provenance)
        self.assertIn("C02", treatment.provenance)
        self.assertIn("09:00 UTC", treatment.plan)
        self.assertIn("10:00 UTC", treatment.record)

    def test_control_summary_does_not_turn_expected_change_into_alert(self):
        summary = summarize_review(self.a, review_episode(self.a))
        self.assertIs(summary.status, Status.ALIGNED)
        self.assertTrue(all("prevista no plano" in item.difference for item in summary.items))
        self.assertTrue(all("C01" in item.provenance for item in summary.items))
        self.assertTrue(all("atualização" not in item.next_action for item in summary.items))

    def test_missing_data_and_missing_plan_have_different_review_actions(self):
        episode = build_demo(Scenario.UNAVAILABLE)
        missing_data = summarize_review(episode, review_episode(episode))
        episode = build_demo(Scenario.NO_PLAN)
        missing_plan = summarize_review(episode, review_episode(episode))
        self.assertIs(missing_data.status, Status.UNAVAILABLE)
        self.assertIs(missing_plan.status, Status.UNAVAILABLE)
        self.assertIn("registro de prescrição válido", missing_data.items[0].next_action)
        self.assertIn("vigência do plano de prescrição", missing_plan.items[0].next_action)
        self.assertEqual(missing_plan.items[0].plan, "Plano prospectivo indisponível.")
        self.assertTrue(all("Não foi possível comparar" in item.difference for item in missing_data.items))

    def test_mixed_summary_preserves_known_alert_and_missing_evidence(self):
        episode = replace(self.b, movement_current=None)
        summary = summarize_review(episode, review_episode(episode))
        self.assertIs(summary.status, Status.REVIEW)
        self.assertIs(summary.items[1].status, Status.UNAVAILABLE)
        self.assertIn("1 com dados insuficientes", summary.text)
        self.assertIn("registro de movimentação válido", summary.items[1].next_action)

    def test_invalid_history_and_retroactive_plan_request_evidence_not_treatment(self):
        episode = replace(self.b, treatment_plan=replace(self.b.treatment_plan, recorded_at=self.b.as_of),
                          movement_history=())
        summary = summarize_review(episode, review_episode(episode))
        self.assertIs(summary.status, Status.UNAVAILABLE)
        self.assertIn("vigência do plano", summary.items[0].next_action)
        self.assertIn("histórico de movimentação", summary.items[1].next_action)
        self.assertIn("P05", summary.items[0].provenance)
        self.assertIn("D05", summary.items[1].provenance)

    def test_summary_rejects_a_review_from_another_plan(self):
        with self.assertRaisesRegex(ValueError, "não corresponde"):
            summarize_review(self.a, review_episode(self.b))

    def test_core_has_only_standard_library_and_no_input_or_external_sender(self):
        source = (ROOT / "app/episodio_internacao_demo.py").read_text()
        tree = ast.parse(source)
        modules = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        self.assertLessEqual(modules, {"dataclasses", "datetime", "enum", "typing"})
        self.assertFalse(any(isinstance(node, ast.Import) for node in ast.walk(tree)))
        calls = {node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name)}
        self.assertFalse(calls & {"open", "eval", "exec", "compile", "__import__"})


if __name__ == "__main__":
    unittest.main()
