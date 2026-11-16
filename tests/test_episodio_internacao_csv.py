"""Contratos de CSV e regras existentes, sem dados de pessoas ou serviços externos."""

import ast
import csv
from datetime import datetime, timedelta, timezone
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
from episodio_internacao_csv import CSVValidationError, HEADER, MAX_BYTES, MAX_ROWS, parse_episode_csv
from episodio_internacao_demo import Status, review_episode, timeline_rows

ALERTS = (ROOT / "Datasets/csv/internacao_alertas.csv").read_bytes()
CONTROL = (ROOT / "Datasets/csv/internacao_controle.csv").read_bytes()


def rows(payload=ALERTS):
    return list(csv.DictReader(io.StringIO(payload.decode())))


def encode(records, header=HEADER):
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=header, lineterminator="\n")
    writer.writeheader()
    writer.writerows(records)
    return output.getvalue().encode()


class EpisodeCSVTests(unittest.TestCase):
    def assert_invalid(self, records, fragment=None):
        with self.assertRaises(CSVValidationError) as raised:
            parse_episode_csv(encode(records))
        if fragment:
            self.assertIn(fragment, str(raised.exception))
        self.assertNotIn("PRIVATE_NAME", str(raised.exception))
        return str(raised.exception)

    def test_fixture_alert_and_control_reuse_identical_observations(self):
        a, b = parse_episode_csv(ALERTS), parse_episode_csv(CONTROL)
        self.assertEqual((a.row_count, b.row_count), (8, 8))
        self.assertEqual(a.declared_source, "simulado")
        self.assertEqual(timeline_rows(a.episode), timeline_rows(b.episode))
        self.assertEqual(review_episode(a.episode).alert_count, 2)
        result = review_episode(b.episode)
        self.assertEqual((result.alert_count, result.unavailable_count), (0, 0))
        self.assertTrue(all(x.status is Status.ALIGNED for x in result.assessments))
        self.assertEqual(parse_episode_csv(ALERTS), a)

    def test_missing_plan_remains_missing_not_inferred_from_history(self):
        parsed = parse_episode_csv(encode([r for r in rows() if r["record_type"] != "plan"]))
        self.assertIsNone(parsed.episode.treatment_plan)
        self.assertIsNone(parsed.episode.movement_plan)
        result = review_episode(parsed.episode)
        self.assertEqual((result.alert_count, result.unavailable_count), (0, 2))
        self.assertEqual([x.rule for x in result.assessments], ["P01", "P01"])

    def test_history_is_not_promoted_to_observation(self):
        parsed = parse_episode_csv(encode([r for r in rows() if r["record_type"] == "history"]))
        self.assertIsNone(parsed.episode.treatment_current)
        self.assertIsNone(parsed.episode.movement_current)
        self.assertEqual(review_episode(parsed.episode).unavailable_count, 2)

    def test_one_channel_missing_does_not_suppress_other_alert(self):
        parsed = parse_episode_csv(encode([r for r in rows() if r["channel"] == "prescricao"]))
        result = review_episode(parsed.episode)
        self.assertEqual((result.alert_count, result.unavailable_count), (1, 1))

    def test_explicit_unavailable_and_missing_history_are_not_normal(self):
        records = rows()
        for r in records:
            if r["record_type"] == "observation":
                r["value"] = "indisponivel"
        result = review_episode(parse_episode_csv(encode(records)).episode)
        self.assertEqual((result.alert_count, result.unavailable_count), (0, 2))
        records = [r for r in rows() if r["record_type"] != "history"]
        self.assertEqual(review_episode(parse_episode_csv(encode(records)).episode).unavailable_count, 2)

    def test_equivalent_timezone_offsets_are_compared_in_utc(self):
        records = rows()
        for r in records:
            for field in ("timestamp", "valid_from", "valid_until", "as_of"):
                if r[field]:
                    instant = datetime.fromisoformat(r[field].replace("Z", "+00:00"))
                    r[field] = instant.astimezone(timezone(timedelta(hours=-3))).isoformat()
        a = parse_episode_csv(encode(records))
        self.assertEqual(a.episode, parse_episode_csv(ALERTS).episode)

    def test_future_current_or_inconsistent_cutoff_is_rejected(self):
        records = rows(); records[-1]["timestamp"] = "2026-01-15T10:00:01Z"
        self.assert_invalid(records, "posterior")
        records = rows(); records[-1]["as_of"] = "2026-01-15T10:01:00Z"
        self.assert_invalid(records, "único episódio")

    def test_stale_observation_uses_existing_unavailable_rule(self):
        records = rows()
        for r in records:
            r["as_of"] = "2026-01-15T10:15:01Z"
        result = review_episode(parse_episode_csv(encode(records)).episode)
        self.assertEqual([x.rule for x in result.assessments], ["D04", "D04"])
        self.assertEqual(result.alert_count, 0)

    def test_plan_outside_observation_window_is_unavailable_not_unexpected(self):
        records = rows()
        for r in records:
            if r["record_type"] == "plan":
                r["valid_from"] = "2026-01-15T09:00:00Z"
                r["valid_until"] = "2026-01-15T10:00:00Z"
        result = review_episode(parse_episode_csv(encode(records)).episode)
        self.assertEqual([x.rule for x in result.assessments], ["P06", "P06"])
        self.assertEqual(result.alert_count, 0)

    def test_global_order_and_history_current_relationship_are_strict(self):
        self.assert_invalid(list(reversed(rows())), "ordem temporal")
        records = rows()
        records[4]["timestamp"] = records[5]["timestamp"] = "2026-01-15T10:00:00Z"
        self.assert_invalid(records, "histórico precisa anteceder")

    def test_duplicate_history_current_and_plan_are_rejected(self):
        records = rows(); records.insert(1, dict(records[0])); self.assert_invalid(records, "duplicado")
        records = rows(); records.append(dict(records[-1])); self.assert_invalid(records, "duplicado")
        records = rows(); extra=dict(records[-1]);extra["timestamp"]="2026-01-15T10:00:01Z"
        for r in records: r["as_of"]="2026-01-15T10:01:00Z"
        extra["as_of"]="2026-01-15T10:01:00Z";records.append(extra);self.assert_invalid(records,"uma observação")
        records = rows(); extra=dict(records[2]);extra["timestamp"]="2026-01-15T09:01:00Z";extra["plan_id"]="PLANO-OUTRO";records.insert(4,extra)
        self.assert_invalid(records,"um plano por canal")

    def test_retroactive_or_invalid_plan_dates_are_rejected(self):
        records = rows()
        for r in records:
            if r["record_type"] == "plan": r["timestamp"]="2026-01-15T10:00:00Z"
        records.sort(key=lambda r:r["timestamp"])
        self.assert_invalid(records,"antes da observação")
        for field, value in [("valid_from","2026-01-15T08:59:59Z"),("valid_until","2026-01-15T10:00:00Z")]:
            records=rows();records[2][field]=value;self.assert_invalid(records,"vigência")

    def test_types_values_units_and_unknown_channels_are_rejected(self):
        for field,value in [("record_type","dose"),("channel","heart_rate"),("unit","bpm"),
                            ("value","75"),("value","PRIVATE_NAME"),("value","=2+2"),
                            ("episode_id","PRIVATE_NAME"),("source","hospital_real")]:
            with self.subTest(field=field,value=value):
                records=rows();records[0][field]=value;self.assert_invalid(records)

    def test_multiple_episode_or_source_is_rejected(self):
        for field,value in [("episode_id","EPISODIO-OUTRO"),("source","atividade_autorizada")]:
            records=rows();records[-1][field]=value;self.assert_invalid(records,"único episódio")
        records=rows()
        for r in records:r["source"]="atividade_autorizada"
        self.assertEqual(parse_episode_csv(encode(records)).declared_source,"atividade_autorizada")

    def test_plan_fields_only_on_plans_and_no_unavailable_expectation(self):
        records=rows();records[0]["plan_id"]="PLANO-ITEM";self.assert_invalid(records,"vazios")
        for field,value in [("plan_id",""),("valid_from",""),("valid_until",""),("value","indisponivel")]:
            records=rows();records[2][field]=value;self.assert_invalid(records)
        records=rows();records[3]["plan_id"]=records[2]["plan_id"];self.assert_invalid(records,"códigos de plano distintos")

    def test_invalid_dates_offsets_and_naive_dates_are_rejected(self):
        for value in ["2026-01-15T08:00:00", "2026-13-15T08:00:00Z", "2026-01-15T08:00:00+00:60",
                      "2026-01-15T08:00:00+14:01", "2026-01-15T08:00:00+15:00", "1999-01-15T08:00:00Z",
                      "2026-01-15T08:00:00.5Z"]:
            with self.subTest(value=value):
                records=rows();records[0]["timestamp"]=value;self.assert_invalid(records)

    def test_bytes_header_encoding_empty_and_cell_bounds(self):
        for payload in [b"", b"\xff", b"a\x00b", b"episode_id,PRIVATE_NAME\n", ALERTS.replace(b",source",b",source,extra",1),
                        b",".join(x.encode() for x in HEADER)+b"\n", b'"unterminated']:
            with self.subTest(payload=payload[:30]),self.assertRaises(CSVValidationError):parse_episode_csv(payload)
        self.assertEqual(parse_episode_csv(b"\xef\xbb\xbf"+ALERTS).episode,parse_episode_csv(ALERTS).episode)
        records=rows();records[0]["source"]="x"*65;self.assert_invalid(records,"64 caracteres")
        records=rows();records[0]["value"]=" ativo";self.assert_invalid(records,"espaços")
        with patch("episodio_internacao_csv.csv.reader") as reader:
            with self.assertRaises(CSVValidationError):parse_episode_csv(b"x"*(MAX_BYTES+1))
            reader.assert_not_called()

    def test_max_rows_is_bounded_before_analysis(self):
        template=rows()[0];start=datetime(2026,1,15,7,tzinfo=timezone.utc)
        records=[dict(template,timestamp=(start+timedelta(seconds=i)).isoformat()) for i in range(MAX_ROWS)]
        self.assertEqual(parse_episode_csv(encode(records)).row_count,MAX_ROWS)
        records.append(dict(template,timestamp=(start+timedelta(seconds=MAX_ROWS)).isoformat()))
        self.assert_invalid(records,"500 registros")

    def test_inconsistent_movement_sequence_is_unavailable_without_clinical_alert(self):
        records=rows();records[-1]["value"]="retorno_leito"
        result=review_episode(parse_episode_csv(encode(records)).episode)
        self.assertEqual(result.movement.rule,"D07")
        self.assertFalse(result.movement.alert)

    def test_parser_has_no_filesystem_or_network_calls(self):
        tree=ast.parse((ROOT/'app/episodio_internacao_csv.py').read_text())
        modules={node.module for node in ast.walk(tree) if isinstance(node,ast.ImportFrom)}
        self.assertLessEqual(modules,{'dataclasses','datetime','episodio_internacao_demo'})
        imports={alias.name for node in ast.walk(tree) if isinstance(node,ast.Import) for alias in node.names}
        self.assertLessEqual(imports,{'csv','hashlib','io','re'})
        calls={node.func.id for node in ast.walk(tree) if isinstance(node,ast.Call) and isinstance(node.func,ast.Name)}
        self.assertFalse(calls & {'open','eval','exec','compile','__import__'})


if __name__ == '__main__':unittest.main()
