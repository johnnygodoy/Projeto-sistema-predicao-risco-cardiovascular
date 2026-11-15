"""Executar: python -m unittest discover -s tests -p 'test_monitoramento*.py' -v"""

import csv
from datetime import datetime, timedelta, timezone
import io
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
from monitoramento_demo import (
    HEADER, MAX_BYTES, MAX_ROWS, SIGNALS, ValidationError, analyze, chart_rows, parse_csv,
    selection_key, sync_selection,
)


def make_csv(values=(72,) * 12, signal="heart_rate", unit="bpm", gap=False):
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(HEADER)
    start = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)
    for n, value in enumerate(values):
        time = start + timedelta(minutes=n) + (timedelta(hours=7) if gap and n >= 6 else timedelta())
        writer.writerow([time.isoformat().replace("+00:00", "Z"), signal, value, unit])
    return buffer.getvalue().encode()


class ParserTests(unittest.TestCase):
    def assert_invalid(self, content):
        with self.assertRaises(ValidationError):
            parse_csv(content)

    def test_valid_csv_bom_crlf_and_utc(self):
        payload = b"\xef\xbb\xbf" + make_csv().replace(b"\n", b"\r\n")
        dataset = parse_csv(payload)
        self.assertEqual(len(dataset.readings), 12)
        self.assertEqual(dataset.readings[0].timestamp.tzinfo, timezone.utc)
        self.assertEqual(dataset, parse_csv(payload))

    def test_offsets_normalized_before_order_check(self):
        payload = make_csv((72, 73)).replace(b"12:00:00Z", b"09:00:00-03:00")
        self.assertEqual(parse_csv(payload).readings[0].timestamp.hour, 12)

    def test_all_units_and_sanity_boundaries(self):
        for signal, spec in SIGNALS.items():
            with self.subTest(signal=signal):
                data = parse_csv(make_csv((spec.minimum, spec.maximum), signal, spec.unit))
                self.assertEqual(len(data.readings), 2)
                self.assert_invalid(make_csv((spec.minimum - 1,), signal, spec.unit))
                self.assert_invalid(make_csv((spec.maximum + 1,), signal, spec.unit))

    def test_header_empty_missing_extra_duplicate_and_order(self):
        valid = make_csv()
        for header in [b"", b"timestamp,signal,value", b"timestamp,signal,value,unit,name",
                       b"timestamp,signal,value,value", b"signal,timestamp,value,unit",
                       b"timestamp;signal;value;unit"]:
            with self.subTest(header=header):
                self.assert_invalid(header + b"\n" + valid.split(b"\n", 1)[1])
        self.assert_invalid(b",")
        self.assert_invalid(b"timestamp,signal,value,unit\n")
        self.assert_invalid(b"")

    def test_byte_and_row_limits(self):
        self.assert_invalid(b"x" * (MAX_BYTES + 1))
        self.assertEqual(len(parse_csv(make_csv([72] * MAX_ROWS)).readings), MAX_ROWS)
        self.assert_invalid(make_csv([72] * (MAX_ROWS + 1)))

    def test_encoding_null_bad_csv_long_cell_blank_row(self):
        for content in [b"\xff", make_csv() + b"\x00", make_csv() + b'"unclosed',
                        make_csv() + b"x" * 65 + b",heart_rate,72,bpm\n",
                        make_csv() + b"\n", "texto" ]:
            with self.subTest(type=type(content)):
                self.assert_invalid(content)

    def test_missing_cell_extra_cell_spaces_and_formulas(self):
        for changed in [b"", b" ", b" 72", b"72 ", b"=1+71", b"+72", b"@SUM(1)",
                        b"NaN", b"inf", b"1e2", b"-1", b"72.1234", b"72,5"]:
            with self.subTest(value=changed):
                self.assert_invalid(make_csv((72,)).replace(b",72,", b"," + changed + b","))
        self.assert_invalid(make_csv() + b"2026-01-02T12:00:00Z,heart_rate,72,bpm,extra\n")

    def test_signal_and_unit_are_strict(self):
        self.assert_invalid(make_csv(signal="name"))
        self.assert_invalid(make_csv(unit="BPM"))
        self.assert_invalid(make_csv(signal="temperature", unit="F", values=(37,)))

    def test_invalid_timestamps(self):
        values = ["2026-01-01", "2026-01-01T12:00:00", "2026-01-01T12:00Z",
                  "2026-02-30T12:00:00Z", "2026-01-01T25:00:00Z",
                  "2026-01-01T12:00:00+02:60", "2026-01-01T12:00:00+14:01",
                  "2026-01-01T12:00:00+15:00", "0001-01-01T00:00:00+14:00",
                  "9999-12-31T23:59:59-14:00", "1999-01-01T12:00:00Z"]
        for value in values:
            with self.subTest(value=value):
                self.assert_invalid(make_csv((72,)).replace(b"2026-01-01T12:00:00Z", value.encode()))

    def test_duplicate_out_of_order_and_equivalent_instants(self):
        lines = make_csv((72, 73)).splitlines(keepends=True)
        self.assert_invalid(lines[0] + lines[1] + lines[1])
        self.assert_invalid(lines[0] + lines[2] + lines[1])
        self.assert_invalid(lines[0] + lines[1] + lines[1].replace(b"12:00:00Z", b"09:00:00-03:00"))

    def test_same_timestamp_different_signals_allowed(self):
        line = make_csv((72,)).splitlines(keepends=True)
        dataset = parse_csv(b"".join(line) + line[1].replace(b"heart_rate,72,bpm", b"spo2,98,%"))
        self.assertEqual(len(dataset.readings), 2)

    def test_quality_notices(self):
        self.assertIn("histórico insuficiente", parse_csv(make_csv((72,))).notices[0])
        self.assertIn("reinicia", parse_csv(make_csv(gap=True)).notices[0])

    def test_rejection_does_not_echo_user_content(self):
        with self.assertRaises(ValidationError) as context:
            parse_csv(make_csv(signal="IDENTIFIER_SHOULD_NOT_APPEAR"))
        self.assertNotIn("IDENTIFIER_SHOULD_NOT_APPEAR", str(context.exception))


class BaselineTests(unittest.TestCase):
    def test_fixture_alerts_and_control(self):
        for filename, expected in [("variacoes_ficticias.csv", 4), ("controle_ficticio.csv", 0)]:
            with self.subTest(filename=filename):
                data = parse_csv((ROOT / "app/monitoramento_exemplos" / filename).read_bytes())
                results = analyze(data)
                self.assertEqual(len(results), 90)
                self.assertEqual(sum(r.review for r in results), expected)
                self.assertEqual(sum(r.score is None for r in results), 25)
                self.assertEqual(results, analyze(data))

    def test_warmup_then_uses_only_previous_measurements(self):
        results = analyze(parse_csv(make_csv((72, 72, 72, 72, 72, 120))))
        self.assertTrue(all(r.score is None for r in results[:5]))
        self.assertEqual(results[5].baseline, 72)
        self.assertEqual(results[5].scale, 3)
        self.assertEqual(results[5].score, 16)
        self.assertTrue(results[5].review)

    def test_no_future_leakage(self):
        a = analyze(parse_csv(make_csv((72,) * 10 + (120,))))
        b = analyze(parse_csv(make_csv((72,) * 10 + (40,))))
        self.assertEqual(a[:-1], b[:-1])
        self.assertEqual(a[-1].baseline, b[-1].baseline)
        prefix = analyze(parse_csv(make_csv((72,) * 10)))
        self.assertEqual(a[:-1], prefix)

    def test_window_size_capped_and_previous_outlier_expires(self):
        results = analyze(parse_csv(make_csv((150,) * 5 + (72,) * 12)))
        self.assertEqual(results[-1].history_count, 10)
        self.assertEqual(results[-1].baseline, 72)
        self.assertFalse(results[-1].review)

    def test_threshold_is_strictly_greater(self):
        results = analyze(parse_csv(make_csv((72,) * 5 + (82.5,))))
        self.assertEqual(results[-1].score, 3.5)
        self.assertFalse(results[-1].review)

    def test_gap_resets_history(self):
        results = analyze(parse_csv(make_csv(gap=True)))
        self.assertEqual(results[5].history_count, 5)
        self.assertEqual(results[6].history_count, 0)
        self.assertTrue(all(r.score is None for r in results[6:11]))
        self.assertEqual(results[11].history_count, 5)

    def test_signals_are_independent(self):
        data = parse_csv((ROOT / "app/monitoramento_exemplos/variacoes_ficticias.csv").read_bytes())
        results = analyze(data)
        self.assertEqual({r.reading.signal for r in results if r.review}, {"heart_rate", "spo2"})

    def test_flat_extreme_series_is_not_a_safety_test(self):
        results = analyze(parse_csv(make_csv((200,) * 12)))
        self.assertFalse(any(r.review for r in results))

    def test_chart_keeps_values_and_missing_reference_without_substitution(self):
        results = analyze(parse_csv(make_csv((72, 73, 71, 74, 72, 120))))
        rows = chart_rows(results, "heart_rate")
        self.assertEqual([row["Valor"] for row in rows], [72, 73, 71, 74, 72, 120])
        self.assertTrue(all(row["Mediana anterior"] is None for row in rows[:5]))
        self.assertEqual([row["Unidade"] for row in rows], ["bpm"] * 6)
        self.assertTrue(rows[-1]["Revisar"])
        self.assertEqual(len(chart_rows(results, "spo2")), 0)

    def test_chart_splits_long_gap_and_does_not_invent_samples(self):
        results = analyze(parse_csv(make_csv(gap=True)))
        rows = chart_rows(results, "heart_rate")
        self.assertEqual(len(rows), 12)
        self.assertEqual([row["Segmento"] for row in rows], [0] * 6 + [1] * 6)
        self.assertTrue(all(row["Mediana anterior"] is None for row in rows[6:11]))


class SelectionTests(unittest.TestCase):
    def test_change_or_removal_clears_preview_and_results(self):
        for changed in [None, selection_key("upload", make_csv((73,))), selection_key("other", make_csv())]:
            state = {"_mon_selected": selection_key("upload", make_csv()),
                     "_mon_preview": object(), "_mon_results": object(), "unrelated": 42}
            sync_selection(state, changed)
            self.assertNotIn("_mon_preview", state)
            self.assertNotIn("_mon_results", state)
            self.assertEqual(state["unrelated"], 42)

    def test_same_selection_keeps_state(self):
        key = selection_key("example", make_csv())
        state = {"_mon_selected": key, "_mon_preview": "preview", "_mon_results": "result"}
        sync_selection(state, key)
        self.assertEqual(state["_mon_results"], "result")


if __name__ == "__main__":
    unittest.main()
