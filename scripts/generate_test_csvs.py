"""Create distinct synthetic fixtures using the application's strict CSV schema.

No people, clinical records, downloads, random data or overwritten files.
"""
import argparse
import csv
from datetime import datetime, timedelta, timezone
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'app'))
from monitoramento_demo import HEADER, SIGNALS, analyze, parse_csv

NAMES = ('controle_sintetico_120.csv', 'variacoes_sinteticas_120.csv')


def make_fixture(with_variations=False):
    buffer = io.StringIO(newline='')
    writer = csv.writer(buffer, lineterminator='\n')
    writer.writerow(HEADER)
    start = datetime(2025, 1, 1, 10, tzinfo=timezone.utc)
    for index in range(24):
        step = (0, 1, 0, -1)[index % 4]
        values = {'heart_rate': 78 + step, 'systolic_bp': 122 + step,
                  'diastolic_bp': 77 + step, 'spo2': 98, 'temperature': 36.6}
        if with_variations:
            if index in (12, 19):
                values['heart_rate'] = {12: 108, 19: 116}[index]
            if index in (14, 21):
                values['spo2'] = {14: 90, 21: 89}[index]
        timestamp = (start + timedelta(minutes=2 * index)).isoformat().replace('+00:00', 'Z')
        for signal, spec in SIGNALS.items():
            writer.writerow((timestamp, signal, values[signal], spec.unit))
    return buffer.getvalue().encode('utf-8')


def validate_fixture(payload, expected_alerts):
    result = analyze(parse_csv(payload))
    summary = {'readings': len(result), 'insufficient_history': sum(x.score is None for x in result),
               'alerts': sum(x.review for x in result),
               'alert_readings': [{'timestamp': x.reading.timestamp.isoformat(),
                                   'signal': x.reading.signal, 'value': x.reading.value,
                                   'score': x.score} for x in result if x.review]}
    if summary['readings'] != 120 or summary['insufficient_history'] != 25 or summary['alerts'] != expected_alerts:
        raise ValueError('Fixture outcome changed; inspect before using it.')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True, help='New directory; existing directory is refused.')
    args = parser.parse_args()
    payloads = [make_fixture(False), make_fixture(True)]
    summaries = [validate_fixture(payload, count) for payload, count in zip(payloads, (0, 4))]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, payload in zip(NAMES, payloads):
        with (args.output_dir / name).open('xb') as handle:
            handle.write(payload)
    print(json.dumps(dict(zip(NAMES, summaries)), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
