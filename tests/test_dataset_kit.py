"""Small media/CSV fixtures only; no downloads, model weights or video bytes."""
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
import wave

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'app'))
sys.path.insert(0, str(ROOT / 'scripts'))
import generate_test_csvs as generate
import prepare_rehab_test_videos as prepare
from audio_local import validate_wav
import monitoramento_demo


class DatasetKitTests(unittest.TestCase):
    def test_synthetic_audio_identity_format_and_silence(self):
        expected = {
            'synthetic_fictional_en.wav': ('2e626dd85110824a2bddfae2950d9e8f1e6f186502fb9b62f4abfa4f89dd040a', False),
            'silence_paired.wav': ('ecf55a63149e0fde4779617b39f5c2768745ddef9d011dbf468620c3a238f1c7', True),
        }
        for name, (digest, silence) in expected.items():
            with self.subTest(name=name):
                payload = (ROOT / 'Datasets/audio' / name).read_bytes()
                self.assertEqual(hashlib.sha256(payload).hexdigest(), digest)
                self.assertEqual(len(payload), 236684)
                sample = validate_wav(payload)
                self.assertEqual(sample.frames, 118320)
                self.assertEqual(sample.seconds, 7.395)
                self.assertEqual(sample.digital_silence, silence)

    def test_csvs_are_generated_deterministically_and_distinct(self):
        for name, varied, alerts in zip(generate.NAMES, (False, True), (0, 4)):
            with self.subTest(name=name):
                payload = (ROOT / 'Datasets/csv' / name).read_bytes()
                self.assertEqual(payload, generate.make_fixture(varied))
                result = generate.validate_fixture(payload, alerts)
                self.assertEqual(result['readings'], 120)
                self.assertEqual(result['insufficient_history'], 25)
                originals = list((Path(monitoramento_demo.__file__).parent / 'monitoramento_exemplos').glob('*.csv'))
                self.assertGreaterEqual(len(originals), 2)
                for old in originals:
                    self.assertNotEqual(payload, old.read_bytes())

    def test_video_recipe_exact_ranges_and_bounded_format(self):
        self.assertEqual([c[3]-c[2] for c in prepare.CLIPS], [144,144,145,145,134])
        for name, identity, start, end, repetition, label in prepare.CLIPS:
            command = prepare.ffmpeg_command('ffmpeg', Path('source.mp4'), Path(name), start, end)
            self.assertIn('-n', command)
            self.assertIn('-nostdin', command)
            self.assertNotIn('-y', command)
            self.assertEqual(command[command.index('-frames:v')+1], str(end-start))
            self.assertEqual(command[command.index('-pix_fmt')+1], 'yuv420p')
            self.assertEqual(command[command.index('-fps_mode')+1], 'cfr')
            self.assertIn('scale=1280:-2', command[command.index('-vf')+1])
            self.assertIn('-an', command)
            self.assertEqual(command[command.index('-map_metadata')+1], '-1')

    def test_wrong_source_is_rejected_without_conversion(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'not-the-source.mp4'
            source.write_bytes(b'not media')
            with self.assertRaises(ValueError):
                prepare.validate_source(source, 'PM012')


if __name__ == '__main__':
    unittest.main()
