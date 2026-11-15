"""Deterministic tests. Mocked adapters are NOT evidence of ASR accuracy."""

import importlib
import io
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from audio_local import (
    AudioError, MAX_BYTES, MAX_SEGMENTS, Segment, selection_key, sync_selection,
    timestamp, transcribe_local, validate_model_dir, validate_wav,
)
from audio_text import sentiment_summary, term_rows


def wav_bytes(seconds=1, channels=1, rate=16000, width=2, value=100):
    out = io.BytesIO()
    with wave.open(out, "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(width)
        writer.setframerate(rate)
        frame = struct.pack("<h", value) if width == 2 else bytes([value] * width)
        writer.writeframes(frame * int(seconds * rate) * channels)
    return out.getvalue()


def edit_u32(data, offset, value):
    result = bytearray(data)
    struct.pack_into("<I", result, offset, value)
    return bytes(result)


class WavTests(unittest.TestCase):
    def test_canonical_valid_and_fingerprint(self):
        sample = validate_wav(wav_bytes())
        self.assertEqual(sample.seconds, 1)
        self.assertEqual(len(sample.fingerprint), 64)
        self.assertFalse(sample.digital_silence)

    def test_minimum_maximum_duration(self):
        self.assertEqual(validate_wav(wav_bytes(0.25)).seconds, 0.25)
        self.assertEqual(validate_wav(wav_bytes(60)).seconds, 60)
        for seconds in [0, 0.249, 60.01]:
            with self.subTest(seconds=seconds), self.assertRaises(AudioError):
                validate_wav(wav_bytes(seconds))

    def test_bytes_budget_and_wrong_type(self):
        for data in [b"x" * (MAX_BYTES + 1), bytearray(wav_bytes()), None]:
            with self.subTest(type=type(data)), self.assertRaises(AudioError):
                validate_wav(data)

    def test_wrong_format_channels_rates_bits(self):
        for options in [dict(channels=2), dict(rate=44100), dict(width=1), dict(width=3)]:
            with self.subTest(options=options), self.assertRaises(AudioError):
                validate_wav(wav_bytes(**options))
        raw = bytearray(wav_bytes())
        struct.pack_into("<H", raw, 20, 3)
        with self.assertRaises(AudioError):
            validate_wav(bytes(raw))

    def test_magic_bad_size_truncation_and_trailing_bytes(self):
        good = wav_bytes()
        for bad in [b"not a wav", b"RF64" + good[4:], good[:-2], good + b"xx",
                    edit_u32(good, 40, 2), edit_u32(good, 28, 32002)]:
            with self.subTest(prefix=bad[:12]), self.assertRaises(AudioError):
                validate_wav(bad)

    def test_extra_chunk_extended_format_and_odd_data(self):
        good = wav_bytes()
        extra = good[:36] + b"JUNK\x00\x00\x00\x00" + good[36:]
        extra = edit_u32(extra, 4, len(extra) - 8)
        odd = edit_u32(edit_u32(good + b"x", 4, len(good) - 7), 40, len(good) - 43)
        for bad in [extra, edit_u32(good, 16, 18), odd]:
            with self.assertRaises(AudioError):
                validate_wav(bad)

    def test_digital_silence_is_not_asr(self):
        sample = validate_wav(wav_bytes(value=0))
        self.assertTrue(sample.digital_silence)
        with patch("audio_local.importlib.import_module") as importer:
            with self.assertRaisesRegex(AudioError, "inteiramente zerado"):
                transcribe_local(sample.payload, "/not-a-model")
            importer.assert_not_called()

    def test_exact_lfs_placeholder_has_actionable_error(self):
        pointer = (b"version https://git-lfs.github.com/spec/v1\n"
                   b"oid sha256:" + b"a" * 64 + b"\nsize 12345678\n")
        self.assertEqual(len(pointer), 133)
        for data in [pointer, pointer.replace(b"\n", b"\r\n")]:
            with self.subTest(size=len(data)), self.assertRaisesRegex(AudioError, "ponteiro Git LFS"):
                validate_wav(data)
        # Only the exact pointer grammar gets this explanation, not arbitrary text.
        for data in [pointer + b"extra", pointer.replace(b"oid sha256:", b"oid other:"), b"git lfs"]:
            with self.subTest(size=len(data)), self.assertRaises(AudioError) as raised:
                validate_wav(data)
            self.assertNotIn("ponteiro Git LFS", str(raised.exception))


class AdapterTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.path = Path(folder.name)
        for name in ["model.bin", "config.json", "tokenizer.json"]:
            (self.path / name).write_text("{}")
        self.payload = wav_bytes()

    def fake_engine(self, segments=None, exception=None, languages=("pt", "en")):
        # Explicit test double. No ASR weights or actual inference in unit tests.
        if segments is None:
            segments = [Segment(0.0, 0.8, "Sem dor no peito.")]
        calls = {}

        class Engine:
            supported_languages = languages
            feature_extractor = SimpleNamespace(sampling_rate=16000)

            def __init__(self, *args, **kwargs):
                calls["constructor"] = (args, kwargs)

            def transcribe(self, waveform, **kwargs):
                calls["waveform"] = waveform
                calls["transcribe"] = kwargs

                def generate():
                    if exception:
                        raise exception
                    yield from segments
                return generate(), SimpleNamespace(language="pt")

        original = importlib.import_module
        stub = SimpleNamespace(WhisperModel=Engine)
        guard = patch("audio_local.importlib.import_module", side_effect=lambda name: stub if name == "faster_whisper" else original(name))
        return guard, calls

    def test_model_requires_existing_absolute_directory(self):
        for value in ["", "tiny", "./tiny", "https://example.com/model", "//server/model", "\\\\server\\model", str(self.path / "missing")]:
            with self.subTest(value=value), self.assertRaises(AudioError):
                validate_model_dir(value)
        self.assertEqual(validate_model_dir(str(self.path)), self.path)

    def test_tokenizer_is_mandatory_prevents_network_fallback(self):
        (self.path / "tokenizer.json").unlink()
        with patch("audio_local.importlib.import_module") as importer:
            with self.assertRaisesRegex(AudioError, "tokenizer.json"):
                transcribe_local(self.payload, str(self.path))
            importer.assert_not_called()

    def test_wrong_preprocessor_rate_and_corruption(self):
        for data in ['{"sampling_rate":8000}', '[]', '{bad', 'x' * 65537]:
            (self.path / "preprocessor_config.json").write_text(data)
            with self.assertRaises(AudioError):
                validate_model_dir(str(self.path))

    def test_optional_dependency_missing_is_actionable(self):
        with patch("audio_local.importlib.import_module", side_effect=ImportError):
            with self.assertRaisesRegex(AudioError, "mesmo ambiente"):
                transcribe_local(self.payload, str(self.path))

    def test_adapter_cpu_local_files_array_and_timestamps(self):
        guard, calls = self.fake_engine()
        with guard:
            result = transcribe_local(self.payload, str(self.path))
        self.assertEqual(result.text, "Sem dor no peito.")
        self.assertEqual(calls["constructor"][0], (str(self.path),))
        self.assertEqual(calls["constructor"][1]["local_files_only"], True)
        self.assertEqual(calls["constructor"][1]["device"], "cpu")
        self.assertEqual(calls["waveform"].dtype.name, "float32")
        self.assertEqual(len(calls["waveform"]), 16000)
        self.assertFalse(calls["transcribe"]["vad_filter"])
        self.assertFalse(calls["transcribe"]["condition_on_previous_text"])
        self.assertNotIn("initial_prompt", calls["transcribe"])
        self.assertNotIn("hotwords", calls["transcribe"])

    def test_invalid_audio_never_imports_model(self):
        with patch("audio_local.importlib.import_module") as importer:
            with self.assertRaises(AudioError):
                transcribe_local(b"bad", str(self.path))
            importer.assert_not_called()

    def test_language_mismatch_is_explicit(self):
        guard, _ = self.fake_engine(languages=("en",))
        with guard, self.assertRaisesRegex(AudioError, "multilíngues"):
            transcribe_local(self.payload, str(self.path), "pt")
        with self.assertRaises(AudioError):
            transcribe_local(self.payload, str(self.path), "auto")

    def test_empty_transcript_is_not_successful_text(self):
        guard, _ = self.fake_engine(segments=[])
        with guard:
            result = transcribe_local(self.payload, str(self.path))
        self.assertEqual(result.text, "")
        self.assertEqual(result.segments, ())

    def test_lazy_inference_failure_clears_partial_result(self):
        guard, _ = self.fake_engine(exception=RuntimeError("private detail"))
        with guard, self.assertRaisesRegex(AudioError, "Nenhum resultado parcial") as raised:
            transcribe_local(self.payload, str(self.path))
        self.assertNotIn("private detail", str(raised.exception))

    def test_invalid_output_timestamps_and_bounds(self):
        cases = [[Segment(float("nan"), 1, "x")], [Segment(0, float("inf"), "x")],
                 [Segment(0, 5, "x")], [Segment(0, 0, "x")],
                 [Segment(0.5, 0.6, "x"), Segment(0.1, 0.2, "y")],
                 [Segment(0, 1, "x" * 20001)], [Segment(0, 1, "")] * (MAX_SEGMENTS + 1)]
        for segments in cases:
            guard, _ = self.fake_engine(segments)
            with self.subTest(segments=len(segments)), guard, self.assertRaises(AudioError):
                transcribe_local(self.payload, str(self.path))


class TextAndStateTests(unittest.TestCase):
    def rows(self, text):
        return term_rows([Segment(0, 1, text)])

    def test_accents_original_evidence_and_negation(self):
        rows = self.rows("NÃO sinto dor torácica, mas tenho palpitações.")
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["Expressão encontrada"], "dor torácica")
        self.assertEqual(rows[0]["Leitura lexical"], "Menção com pista de negação")
        self.assertEqual(rows[1]["Leitura lexical"], "Menção sem pista de negação")

    def test_uncertainty_and_clause_reset(self):
        self.assertEqual(self.rows("Não sei se é dor no peito.")[0]["Leitura lexical"], "Menção com pista de incerteza")
        self.assertEqual(self.rows("Sem febre. Dor no peito.")[0]["Leitura lexical"], "Menção sem pista de negação")

    def test_english_boundaries_and_no_matches(self):
        self.assertEqual(self.rows("No chest pain.")[0]["Leitura lexical"], "Menção com pista de negação")
        self.assertEqual(self.rows("A word with no clinical term."), [])
        self.assertEqual(self.rows("tonturando"), [])

    def test_scope_does_not_cross_asr_segments(self):
        rows = term_rows([Segment(0, 1, "Não"), Segment(1, 2, "dor no peito")])
        self.assertEqual(rows[0]["Leitura lexical"], "Menção sem pista de negação")

    def test_lexical_sentiment_not_clinical_or_default_neutral(self):
        self.assertIsNone(sentiment_summary("dor no peito")['score'])
        result = sentiment_summary("Estou bem. Tenho medo.")
        self.assertEqual(result['score'], 0)
        self.assertEqual(result['positive'], ['bem'])
        self.assertEqual(result['negative'], ['medo'])
        result = sentiment_summary("Não estou bem.")
        self.assertEqual(result['excluded'], ['bem'])
        self.assertIsNone(result['score'])

    def test_selection_invalidates_on_file_model_language(self):
        key = selection_key(b"a", "/model", "pt")
        for newer in [selection_key(b"b", "/model", "pt"),
                      selection_key(b"a", "/other", "pt"),
                      selection_key(b"a", "/model", "en"), None]:
            state = {"_audio_selection": key, "_audio_result": "old", "_audio_preview": "old", "_audio_error": "old"}
            sync_selection(state, newer)
            self.assertNotIn("_audio_result", state)
            self.assertNotIn("_audio_preview", state)
            self.assertNotIn("_audio_error", state)

    def test_timestamp_rounding(self):
        self.assertEqual(timestamp(0), "00:00.000")
        self.assertEqual(timestamp(59.9996), "01:00.000")


if __name__ == "__main__":
    unittest.main()
