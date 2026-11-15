"""Testes de software com pontos simulados; não validam pose humana."""
from pathlib import Path
import json
import math
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import video_movimento_local as video


MP4_HEADER = (24).to_bytes(4, "big") + b"ftypisom" + b"\0" * 12


def clip(angle=90, seconds=3.0, invalid=()):
    samples = []
    for index in range(round(seconds * 5)):
        samples.append({"time_seconds": round(index / 5, 6),
                        "angle_degrees": None if index in invalid else angle,
                        "quality_min": 0.95, "reason": "sem_pose" if index in invalid else None})
    return {"metadata": {"width": 640, "height": 480, "fps": 25, "frames": round(seconds * 25), "duration_seconds": seconds}, "samples": samples}


class ValidationTests(unittest.TestCase):
    def test_upload_empty_oversize_and_spoof(self):
        for data in (b"", b"not an mp4", b"x" * (video.MAX_BYTES + 1), b"\0\0\0\1ftyp" + b"x" * 20):
            with self.subTest(size=len(data)), self.assertRaises(video.VideoError):
                video.validate_upload(data)
        video.validate_upload(MP4_HEADER)

    def test_valid_metadata_and_limits(self):
        self.assertEqual(video.validate_metadata(clip()["metadata"])["duration_seconds"], 3)
        for key, value in (("fps", 0), ("fps", math.nan), ("fps", 61), ("frames", 1), ("frames", 99999), ("width", 0), ("height", 9000)):
            meta = clip()["metadata"] | {key: value}
            with self.subTest(key=key, value=value), self.assertRaises(video.VideoError):
                video.validate_metadata(meta)

    def test_options_invalid(self):
        for options in (video.Options(side="other"), video.Options(joint="knee"), video.Options(tolerance_degrees=math.nan), video.Options(min_event_seconds=0), video.Options(min_quality=1)):
            with self.assertRaises(video.VideoError):
                options.validate()

    def test_full_hd_and_long_source_still_rejected(self):
        with self.assertRaisesRegex(video.VideoError, "1280"):
            video.validate_metadata({"width": 1920, "height": 1080, "fps": 30, "frames": 1304})
        with self.assertRaisesRegex(video.VideoError, "2 a 15"):
            video.validate_metadata({"width": 1280, "height": 720, "fps": 30, "frames": 1304})

    def test_model_missing_size_and_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weight.task"
            with self.assertRaises(video.VideoError): video.verify_model(path)
            path.write_bytes(b"bad")
            with self.assertRaises(video.VideoError): video.verify_model(path)
            with patch.object(video, "MODEL_BYTES", 3):
                with self.assertRaises(video.VideoError): video.verify_model(path)


class GeometryTests(unittest.TestCase):
    def test_indexes_joint_and_side_explicit(self):
        self.assertEqual(video.landmark_indexes(video.Options()), (12, 14, 16))
        self.assertEqual(video.landmark_indexes(video.Options(side="left")), (11, 13, 15))
        self.assertEqual(video.landmark_indexes(video.Options(joint="shoulder")), (24, 12, 14))
        self.assertEqual(video.landmark_indexes(video.Options(joint="shoulder", side="left")), (23, 11, 13))
        with self.assertRaises(video.VideoError):
            video.landmark_indexes(video.Options(joint="other"))

    def test_shoulder_both_sides_and_hip_quality(self):
        for side in ("right", "left"):
            options = video.Options(side=side, joint="shoulder")
            hip, shoulder, elbow = video.landmark_indexes(options)
            item = self.pose()
            points = item.pose_landmarks[0]
            for i, (x, y) in zip((hip, shoulder, elbow), ((0.5, 0.9), (0.5, 0.4), (0.8, 0.4))):
                points[i].x, points[i].y = x, y
            self.assertEqual(video.pose_sample(item, 0, 640, 480, options)["angle_degrees"], 90)
            # Punho não participa da métrica do ombro.
            points[16 if side == "right" else 15].visibility = 0
            self.assertEqual(video.pose_sample(item, 0, 640, 480, options)["angle_degrees"], 90)
            points[hip].presence = 0.69
            row = video.pose_sample(item, 0, 640, 480, options)
            self.assertIsNone(row["angle_degrees"])
            self.assertEqual(row["reason"], "baixa_confianca_ou_oclusao")
            points[hip].presence, points[hip].y = 0.9, 1.1
            self.assertIsNone(video.pose_sample(item, 0, 640, 480, options)["angle_degrees"])

    def test_shoulder_angle_is_relative_to_trunk(self):
        down = [(0.5, 0.9), (0.5, 0.4), (0.5, 0.7)]
        raised = [(0.5, 0.9), (0.5, 0.4), (0.5, 0.1)]
        self.assertAlmostEqual(video.image_plane_angle(down, 640, 480), 0)
        self.assertAlmostEqual(video.image_plane_angle(raised, 640, 480), 180)
        # Rotate all pixel vectors: relative angle is not angle from image vertical.
        tilted = [(0.7, 0.7), (0.4, 0.4), (0.6, 0.6)]
        self.assertAlmostEqual(video.image_plane_angle(tilted, 640, 480), 0, places=5)

    def test_right_straight_degenerate_and_nonfinite(self):
        self.assertAlmostEqual(video.elbow_angle([(0, 0), (1, 0), (1, 1)], 640, 480), 90)
        self.assertAlmostEqual(video.elbow_angle([(0, 0), (0.5, 0), (1, 0)], 640, 480), 180)
        self.assertIsNone(video.elbow_angle([(0, 0)] * 3, 640, 480))
        self.assertIsNone(video.elbow_angle([(math.nan, 0), (0, 1), (1, 1)], 640, 480))

    def test_pixel_aspect_correction(self):
        angle = video.elbow_angle([(0, 0), (0.5, 0.5), (1, 0)], 2, 1)
        self.assertAlmostEqual(angle, 126.86989764584402)

    def pose(self, quality=0.9):
        points = [SimpleNamespace(x=0.5, y=0.5, visibility=quality, presence=quality) for _ in range(33)]
        points[12].x, points[12].y = 0.25, 0.5
        points[16].x, points[16].y = 0.5, 0.75
        return SimpleNamespace(pose_landmarks=[points])

    def test_valid_angle_and_quality_filters(self):
        self.assertEqual(video.pose_sample(self.pose(), 1, 640, 480, video.Options())["angle_degrees"], 90)
        low = video.pose_sample(self.pose(0.69), 1, 640, 480, video.Options())
        self.assertIsNone(low["angle_degrees"])
        self.assertEqual(low["reason"], "baixa_confianca_ou_oclusao")
        for poses, reason in (([], "sem_pose"), (self.pose().pose_landmarks * 2, "multiplas_poses")):
            row = video.pose_sample(SimpleNamespace(pose_landmarks=poses), 0, 640, 480, video.Options())
            self.assertEqual(row["reason"], reason)
            self.assertIsNone(row["angle_degrees"])

    def test_missing_or_nonfinite_points(self):
        item = self.pose()
        item.pose_landmarks[0][14].visibility = math.nan
        self.assertEqual(video.pose_sample(item, 0, 1, 1, video.Options())["reason"], "pontos_invalidos")
        item = self.pose()
        item.pose_landmarks[0][12].x = -0.1
        self.assertIsNone(video.pose_sample(item, 0, 1, 1, video.Options())["angle_degrees"])


class ComparisonTests(unittest.TestCase):
    def test_shoulder_report_names_points_and_limits(self):
        report = video.compare(clip(), clip(125), video.Options(joint="shoulder", side="left"))
        self.assertEqual(report["options"]["joint"], "shoulder")
        self.assertEqual(report["angle_definition"]["point_order"], ["hip", "shoulder", "elbow"])
        self.assertEqual(report["angle_definition"]["landmark_indexes"], [23, 11, 13])
        self.assertIn("3D", report["limitations"][0])
        self.assertEqual(report["events"][0]["max_delta_degrees"], 35)

    def test_control_has_no_event(self):
        report = video.compare(clip(), clip(), video.Options())
        self.assertEqual(report["status"], "analisavel")
        self.assertEqual(report["valid_pairs"], 15)
        self.assertEqual(report["events"], [])

    def test_event_persistence_and_timestamps(self):
        report = video.compare(clip(), clip(125), video.Options())
        self.assertEqual(len(report["events"]), 1)
        self.assertEqual(report["events"][0]["start_seconds"], 0)
        self.assertEqual(report["events"][0]["end_seconds"], 2.8)
        self.assertEqual(report["events"][0]["max_delta_degrees"], 35)

    def test_isolated_spike_and_exact_threshold_not_flagged(self):
        observed = clip(110)
        observed["samples"][5]["angle_degrees"] = 150
        self.assertEqual(video.compare(clip(), observed, video.Options())["events"], [])

    def test_missing_samples_break_event_not_interpolated(self):
        observed = clip(125, invalid=(7,))
        report = video.compare(clip(), observed, video.Options())
        self.assertEqual(len(report["events"]), 2)
        self.assertEqual(report["events"][0]["end_seconds"], 1.2)
        self.assertEqual(report["events"][1]["start_seconds"], 1.6)

    def test_insufficient_coverage_is_not_normal(self):
        report = video.compare(clip(), clip(125, invalid=tuple(range(6))), video.Options())
        self.assertEqual(report["status"], "nao_analisavel")
        self.assertEqual(report["events"], [])

    def test_no_pose_and_too_few_points(self):
        report = video.compare(clip(), clip(invalid=tuple(range(15))), video.Options())
        self.assertEqual(report["valid_pairs"], 0)
        self.assertEqual(report["status"], "nao_analisavel")
        short = clip()
        short["samples"] = short["samples"][:9]
        self.assertEqual(video.compare(short, short, video.Options())["status"], "nao_analisavel")

    def test_duration_mismatch_is_rejected(self):
        with self.assertRaises(video.VideoError): video.compare(clip(), clip(seconds=4), video.Options())

    def test_unmatched_timestamps_are_not_scored(self):
        observed = clip()
        for row in observed["samples"]: row["time_seconds"] += 10
        self.assertEqual(video.compare(clip(), observed, video.Options())["valid_pairs"], 0)

    def test_json_has_no_nan(self):
        json.dumps(video.compare(clip(), clip(invalid=(0,)), video.Options()), allow_nan=False)


class ProcessTests(unittest.TestCase):
    def test_process_timeout_cleanup_and_safe_command(self):
        directories = []
        def timeout(args, **kwargs):
            directories.append(Path(args[-1]))
            self.assertIs(kwargs["shell"], False)
            self.assertEqual(kwargs["timeout"], 45)
            self.assertEqual(args[1], "-I")
            raise subprocess.TimeoutExpired(args, 45)
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.task"
            model.write_bytes(b"unit-test-only")
            with patch.object(video, "verify_model"), patch.object(video.subprocess, "run", side_effect=timeout):
                with self.assertRaisesRegex(video.VideoError, "Tempo limite"):
                    video.analyze_pair(MP4_HEADER, MP4_HEADER, model)
        self.assertTrue(directories)
        self.assertTrue(all(not path.exists() for path in directories))

    def test_process_success_and_input_hashes(self):
        directories = []
        def run(args, **kwargs):
            root = Path(args[-1])
            directories.append(root)
            (root / "result.json").write_text(json.dumps(clip()), encoding="utf-8")
            return SimpleNamespace(returncode=0)
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.task"
            model.write_bytes(b"unit-test-only")
            with patch.object(video, "verify_model"), patch.object(video.subprocess, "run", side_effect=run):
                report = video.analyze_pair(MP4_HEADER, MP4_HEADER, model)
        self.assertEqual(report["reference_sha256"], report["observed_sha256"])
        self.assertTrue(report["same_input_control"])
        self.assertTrue(all(not path.exists() for path in directories))

    def test_worker_error_does_not_report_success(self):
        def run(args, **kwargs):
            (Path(args[-1]) / "result.json").write_text(json.dumps({"error": "Vídeo inválido."}), encoding="utf-8")
            return SimpleNamespace(returncode=1)
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.task"
            model.write_bytes(b"unit-test-only")
            with patch.object(video, "verify_model"), patch.object(video.subprocess, "run", side_effect=run):
                with self.assertRaisesRegex(video.VideoError, "Vídeo inválido"):
                    video.analyze_pair(MP4_HEADER, MP4_HEADER, model)


if __name__ == "__main__":
    unittest.main()
