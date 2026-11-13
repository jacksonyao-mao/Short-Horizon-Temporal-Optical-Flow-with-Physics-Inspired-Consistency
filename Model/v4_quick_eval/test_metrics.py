"""Small deterministic tests; no KITTI data, GPU, or trained checkpoint needed."""
import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path

import torch

from evaluate import recover_rgb, select_samples
from compare import compare_runs
from metrics import flow_statistics, paired_bootstrap, sample_at_flow, summarize, two_step_statistics


def l1_map(src, tgt, flow):
    warped, _ = sample_at_flow(tgt, flow)
    return (src - warped).abs().mean(1)


def row(scene, error, pixels=1):
    pred = torch.zeros(2, 1, pixels)
    pred[0] = error
    out = flow_statistics(pred, torch.zeros_like(pred), torch.ones(1, pixels), torch.zeros(1, pixels))
    out.update(sample_id=scene + "_10", scene_id=scene,
               two_step_l1_sum=0., two_step_photo_sum=0., two_step_pixels=1,
               two_step_candidate_pixels=1)
    return out


class MetricTests(unittest.TestCase):
    def test_outlier_requires_both_thresholds_and_validity(self):
        gt = torch.zeros(2, 1, 4)
        gt[0, 0, 1] = 100
        pred = gt.clone()
        pred[0] += torch.tensor([[4., 4., 3., 100.]])
        stats = flow_statistics(pred, gt, torch.tensor([[1., 1., 1., 0.]]),
                                torch.tensor([[1, 1, 0, 0]]))
        self.assertEqual(stats["all_bad_pixels"], 1)
        self.assertEqual(stats["fg_pixels"], 2)
        self.assertAlmostEqual(stats["fl_fg_percent"], 50.)
        self.assertEqual(stats["bg_pixels"], 1)
        self.assertAlmostEqual(stats["epe_all"], 11/3)

    def test_empty_foreground_is_undefined_not_zero(self):
        self.assertIsNone(row("000001", 2)["epe_fg"])

    def test_global_average_is_pixel_weighted(self):
        result = summarize([row("000001", 1, 1), row("000002", 3, 3)])
        self.assertAlmostEqual(result["epe_all"], 2.5)
        self.assertAlmostEqual(result["per_image_epe_mean"], 2.)

    def test_nan_prediction_rejected_even_outside_gt_mask(self):
        pred = torch.full((2, 1, 1), float("nan"))
        with self.assertRaises(ValueError):
            flow_statistics(pred, torch.zeros_like(pred), torch.zeros(1, 1), torch.zeros(1, 1))

    def test_two_step_translation_direction_and_coverage(self):
        h, w = 5, 9
        x = torch.arange(w).float()[None].repeat(h, 1)
        rgb = torch.stack([(x - t).clamp(min=0)[None].repeat(3, 1, 1) / w for t in range(4)])
        flows = torch.zeros(3, 2, h, w)
        flows[:, 0] = 1
        stats = two_step_statistics(rgb, flows, l1_map, radius=0)
        self.assertLess(stats["two_step_l1"], 1e-6)
        self.assertEqual(stats["two_step_pixels"], 2 * h * (w - 2))

    def test_intermediate_out_of_bounds_cannot_reenter(self):
        rgb = torch.zeros(3, 3, 3, 5)
        flows = torch.zeros(2, 2, 3, 5)
        flows[0, 0] = 2
        flows[1, 0] = -2
        stats = two_step_statistics(rgb, flows, l1_map, radius=0)
        self.assertEqual(stats["two_step_pixels"], 9)  # final shift zero, but first step clips 2 columns

    def test_photometric_neighborhood_erosion(self):
        rgb = torch.zeros(3, 3, 9, 9)
        flows = torch.zeros(2, 2, 9, 9)
        stats = two_step_statistics(rgb, flows, l1_map, radius=3)
        self.assertEqual(stats["two_step_pixels"], 9)

    def test_rgb_denormalization(self):
        rgb = torch.rand(4, 3, 5, 5)
        stats = {"mean": [0.3, 0.4, 0.5], "std": [0.2, 0.3, 0.4]}
        mean = torch.tensor(stats["mean"]).view(1, 3, 1, 1)
        std = torch.tensor(stats["std"]).view(1, 3, 1, 1)
        actual = recover_rgb((rgb - mean) / std, {"normalize": True}, stats)
        torch.testing.assert_close(actual, rgb)

    def test_bootstrap_pairs_ids_not_row_positions(self):
        a = [row("000001", 1), row("000002", 3)]
        b = [row("000002", 4), row("000001", 2)]
        result = paired_bootstrap(a, b, repeats=200)
        self.assertEqual(result["metrics"]["epe_all"]["ci95"], [-1., -1.])
        self.assertIsNone(result["metrics"]["epe_fg"]["ci95"])

    def test_bootstrap_rejects_missing_ids_and_changed_masks(self):
        a = [row("000001", 1), row("000002", 3)]
        with self.assertRaises(ValueError):
            paired_bootstrap(a, a[:1], repeats=200)
        b = copy.deepcopy(a)
        b[0]["fg_pixels"] = 1
        with self.assertRaises(ValueError):
            paired_bootstrap(a, b, repeats=200)

    def test_held_out_rejects_scene_overlap(self):
        samples = [{"seq_id": "000001", "flow_path": "000001_10.png"},
                   {"seq_id": "000002", "flow_path": "000002_10.png"}]
        with self.assertRaises(ValueError):
            select_samples(samples, {"000001_10"}, {"000001_11"}, "held-out")
        selected = select_samples(samples, {"000002"}, {"000001"}, "held-out")
        self.assertEqual(len(selected), 1)
        with self.assertRaises(ValueError):
            select_samples(samples, {"unknown"}, None, "training-diagnostic")

    def test_comparison_export_and_protocol_guards(self):
        with tempfile.TemporaryDirectory() as folder:
            directories = [Path(folder) / name for name in ("a", "b")]
            rows = [row("000001", 1), row("000002", 3)]
            for r in rows:
                r["sample_signature"] = r["sample_id"]
            summary = {"num_samples": 2, "evaluation_protocol": {"split": "training"},
                       "metrics": {"two_step_coverage_percent": 100.}}
            for d in directories:
                d.mkdir()
                (d / "summary.json").write_text(json.dumps(summary))
                with (d / "per_scene.csv").open("w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
            result = compare_runs(*directories, repeats=200)
            self.assertEqual(result["metrics"]["epe_all"]["ci95"], [0., 0.])
            json.dumps(result, allow_nan=False)
            summary["evaluation_protocol"]["split"] = "validation"
            (directories[1] / "summary.json").write_text(json.dumps(summary))
            with self.assertRaises(ValueError):
                compare_runs(*directories, repeats=200)


if __name__ == "__main__":
    unittest.main()
