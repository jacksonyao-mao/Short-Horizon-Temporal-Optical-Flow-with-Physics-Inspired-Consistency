#!/usr/bin/env python3
"""One inference evaluation per v4 checkpoint, followed by paired scene bootstrap.

Uses the reviewed v4_quick_eval package beside this script. Does not train models
or modify checkpoints. Run from model_v4_Flychairs after all requested checkpoints
exist. Output includes per-scene sufficient statistics, a combined table, and
pairwise confidence intervals. Reconstruction support is model-specific; report
two-step coverage alongside the reconstruction scores.
"""

import argparse
import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parent.parent
ARCHITECTURES = ("later", "early", "standalone", "cnn")


def file_fingerprint(path):
    """Match v4_quick_eval's length-delimited file-content fingerprint."""
    digest = hashlib.sha256()
    path = Path(path)
    digest.update(str(path.stat().st_size).encode() + b"\0")
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_to_v4(path, v4):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (v4 / path).resolve()


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--v4-dir", default=str(ROOT))
    p.add_argument("--data-root", default="Data", help="Data root with Additional_frames/ and Flow/.")
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--suffix", default="from_chairs")
    p.add_argument("--architectures", nargs="+", choices=ARCHITECTURES,
                   default=["later", "early", "standalone"])
    p.add_argument("--output-dir", default="evaluation_outputs/extended_metrics")
    p.add_argument("--device", default="cuda", help="cuda, cuda:0, or cpu. CUDA_VISIBLE_DEVICES is respected.")
    p.add_argument("--split", default=None, help="Defaults to each checkpoint's recorded disk split.")
    p.add_argument("--stats-file", default=None, help="Optional shared training-statistics override.")
    p.add_argument("--obj-map-dir", default=None)
    p.add_argument("--protocol", choices=("training-diagnostic", "held-out"), default="training-diagnostic")
    p.add_argument("--ids-file", default=None)
    p.add_argument("--training-ids-file", default=None)
    p.add_argument("--input-mask", choices=("annotation", "ones"), default="annotation")
    p.add_argument("--max-samples", type=int, default=None, help="Debug subset; outputs are marked partial.")
    p.add_argument("--repeats", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--reuse-existing", action="store_true",
                   help="Reuse completed per-scene outputs after checkpoint/protocol checks; no repeated inference.")
    p.add_argument("--compare-only", action="store_true",
                   help="Read existing per-scene outputs and recompute table/bootstrap; no checkpoint loading or inference.")
    return p


def check_completed(directory, checkpoint, args, data_root, quick):
    summary_path = directory / "summary.json"
    csv_path = directory / "per_scene.csv"
    if not summary_path.is_file() or not csv_path.is_file():
        raise ValueError(f"Incomplete evaluation in {directory}; use a new output directory.")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    protocol = summary["evaluation_protocol"]
    if summary["checkpoint_sha256"] != file_fingerprint(checkpoint):
        raise ValueError(f"Saved checkpoint fingerprint differs: {directory}")
    if protocol["metric_source_sha256"] != file_fingerprint(quick / "metrics.py"):
        raise ValueError(f"Metric implementation changed; re-evaluate into a new directory: {directory}")
    if Path(summary["data_root"]).resolve() != data_root:
        raise ValueError(f"Data root differs from completed run: {directory}")
    expected = {"protocol": args.protocol,
                "input_mask": args.input_mask if summary["config"]["model"].get("uno_use_valid_mask") else "unused",
                "partial_run": args.max_samples is not None}
    if args.split is not None:
        expected["split"] = args.split
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(f"Completed run has different {key}: {directory}")
    if args.max_samples is not None and summary["num_samples"] != args.max_samples:
        raise ValueError(f"Completed subset size differs from --max-samples: {directory}")
    # Explicit ID/statistics overrides require a fresh run: equal filenames alone
    # cannot prove that their contents have not changed since evaluation.
    if any((args.ids_file, args.training_ids_file, args.stats_file, args.obj_map_dir)):
        raise ValueError("Do not combine --reuse-existing with ID/statistics/object-map overrides; use --compare-only for unchanged saved results.")
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != summary["num_samples"]:
        raise ValueError(f"CSV row count differs from saved summary: {directory}")
    return summary


def run(args):
    if len(set(args.architectures)) != len(args.architectures):
        raise ValueError("Architectures must be distinct.")
    if args.repeats < 100 or (args.max_samples is not None and args.max_samples < 1):
        raise ValueError("Use >=100 bootstrap repeats and a positive sample limit.")
    v4 = Path(args.v4_dir).resolve()
    quick = v4 / "v4_quick_eval"
    for name in ("evaluate.py", "metrics.py", "compare.py"):
        if not (quick / name).is_file():
            raise FileNotFoundError(f"Required evaluator file missing: {quick / name}")
    output = relative_to_v4(args.output_dir, v4)
    data_root = relative_to_v4(args.data_root, v4)
    checkpoints = relative_to_v4(args.checkpoint_dir, v4)
    paths = {arch: checkpoints / f"fullpipeline_{arch}_kitti_{args.suffix}_best.pth"
             for arch in args.architectures}
    if not args.compare_only:
        for path in paths.values():
            if not path.is_file():
                raise FileNotFoundError(f"Checkpoint missing: {path}. Finish/copy that run, or select available --architectures.")
        if args.protocol == "held-out" and (not args.ids_file or not args.training_ids_file):
            raise ValueError("held-out requires --ids-file and --training-ids-file.")
    output.mkdir(parents=True, exist_ok=True)
    commands = []
    for arch, checkpoint in paths.items():
        directory = output / arch
        if args.compare_only:
            continue
        if (directory / "summary.json").exists() or (directory / "per_scene.csv").exists():
            if not args.reuse_existing:
                raise FileExistsError(f"Results already exist in {directory}; use --reuse-existing, --compare-only, or a new --output-dir.")
            summary = check_completed(directory, checkpoint, args, data_root, quick)
            if summary["architecture"] != arch:
                raise ValueError(f"Completed architecture differs from {arch}: {directory}")
            print(f"[Reuse] {arch}: {directory}", flush=True)
            continue
        command = [sys.executable, str(quick / "evaluate.py"), "--v4-dir", str(v4),
                   "--checkpoint", str(checkpoint), "--data-root", str(data_root),
                   "--input-mask", args.input_mask, "--device", args.device,
                   "--protocol", args.protocol, "--output-dir", str(directory)]
        for flag, value in (("--split", args.split), ("--max-samples", args.max_samples)):
            if value is not None:
                command += [flag, str(value)]
        for flag, value in (("--stats-file", args.stats_file), ("--obj-map-dir", args.obj_map_dir),
                            ("--ids-file", args.ids_file), ("--training-ids-file", args.training_ids_file)):
            if value is not None:
                command += [flag, str(relative_to_v4(value, v4))]
        print(f"[Evaluate] {arch}", flush=True)
        subprocess.run(command, cwd=v4, check=True)
        commands.append(command)
    table = []
    for arch in args.architectures:
        summary = json.loads((output / arch / "summary.json").read_text(encoding="utf-8"))
        if summary["architecture"] != arch:
            raise ValueError(f"Saved result is not architecture {arch}.")
        table.append({"architecture": arch, "checkpoint_epoch": summary["checkpoint_epoch"],
                      "num_samples": summary["num_samples"], **summary["metrics"]})
    comparisons = output / "comparisons"
    comparisons.mkdir(exist_ok=True)
    preferred_order = [arch for arch in ("early", "standalone", "later") if arch in args.architectures]
    for a, b in itertools.combinations(preferred_order, 2):
        final_path = comparisons / f"{a}_minus_{b}.json"
        pending = comparisons / f".pending_{uuid.uuid4().hex}.json"
        command = [sys.executable, str(quick / "compare.py"), "--a", str(output / a),
                   "--b", str(output / b), "--repeats", str(args.repeats),
                   "--seed", str(args.seed), "--output", str(pending)]
        print(f"[Compare] {a} minus {b}: negative favors {a}", flush=True)
        try:
            subprocess.run(command, cwd=v4, check=True)
            os.replace(pending, final_path)
        finally:
            pending.unlink(missing_ok=True)
        commands.append(command)
    with (output / "comparison_table.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    manifest = {"arguments": vars(args), "commands_executed": commands,
                "source": "v4_quick_eval; checkpoints and training files are unchanged",
                "reconstruction_support": "model-specific in-bounds mask; always report coverage alongside error",
                "bootstrap_unit": "KITTI sequence/clip ID; not training-seed uncertainty"}
    (output / "launcher_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[Done] {output / 'comparison_table.csv'}", flush=True)
    print(f"Per-scene CSVs: {output}/ARCH/per_scene.csv; bootstrap JSONs: {comparisons}", flush=True)
    return output


if __name__ == "__main__":
    try:
        run(parser().parse_args())
    except (ValueError, FileNotFoundError, FileExistsError, subprocess.CalledProcessError) as error:
        raise SystemExit(str(error)) from error
