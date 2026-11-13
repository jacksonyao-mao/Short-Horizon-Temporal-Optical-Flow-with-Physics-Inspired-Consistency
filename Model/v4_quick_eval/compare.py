#!/usr/bin/env python3
"""Compare two evaluator output directories using paired scene bootstrap."""
import argparse
import csv
import json
from pathlib import Path

from metrics import paired_bootstrap


def load_run(directory):
    directory = Path(directory)
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    with (directory / "per_scene.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != summary["num_samples"]:
        raise ValueError("CSV sample count differs from summary metadata.")
    return summary, rows


def compare_runs(a_dir, b_dir, repeats=5000, seed=42):
    a, rows_a = load_run(a_dir)
    b, rows_b = load_run(b_dir)
    if a["evaluation_protocol"] != b["evaluation_protocol"]:
        raise ValueError("Evaluation protocols differ. Re-evaluate with matching settings.")
    signatures_a = {r["sample_id"]: r["sample_signature"] for r in rows_a}
    signatures_b = {r["sample_id"]: r["sample_signature"] for r in rows_b}
    if signatures_a != signatures_b:
        raise ValueError("Samples, source images, or ground-truth/object maps differ.")
    result = paired_bootstrap(rows_a, rows_b, repeats, seed)
    result.update({"a": str(Path(a_dir).resolve()), "b": str(Path(b_dir).resolve()),
                   "evaluation_protocol": a["evaluation_protocol"],
                   "coverage_a_percent": a["metrics"]["two_step_coverage_percent"],
                   "coverage_b_percent": b["metrics"]["two_step_coverage_percent"]})
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--a", required=True, help="First evaluation output directory.")
    p.add_argument("--b", required=True, help="Second evaluation output directory.")
    p.add_argument("--repeats", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    result = compare_runs(args.a, args.b, args.repeats, args.seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result["metrics"], indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
