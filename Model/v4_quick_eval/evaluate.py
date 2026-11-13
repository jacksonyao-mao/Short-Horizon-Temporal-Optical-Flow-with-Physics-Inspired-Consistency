#!/usr/bin/env python3
"""Evaluate original V4 checkpoints without editing V4 or performing training."""
import argparse
import csv
import hashlib
import importlib
import inspect
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch

from metrics import flow_statistics, summarize, two_step_statistics


def read_ids(path):
    if path is None:
        return None
    ids = [s.split("#", 1)[0].strip() for s in Path(path).read_text().splitlines()]
    ids = [s for s in ids if s]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("ID files must be nonempty and contain no duplicate IDs.")
    return set(ids)


def select_samples(samples, ids, training_ids, protocol):
    available = {Path(s["flow_path"]).stem for s in samples}
    scenes = {s["seq_id"] for s in samples}
    if ids is not None and ids - available - scenes:
        raise ValueError(f"Unknown evaluation IDs: {sorted(ids - available - scenes)}")
    selected = [s for s in samples if ids is None or s["seq_id"] in ids
                or Path(s["flow_path"]).stem in ids]
    if not selected:
        raise ValueError("No selected samples.")
    if protocol == "held-out":
        if ids is None or training_ids is None:
            raise ValueError("held-out requires --ids-file and --training-ids-file.")
        training_scenes = {s.split("_")[0] for s in training_ids}
        overlap = {s["seq_id"] for s in selected} & training_scenes
        if overlap:
            raise ValueError(f"Evaluation/training scene overlap: {sorted(overlap)}")
    return selected


def load_runtime(v4_dir):
    """Support the V4 distribution's absolute and package-relative imports."""
    v4_dir = Path(v4_dir).resolve()
    if not (v4_dir / "training" / "trainer.py").is_file():
        raise FileNotFoundError(f"V4 trainer.py missing in {v4_dir}")
    sys.path.insert(0, str(v4_dir))
    package = types.ModuleType("_v4_eval_runtime")
    package.__path__ = [str(v4_dir)]
    sys.modules[package.__name__] = package
    trainer = importlib.import_module(package.__name__ + ".training.trainer")
    data = importlib.import_module(package.__name__ + ".data.DataLoader")
    return trainer, data


def sha256_files(paths):
    digest = hashlib.sha256()
    for path in paths:
        # Hash file boundaries as well as contents; paths may differ across machines.
        digest.update(str(Path(path).stat().st_size).encode() + b"\0")
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def recover_rgb(imgs, data_cfg, stats):
    if data_cfg["normalize"]:
        mean = imgs.new_tensor(stats["mean"]).view(1, 3, 1, 1)
        std = imgs.new_tensor(stats["std"]).view(1, 3, 1, 1)
        imgs = imgs * std + mean
    if imgs.min() < -1e-5 or imgs.max() > 1 + 1e-5:
        raise ValueError("Normalization statistics do not reconstruct RGB in [0,1].")
    return imgs.clamp(0, 1)


def run(args):
    v4 = Path(args.v4_dir).resolve()
    output = Path(args.output_dir).resolve()
    if (output / "summary.json").exists() or (output / "per_scene.csv").exists():
        raise FileExistsError("Output already contains evaluation files; choose a new directory.")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. Use --device cpu for a smoke test.")
    checkpoint_path = Path(args.checkpoint).resolve()
    # Original V4 checkpoints contain tensors and ordinary config/stats dictionaries.
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    cfg = (json.loads(Path(args.config).read_text()) if args.config else ckpt.get("config"))
    if not cfg:
        raise ValueError("Checkpoint has no configuration; supply --config.")
    if ckpt.get("config"):
        saved = ckpt["config"]
        if cfg["model"] != saved["model"]:
            raise ValueError("Supplied model configuration differs from the checkpoint.")
        for key in ("seq_len", "return_pair_only", "normalize", "crop_size"):
            if cfg["data"][key] != saved["data"][key]:
                raise ValueError(f"Supplied preprocessing differs from checkpoint: {key}")
    d = cfg["data"]
    if d["seq_len"] < 3 or d["return_pair_only"]:
        raise ValueError("Use a multi-frame KITTI checkpoint, not a two-frame Chairs checkpoint.")
    split = args.split or d["split"]
    stats_path = Path(args.stats_file or d["stats_file"])
    if not stats_path.is_absolute():
        stats_path = v4 / stats_path
    # Prevent V4's loader from silently estimating normalization on evaluation images.
    if not stats_path.is_file():
        raise FileNotFoundError(f"Training normalization statistics required: {stats_path}")
    stats = json.loads(stats_path.read_text())
    if len(stats["mean"]) != 3 or len(stats["std"]) != 3 or min(stats["std"]) <= 0:
        raise ValueError("Expected three RGB means and three positive standard deviations.")
    runtime, data_module = load_runtime(v4)
    dataset = data_module.TempFlowDataset_disp(
        root=str(Path(args.data_root).resolve()), split=split,
        image_folder=d["image_folder"], flow_type=d["flow_type"], disp_type=d["disp_type"],
        seq_len=d["seq_len"], center_frame_idx=d["center_frame_idx"],
        crop_size=tuple(d["crop_size"]) if d["crop_size"] else None,
        normalize=d["normalize"], stats_in=str(stats_path), return_pair_only=False)
    dataset.samples = select_samples(dataset.samples, read_ids(args.ids_file),
                                     read_ids(args.training_ids_file), args.protocol)
    if args.max_samples is not None:
        if args.max_samples < 1:
            raise ValueError("max-samples must be positive.")
        dataset.samples = dataset.samples[:args.max_samples]
    obj_dir = (Path(args.obj_map_dir) if args.obj_map_dir else
               Path(args.data_root) / "Flow" / split / "obj_map")
    for sample in dataset.samples:
        path = obj_dir / Path(sample["flow_path"]).name
        if not path.is_file():
            raise FileNotFoundError(f"Object map required for foreground/background evaluation: {path}")
    modules = runtime.build_modules(cfg, device)
    for name, module in modules.items():
        key = f"{name}_state_dict"
        if key not in ckpt:
            raise KeyError(f"Checkpoint missing {key}")
        module.load_state_dict(ckpt[key], strict=True)
        module.eval()
    epoch = ckpt.get("epoch")
    del ckpt
    trainer = runtime.Trainer(cfg, modules, optimizer=None, writer=None, device=device)
    loss_cfg = cfg["loss"]
    photo_cfg = {"lambda_l1": loss_cfg.get("photo_lambda_l1", 0.15),
                 "lambda_ssim": loss_cfg.get("photo_lambda_ssim", 0.85),
                 "lambda_census": loss_cfg.get("photo_lambda_census", 0.30),
                 "census_patch": loss_cfg.get("photo_census_patch", 7)}
    if photo_cfg["census_patch"] < 1 or photo_cfg["census_patch"] % 2 != 1:
        raise ValueError("Census patch must be a positive odd integer.")
    radius = max(1, photo_cfg["census_patch"] // 2)
    def photo_map(src, tgt, flow):
        return runtime.pair_photometric_map(src, tgt, flow, **photo_cfg)[0]
    rows = []
    import cv2
    with torch.inference_mode():
        for i, sample in enumerate(dataset.samples):
            batch = dataset[i]
            imgs = batch["imgs"].to(device)
            valid = batch["valid"].to(device)
            input_mask = valid if args.input_mask == "annotation" else torch.ones_like(valid)
            out = trainer.forward_pipeline(imgs[None], input_mask[None])
            flows = out["flows"][0]
            pair_index = int(batch["src_idx_in_seq"].item())
            obj_path = obj_dir / Path(sample["flow_path"]).name
            obj = cv2.imread(str(obj_path), cv2.IMREAD_UNCHANGED)
            if obj is None or obj.ndim != 2:
                raise ValueError(f"Expected a single-channel KITTI object map: {obj_path}")
            if d["crop_size"]:
                obj = data_module.center_crop(obj, tuple(d["crop_size"]))
            obj = torch.from_numpy(np.asarray(obj).astype(np.int64)).to(device)
            row = {"sample_id": Path(sample["flow_path"]).stem, "scene_id": sample["seq_id"],
                   "src_idx_in_seq": pair_index,
                   "sample_signature": sha256_files([*sample["img_paths"], sample["flow_path"], obj_path])}
            row.update(flow_statistics(flows[pair_index], batch["flow"].to(device), valid, obj))
            row.update(two_step_statistics(recover_rgb(imgs, d, stats), flows, photo_map, radius))
            rows.append(row)
            print(f"[{i+1}/{len(dataset)}] {row['sample_id']} EPE={row['epe_all']} "
                  f"two-step L1={row['two_step_l1']}", flush=True)
    protocol = {
        "schema_version": 1, "split": split, "protocol": args.protocol,
        "metric_source_sha256": sha256_files([Path(__file__).with_name("metrics.py")]),
        "v4_photo_source_sha256": hashlib.sha256("\n".join(inspect.getsource(fn) for fn in
            [runtime.pair_photometric_map, runtime.ssim_map_loss, runtime.soft_census_transform,
             runtime.warp_image, runtime.make_sampling_grid, runtime.warp_valid_mask]).encode()).hexdigest(),
        "partial_run": args.max_samples is not None,
        "input_mask": args.input_mask if cfg["model"].get("uno_use_valid_mask") else "unused",
        "crop_size": d["crop_size"], "seq_len": d["seq_len"],
        "image_folder": d["image_folder"], "flow_type": d["flow_type"],
        "normalization": {"enabled": d["normalize"], "mean": stats["mean"], "std": stats["std"]},
        "photometric": photo_cfg, "photometric_rgb_range": [0, 1],
        "photometric_mask": "intermediate-and-final in-bounds, neighborhood erosion",
        "mask_radius": radius, "photometric_weighting": "uniform; no confidence/occlusion weighting",
        "foreground": "KITTI obj_map > 0, intersected with valid flow",
        "outlier": "EPE > 3 pixels AND EPE/(GT magnitude + 1e-9) > 0.05",
    }
    summary = {"evaluation_protocol": protocol, "num_samples": len(rows),
               "architecture": cfg["model"]["architecture_type"], "checkpoint_epoch": epoch,
               "checkpoint": str(checkpoint_path), "v4_dir": str(v4), "data_root": str(Path(args.data_root).resolve()),
               "checkpoint_sha256": sha256_files([checkpoint_path]), "config": cfg,
               "device": str(device), "torch_version": str(torch.__version__),
               "trainable_and_frozen_parameters": sum(p.numel() for m in modules.values() for p in m.parameters()),
               "training_ids_file": args.training_ids_file, "evaluation_ids_file": args.ids_file,
               "declared_training_ids": sorted(read_ids(args.training_ids_file) or []),
               "metrics": summarize(rows),
               "notes": ["Held-out status depends on the accuracy of the supplied training-ID list.",
                         "An all-ones mask is an inference intervention, not a retrained RGB-only baseline.",
                         "Two-step scores use RGB [0,1] and uniform weights; they are not raw V4 training losses.",
                         "Reconstruction support depends on predictions: compare coverage as well as errors.",
                         "No occlusion removal: reconstruction can be affected by occlusion/illumination."]}
    if summary["metrics"]["epe_all"] is None:
        raise ValueError("No valid ground-truth pixels in selected evaluation data.")
    if summary["metrics"]["two_step_l1"] is None:
        raise ValueError("No valid two-step reconstruction pixels; inspect predicted flows.")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "per_scene.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(summary["metrics"], indent=2, allow_nan=False))
    print(f"Saved to {output}")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--v4-dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", help="Optional; otherwise use the checkpoint's saved config.")
    p.add_argument("--data-root", required=True, help="V4 data root containing Flow/ and Additional_frames/.")
    p.add_argument("--stats-file", help="Training RGB statistics; relative paths resolve against V4 directory.")
    p.add_argument("--obj-map-dir", help="Override Flow/<split>/obj_map.")
    p.add_argument("--split", help="Disk split; defaults to checkpoint config.")
    p.add_argument("--protocol", choices=["training-diagnostic", "held-out"], default="training-diagnostic")
    p.add_argument("--ids-file", help="One six-digit scene ID or flow stem per line; # comments allowed.")
    p.add_argument("--training-ids-file", help="Actual training scene IDs; required for held-out evaluation.")
    p.add_argument("--input-mask", choices=["annotation", "ones"], default="annotation")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-samples", type=int, help="Debug limit; marked as partial in the output.")
    p.add_argument("--output-dir", required=True)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
