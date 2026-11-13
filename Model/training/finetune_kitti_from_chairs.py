#!/usr/bin/env python3
"""
Fine-tune on KITTI from a FlyingChairs pretrained checkpoint.

This script:
1. Builds the normal KITTI multi-frame model using config.json.
2. Loads transferable weights from a FlyingChairs checkpoint:
   - pair_encoder
   - visual_branch
   - motion_branch
   - fusion
   - decoder
3. Skips / reinitializes:
   - uno
   - latent_head

Reason: FlyingChairs pretraining usually uses seq_len=2, so num_pairs=1.
KITTI training usually uses seq_len=5, so num_pairs=4. UNO input channels
and latent_head output channels depend on num_pairs.

Run example:
~/miniconda3/envs/stats402/bin/python finetune_kitti_from_chairs.py \
  --config config.json \
  --pretrained_chairs checkpoints/chairs_pretrain_later_best.pth \
  --architecture_type later \
  --epochs 200 \
  --batch_size 2 \
  --lr 1e-4
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


THIS_DIR = Path(__file__).resolve().parent.parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))


try:
    from data.DataLoader import TempFlowDataset_disp
except ImportError as e:
    raise ImportError(
        "Could not import TempFlowDataset_disp from DataLoader.py. "
        "Make sure this script is in the same folder as DataLoader.py."
    ) from e


def load_trainer_namespace() -> Dict[str, Any]:
    """
    Load trainer.py even if it contains relative imports such as:
        from models.Encoders import ...
    This avoids the direct-script error:
        attempted relative import with no known parent package
    """
    trainer_path = THIS_DIR / "training" / "trainer.py"
    if not trainer_path.exists():
        raise FileNotFoundError(f"trainer.py not found at {trainer_path}")

    src = trainer_path.read_text(encoding="utf-8")
    src = src.replace("from models.Encoders import", "from models.Encoders import")
    src = src.replace("from models.Decoders import", "from models.Decoders import")
    src = src.replace("from .neuralop_seg.uno import", "from neuralop_seg.uno import")

    ns: Dict[str, Any] = {
        "__file__": str(trainer_path),
        "__name__": "trainer_runtime_patched",
    }
    exec(compile(src, str(trainer_path), "exec"), ns)
    return ns


_trainer_ns = load_trainer_namespace()
Trainer = _trainer_ns["Trainer"]
build_modules = _trainer_ns["build_modules"]
save_checkpoint = _trainer_ns["save_checkpoint"]
set_seed = _trainer_ns.get("set_seed")


def fallback_set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if set_seed is None:
    set_seed = fallback_set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune KITTI from FlyingChairs pretrained checkpoint; skip UNO and latent_head."
    )
    parser.add_argument("--config", type=str, default=str(THIS_DIR / "config.json"))
    parser.add_argument("--pretrained_chairs", type=str, required=True)
    parser.add_argument("--architecture_type", type=str,
                        choices=["later", "early", "standalone", "cnn"], default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--save_suffix", type=str, default="from_chairs")
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--strict_transfer", action="store_true")
    parser.add_argument("--transfer_modules", type=str,
                        default="pair_encoder,visual_branch,motion_branch,fusion,decoder")
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def apply_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    if args.architecture_type is not None:
        cfg["model"]["architecture_type"] = args.architecture_type

    arch = cfg["model"].get("architecture_type", "later")
    suffix = args.save_suffix

    cfg["experiment"]["experiment_name"] = f"{arch}_kitti_{suffix}"
    cfg["experiment"]["checkpoint_name"] = f"fullpipeline_{arch}_kitti_{suffix}_latest.pth"
    cfg["experiment"]["best_checkpoint_name"] = f"fullpipeline_{arch}_kitti_{suffix}_best.pth"
    cfg["experiment"]["config_dump_name"] = f"config_{arch}_kitti_{suffix}.json"

    if args.epochs is not None:
        cfg["train"]["num_epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["train"]["lr"] = args.lr
    if args.num_workers is not None:
        cfg["train"]["num_workers"] = args.num_workers
    if args.data_root is not None:
        cfg["data"]["data_root"] = args.data_root

    return cfg


def _filtered_state_dict_load(module: torch.nn.Module,
                              source_state: Dict[str, torch.Tensor],
                              module_name: str,
                              strict: bool = False) -> Tuple[int, int]:
    target_state = module.state_dict()

    if strict:
        module.load_state_dict(source_state, strict=True)
        return len(source_state), 0

    compatible = {}
    skipped = {}

    for k, v in source_state.items():
        if k in target_state and tuple(target_state[k].shape) == tuple(v.shape):
            compatible[k] = v
        else:
            skipped[k] = tuple(v.shape) if hasattr(v, "shape") else "non_tensor"

    module.load_state_dict(compatible, strict=False)

    missing_target = set(target_state.keys()) - set(compatible.keys())
    print(
        f"[transfer] {module_name}: loaded {len(compatible)} tensors, "
        f"skipped {len(skipped)}, missing/unloaded target tensors {len(missing_target)}"
    )

    if len(skipped) > 0:
        print(f"[transfer] {module_name}: skipped preview = {list(skipped.items())[:8]}")

    return len(compatible), len(skipped)


def load_chairs_pretrained_weights(
    modules: Dict[str, torch.nn.Module],
    ckpt_path: str,
    transfer_modules: Iterable[str],
    device: torch.device,
    strict_transfer: bool = False,
) -> None:
    ckpt_path = str(Path(ckpt_path).expanduser().resolve())
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Pretrained checkpoint not found: {ckpt_path}")

    print(f"[transfer] Loading FlyingChairs checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    print("[transfer] Will transfer:", list(transfer_modules))
    print("[transfer] Will reinitialize / skip: uno, latent_head")

    for name in transfer_modules:
        name = name.strip()
        if not name:
            continue

        if name not in modules:
            print(f"[transfer] WARNING: module '{name}' not in current model, skipping.")
            continue

        key = f"{name}_state_dict"
        if key not in ckpt:
            print(f"[transfer] WARNING: checkpoint has no key '{key}', skipping {name}.")
            continue

        _filtered_state_dict_load(
            modules[name],
            ckpt[key],
            module_name=name,
            strict=strict_transfer,
        )

    for skipped_name in ["uno", "latent_head"]:
        if skipped_name in modules:
            print(f"[transfer] {skipped_name}: NOT loaded; using fresh KITTI-size initialization.")


def main():
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args)

    set_seed(cfg["train"]["seed"])

    save_dir = (THIS_DIR / cfg["experiment"]["save_dir"]).resolve()
    tb_dir = (THIS_DIR / cfg["experiment"]["tensorboard_dir"] / cfg["experiment"]["experiment_name"]).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)

    config_dump_path = save_dir / cfg["experiment"]["config_dump_name"]
    with open(config_dump_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    writer = SummaryWriter(log_dir=str(tb_dir))

    print("[data] KITTI root:", cfg["data"]["data_root"])
    print("[data] seq_len:", cfg["data"]["seq_len"])
    print("[data] crop_size:", cfg["data"]["crop_size"])
    print("[model] architecture_type:", cfg["model"].get("architecture_type", "later"))

    dataset = TempFlowDataset_disp(
        root=cfg["data"]["data_root"],
        split=cfg["data"]["split"],
        image_folder=cfg["data"]["image_folder"],
        flow_type=cfg["data"]["flow_type"],
        disp_type=cfg["data"]["disp_type"],
        seq_len=cfg["data"]["seq_len"],
        center_frame_idx=cfg["data"]["center_frame_idx"],
        crop_size=tuple(cfg["data"]["crop_size"]),
        normalize=cfg["data"]["normalize"],
        stats_in=cfg["data"]["stats_file"],
        return_pair_only=cfg["data"]["return_pair_only"],
    )

    train_loader = DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=cfg["train"]["shuffle"],
        num_workers=cfg["train"]["num_workers"],
        pin_memory=cfg["train"]["pin_memory"],
        drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[system] device:", device)

    modules = build_modules(cfg, device)

    transfer_modules = [x.strip() for x in args.transfer_modules.split(",") if x.strip()]
    load_chairs_pretrained_weights(
        modules=modules,
        ckpt_path=args.pretrained_chairs,
        transfer_modules=transfer_modules,
        device=device,
        strict_transfer=args.strict_transfer,
    )

    optimizer = torch.optim.AdamW(
        [p for module in modules.values() for p in module.parameters()],
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    trainer = Trainer(cfg, modules, optimizer, writer, device)

    sanity_batch = next(iter(train_loader))
    with torch.no_grad():
        sanity_out = trainer.forward_pipeline(
            sanity_batch["imgs"].to(device),
            sanity_batch["valid"].to(device),
        )

    print("[sanity] imgs:", tuple(sanity_batch["imgs"].shape))
    print("[sanity] pred flows:", tuple(sanity_out["flows"].shape))
    print("[sanity] flow_inits:", None if sanity_out["flow_inits"] is None else tuple(sanity_out["flow_inits"].shape))
    print("[sanity] pair_feats:", tuple(sanity_out["pair_feats"].shape))
    print("[sanity] fused_seq:", tuple(sanity_out["fused_seq"].shape))
    print("[sanity] refined_seq:", tuple(sanity_out["refined_seq"].shape))
    print("[sanity] flow_residuals:", tuple(sanity_out["flow_residuals"].shape))

    best_loss = float("inf")
    milestone_epochs = set(cfg["train"].get("save_epoch_checkpoints", []))

    for epoch in range(cfg["train"]["num_epochs"]):
        running: Dict[str, float] = {}
        n_batches = 0

        for batch in train_loader:
            stats = trainer.train_step(batch)
            for key, value in stats.items():
                running[key] = running.get(key, 0.0) + float(value)
            n_batches += 1

        avg = {k: v / max(n_batches, 1) for k, v in running.items()}

        for key, value in avg.items():
            writer.add_scalar(f"train_epoch/{key}", value, epoch + 1)

        print(
            f"Epoch {epoch + 1}/{cfg['train']['num_epochs']} | "
            f"loss={avg.get('loss', 0.0):.4f} | "
            f"flow={avg.get('loss_flow', 0.0):.4f} | "
            f"self={avg.get('loss_self', 0.0):.4f} | "
            f"temp={avg.get('loss_temp', 0.0):.4f} | "
            f"acc_photo={avg.get('loss_acc_photo', 0.0):.4f} | "
            f"fb_loss={avg.get('loss_fb', 0.0):.4f} | "
            f"smooth={avg.get('loss_smooth', 0.0):.4f} | "
            f"latent={avg.get('latent_delta_mean_abs', 0.0):.4f} | "
            f"flow_res={avg.get('flow_residual_mean_abs', 0.0):.4f}"
        )

        latest_path = save_dir / cfg["experiment"]["checkpoint_name"]
        save_checkpoint(latest_path, epoch + 1, modules, optimizer, stats=avg, config=cfg)

        if avg.get("loss", float("inf")) < best_loss:
            best_loss = avg["loss"]
            best_path = save_dir / cfg["experiment"]["best_checkpoint_name"]
            save_checkpoint(best_path, epoch + 1, modules, optimizer, stats=avg, config=cfg)

        if (epoch + 1) in milestone_epochs:
            milestone_path = save_dir / f"{cfg['experiment']['experiment_name']}_epoch_{epoch + 1}.pth"
            save_checkpoint(milestone_path, epoch + 1, modules, optimizer, stats=avg, config=cfg)

    writer.close()
    print("[done] KITTI fine-tuning finished.")
    print("[done] Latest checkpoint:", save_dir / cfg["experiment"]["checkpoint_name"])
    print("[done] Best checkpoint:", save_dir / cfg["experiment"]["best_checkpoint_name"])


if __name__ == "__main__":
    main()
