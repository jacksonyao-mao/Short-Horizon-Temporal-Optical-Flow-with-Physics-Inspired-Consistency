"""
pre_trainer.py

FlyingChairs pretraining launcher for your current optical-flow architecture.

Main design:
    - Uses FlyingChairs pair input: [I1, I2]
    - Keeps your model pipeline and still applies UNO if architecture_type is early/later/standalone
    - Uses simple supervised pretraining loss:
          EPE or weighted EPE + lambda_photo * photometric + lambda_smooth * edge-aware smoothness
    - Does NOT use temporal, accumulated, or forward-backward losses in this pretraining stage.

Example:
    python pre_trainer.py \
        --chairs_root /path/to/FlyingChairs_release \
        --config config.json \
        --architecture_type later \
        --epochs 50 \
        --batch_size 4 \
        --crop_h 352 --crop_w 512
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Robust imports: works both inside a package and when copied beside your project files.
try:
    from data.dataloader_FlowChair import FlyingChairsDataset
    from training.trainer import (
        build_modules,
        save_checkpoint,
        set_seed,
        weighted_epe_loss,
        multiframe_photometric_loss,
        multiframe_edge_aware_smoothness_loss,
        select_gt_flow,
    )
except ImportError:
    THIS_DIR = Path(__file__).resolve().parent.parent
    if str(THIS_DIR) not in sys.path:
        sys.path.insert(0, str(THIS_DIR))
    from data.dataloader_FlowChair import FlyingChairsDataset
    from training.trainer import (
        build_modules,
        save_checkpoint,
        set_seed,
        weighted_epe_loss,
        multiframe_photometric_loss,
        multiframe_edge_aware_smoothness_loss,
        select_gt_flow,
    )


def standard_epe_loss(pred: torch.Tensor, gt: torch.Tensor, valid: Optional[torch.Tensor] = None) -> torch.Tensor:
    epe = torch.norm(pred - gt, dim=1, keepdim=True)
    if valid is not None:
        if valid.ndim == 3:
            valid = valid.unsqueeze(1)
        valid = valid.float()
        return (epe * valid).sum() / (valid.sum() + 1e-6)
    return epe.mean()


class FlyingChairsPreTrainer:
    def __init__(self, cfg: Dict[str, Any], modules: Dict[str, torch.nn.Module], optimizer, writer, device):
        self.cfg = cfg
        self.modules = modules
        self.optimizer = optimizer
        self.writer = writer
        self.device = device
        self.global_step = 0

    def train_mode(self):
        for m in self.modules.values():
            m.train()

    def forward_pipeline(self, imgs: torch.Tensor, valid: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        This intentionally mirrors Trainer.forward_pipeline.

        Where UNO is applied:
            early:      pair_feats -> UNO -> refined_pair_feats -> fusion -> decoder
            later:      pair_feats -> fusion -> UNO -> fused_seq + latent_delta -> decoder
            standalone: pair_feats -> fusion -> UNO -> latent_delta only -> decoder

        For FlyingChairs, imgs has T=2, so there is only one adjacent pair.
        """
        arch = self.cfg["model"].get("architecture_type", "later")

        if arch == "cnn":
            from training.trainer import Trainer
            return Trainer.forward_pipeline(self, imgs, valid)

        pair_out = self.modules["pair_encoder"](imgs)
        pair_feats = pair_out["pair_feats"]       # [B,1,C,h,w]
        flow_inits = pair_out["flow_inits"]       # [B,1,2,h,w]
        corrs = pair_out["corrs"]

        if flow_inits is None:
            raise RuntimeError("predict_flow_init=True is required for this architecture.")

        # Import here to avoid needing extra top-level imports in some project layouts.
        try:
            from models.Encoders import build_uno_input_2d, downsample_valid_mask
        except ImportError:
            from models.Encoders import build_uno_input_2d, downsample_valid_mask

        if arch == "early":
            valid_ds = None
            if self.cfg["model"].get("uno_use_valid_mask", True):
                valid_ds = downsample_valid_mask(valid, pair_feats.shape[-2:])

            # UNO APPLIED HERE for early integration.
            uno_in = build_uno_input_2d(pair_feats, flow_inits, valid_mask=valid_ds)
            uno_feat = self.modules["uno"](uno_in)
            b, tm, _, h, w = pair_feats.shape
            latent_delta = self.modules["latent_head"](uno_feat, b, tm, h, w)
            refined_pair_feats = pair_feats + latent_delta

            visual_feats = self.modules["visual_branch"](imgs)
            motion_feats = self.modules["motion_branch"](refined_pair_feats)
            fused_seq = self.modules["fusion"](visual_feats, motion_feats)
            refined_seq = fused_seq

        elif arch == "later":
            visual_feats = self.modules["visual_branch"](imgs)
            motion_feats = self.modules["motion_branch"](pair_feats)
            fused_seq = self.modules["fusion"](visual_feats, motion_feats)

            valid_ds = None
            if self.cfg["model"].get("uno_use_valid_mask", True):
                valid_ds = downsample_valid_mask(valid, fused_seq.shape[-2:])

            # UNO APPLIED HERE for late integration.
            uno_in = build_uno_input_2d(fused_seq, flow_inits, valid_mask=valid_ds)
            uno_feat = self.modules["uno"](uno_in)
            b, tm, _, h, w = fused_seq.shape
            latent_delta = self.modules["latent_head"](uno_feat, b, tm, h, w)
            refined_seq = fused_seq + latent_delta

        elif arch == "standalone":
            visual_feats = self.modules["visual_branch"](imgs)
            motion_feats = self.modules["motion_branch"](pair_feats)
            fused_seq = self.modules["fusion"](visual_feats, motion_feats)

            valid_ds = None
            if self.cfg["model"].get("uno_use_valid_mask", True):
                valid_ds = downsample_valid_mask(valid, fused_seq.shape[-2:])

            # UNO APPLIED HERE for standalone integration.
            uno_in = build_uno_input_2d(fused_seq, flow_inits, valid_mask=valid_ds)
            uno_feat = self.modules["uno"](uno_in)
            b, tm, _, h, w = fused_seq.shape
            latent_delta = self.modules["latent_head"](uno_feat, b, tm, h, w)
            refined_seq = latent_delta

        else:
            raise ValueError(f"Unsupported architecture_type={arch}")

        flows, flow_residuals = self.modules["decoder"](refined_seq, flow_inits=flow_inits)
        return {
            "flows": flows,
            "flow_inits": flow_inits,
            "pair_feats": pair_feats,
            "corrs": corrs,
            "fused_seq": fused_seq,
            "latent_delta": latent_delta,
            "refined_seq": refined_seq,
            "flow_residuals": flow_residuals,
        }

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.train_mode()

        imgs = batch["imgs"].to(self.device)      # [B,2,3,H,W]
        gt_flow = batch["flow"].to(self.device)   # [B,2,H,W]
        valid = batch["valid"].to(self.device)    # [B,H,W]
        img_src = batch["img_src"].to(self.device)
        src_idx = batch["src_idx_in_seq"].to(self.device)

        out = self.forward_pipeline(imgs, valid)
        pred_flows = out["flows"]                 # [B,1,2,H,W]
        pred = select_gt_flow(pred_flows, src_idx) # [B,2,H,W]

        loss_cfg = self.cfg["loss"]
        if loss_cfg.get("chairs_use_weighted_epe", False):
            loss_flow = weighted_epe_loss(
                pred, gt_flow, img_src, valid,
                edge_weight_scale=loss_cfg.get("lambda_edge_weight", 1.0),
            )
        else:
            loss_flow = standard_epe_loss(pred, gt_flow, valid)

        if loss_cfg.get("lambda_self", 0.0) > 0:
            loss_photo = multiframe_photometric_loss(
                imgs,
                pred_flows,
                lambda_l1=loss_cfg.get("photo_lambda_l1", 0.15),
                lambda_ssim=loss_cfg.get("photo_lambda_ssim", 0.85),
                lambda_census=loss_cfg.get("photo_lambda_census", 0.30),
                census_patch=loss_cfg.get("photo_census_patch", 7),
                reduction="mean",
                use_confidence=loss_cfg.get("photo_use_confidence", False),
                texture_floor=loss_cfg.get("photo_texture_floor", 0.05),
                robust_beta=loss_cfg.get("photo_robust_beta", 10.0),
                fb_conf=None,
            )
        else:
            loss_photo = pred.new_tensor(0.0)

        if loss_cfg.get("lambda_smooth", 0.0) > 0:
            loss_smooth = multiframe_edge_aware_smoothness_loss(pred_flows, imgs)
        else:
            loss_smooth = pred.new_tensor(0.0)

        loss = (
            loss_flow
            + loss_cfg.get("lambda_self", 0.05) * loss_photo
            + loss_cfg.get("lambda_smooth", 0.01) * loss_smooth
        )

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for m in self.modules.values() for p in m.parameters()], max_norm=1.0)
        self.optimizer.step()

        stats = {
            "loss": float(loss.item()),
            "loss_flow": float(loss_flow.item()),
            "loss_photo": float(loss_photo.item()),
            "loss_smooth": float(loss_smooth.item()),
            "latent_delta_mean_abs": float(out["latent_delta"].abs().mean().item()),
            "flow_residual_mean_abs": float(out["flow_residuals"].abs().mean().item()),
        }

        if self.global_step % self.cfg["train"].get("log_every_n_steps", 20) == 0:
            for k, v in stats.items():
                self.writer.add_scalar(f"chairs_train_step/{k}", v, self.global_step)
        self.global_step += 1
        return stats


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def adapt_config_for_chairs(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    cfg = json.loads(json.dumps(cfg))  # deep copy through json-compatible config

    cfg.setdefault("data", {})
    cfg["data"]["seq_len"] = 2
    cfg["data"]["return_pair_only"] = True
    cfg["data"]["crop_size"] = [args.crop_h, args.crop_w]

    if args.architecture_type is not None:
        cfg["model"]["architecture_type"] = args.architecture_type

    # Stage name and outputs.
    cfg.setdefault("experiment", {})
    arch = cfg["model"].get("architecture_type", "later")
    cfg["experiment"]["experiment_name"] = args.experiment_name or f"chairs_pretrain_{arch}"
    cfg["experiment"]["save_dir"] = args.save_dir
    cfg["experiment"]["tensorboard_dir"] = args.tensorboard_dir
    cfg["experiment"]["checkpoint_name"] = f"chairs_pretrain_{arch}_latest.pth"
    cfg["experiment"]["best_checkpoint_name"] = f"chairs_pretrain_{arch}_best.pth"
    cfg["experiment"]["config_dump_name"] = f"config_chairs_pretrain_{arch}.json"

    cfg.setdefault("train", {})
    cfg["train"]["num_epochs"] = args.epochs
    cfg["train"]["batch_size"] = args.batch_size
    cfg["train"]["lr"] = args.lr
    cfg["train"].setdefault("weight_decay", args.weight_decay)
    cfg["train"]["weight_decay"] = args.weight_decay
    cfg["train"]["num_workers"] = args.num_workers
    cfg["train"]["pin_memory"] = True
    cfg["train"].setdefault("seed", 1234)
    cfg["train"].setdefault("log_every_n_steps", 20)
    cfg["train"]["save_epoch_checkpoints"] = args.save_epochs

    # Clean FlyingChairs pretraining loss. No temporal/FB/accumulated losses here.
    cfg.setdefault("loss", {})
    cfg["loss"]["chairs_use_weighted_epe"] = args.weighted_epe
    cfg["loss"]["lambda_edge_weight"] = args.edge_weight_scale
    cfg["loss"]["lambda_self"] = args.lambda_photo
    cfg["loss"]["lambda_smooth"] = args.lambda_smooth
    cfg["loss"]["lambda_temp"] = 0.0
    cfg["loss"]["lambda_acc_photo"] = 0.0
    cfg["loss"]["lambda_acc_smooth"] = 0.0
    cfg["loss"]["lambda_fb"] = 0.0
    cfg["loss"]["photo_use_fb_consistency"] = False
    cfg["loss"]["photo_bidirectional"] = False
    cfg["loss"].setdefault("photo_lambda_l1", 0.15)
    cfg["loss"].setdefault("photo_lambda_ssim", 0.85)
    cfg["loss"].setdefault("photo_lambda_census", 0.30)
    cfg["loss"].setdefault("photo_census_patch", 7)
    # Safer for supervised Chairs pretraining. Turn on manually only if wanted.
    cfg["loss"]["photo_use_confidence"] = args.photo_use_confidence

    return cfg


def parse_args():
    p = argparse.ArgumentParser(description="FlyingChairs pretraining for your optical flow model")
    p.add_argument("--config", type=str, required=True, help="Existing KITTI config.json to reuse model settings")
    p.add_argument("--chairs_root", type=str, required=True, help="Path to FlyingChairs_release or its data folder")
    p.add_argument("--chairs_split_file", type=str, default=None, help="Optional chairs_split.txt")
    p.add_argument("--architecture_type", type=str, choices=["later", "early", "standalone", "cnn"], default=None)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--crop_h", type=int, default=352)
    p.add_argument("--crop_w", type=int, default=512)
    p.add_argument("--stats_file", type=str, default="chairs_stats.json")
    p.add_argument("--save_dir", type=str, default="checkpoints")
    p.add_argument("--tensorboard_dir", type=str, default="runs")
    p.add_argument("--experiment_name", type=str, default=None)
    p.add_argument("--lambda_photo", type=float, default=0.05, help="Set 0.0 to disable photometric loss")
    p.add_argument("--lambda_smooth", type=float, default=0.01)
    p.add_argument("--weighted_epe", action="store_true", help="Use your weighted EPE instead of standard EPE")
    p.add_argument("--edge_weight_scale", type=float, default=1.0)
    p.add_argument("--photo_use_confidence", action="store_true")
    p.add_argument("--save_epochs", type=int, nargs="*", default=[])
    return p.parse_args()


def main():
    args = parse_args()
    cfg = adapt_config_for_chairs(load_config(args.config), args)
    set_seed(cfg["train"].get("seed", 1234))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    print("architecture_type:", cfg["model"].get("architecture_type", "later"))
    print("FlyingChairs seq_len:", cfg["data"]["seq_len"])
    print("loss: EPE +", cfg["loss"]["lambda_self"], "* photo +", cfg["loss"]["lambda_smooth"], "* smooth")

    script_dir = Path(__file__).resolve().parent.parent
    save_dir = (script_dir / cfg["experiment"]["save_dir"]).resolve()
    tb_dir = (script_dir / cfg["experiment"]["tensorboard_dir"] / cfg["experiment"]["experiment_name"]).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)

    with (save_dir / cfg["experiment"]["config_dump_name"]).open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    train_set = FlyingChairsDataset(
        root=args.chairs_root,
        split="train",
        split_file=args.chairs_split_file,
        crop_size=(args.crop_h, args.crop_w),
        resize_size=None,
        normalize=True,
        stats_in=args.stats_file,
        augment=True,
        photometric_safe_aug=True,
    )
    loader = DataLoader(
        train_set,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=cfg["train"].get("pin_memory", True),
        drop_last=True,
    )

    modules = build_modules(cfg, device)
    optimizer = torch.optim.AdamW(
        [p for m in modules.values() for p in m.parameters() if p.requires_grad],
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"].get("weight_decay", 1e-4),
    )
    writer = SummaryWriter(log_dir=str(tb_dir))
    pretrainer = FlyingChairsPreTrainer(cfg, modules, optimizer, writer, device)

    # Shape sanity check, including UNO position output.
    sanity = next(iter(loader))
    with torch.no_grad():
        out = pretrainer.forward_pipeline(sanity["imgs"].to(device), sanity["valid"].to(device))
    print("imgs:", tuple(sanity["imgs"].shape))
    print("pred flows:", tuple(out["flows"].shape))
    print("flow_inits:", tuple(out["flow_inits"].shape))
    print("pair_feats:", tuple(out["pair_feats"].shape))
    print("fused_seq:", tuple(out["fused_seq"].shape))
    print("latent_delta (zero for cnn):", tuple(out["latent_delta"].shape))
    print("refined_seq:", tuple(out["refined_seq"].shape))

    best_loss = float("inf")
    for epoch in range(1, cfg["train"]["num_epochs"] + 1):
        running: Dict[str, float] = {}
        pbar = tqdm(loader, desc=f"FlyingChairs epoch {epoch}")
        n = 0
        for batch in pbar:
            stats = pretrainer.train_step(batch)
            for k, v in stats.items():
                running[k] = running.get(k, 0.0) + v
            n += 1
            pbar.set_postfix({"loss": stats["loss"], "epe": stats["loss_flow"], "photo": stats["loss_photo"]})

        avg = {k: v / max(n, 1) for k, v in running.items()}
        for k, v in avg.items():
            writer.add_scalar(f"chairs_train_epoch/{k}", v, epoch)

        print(
            f"Epoch {epoch}/{cfg['train']['num_epochs']} | "
            f"loss={avg.get('loss', 0):.4f} | "
            f"flow={avg.get('loss_flow', 0):.4f} | "
            f"photo={avg.get('loss_photo', 0):.4f} | "
            f"smooth={avg.get('loss_smooth', 0):.4f} | "
            f"latent={avg.get('latent_delta_mean_abs', 0):.4f}"
        )

        latest_path = save_dir / cfg["experiment"]["checkpoint_name"]
        save_checkpoint(latest_path, epoch, modules, optimizer, stats=avg, config=cfg)

        if avg.get("loss", float("inf")) < best_loss:
            best_loss = avg["loss"]
            best_path = save_dir / cfg["experiment"]["best_checkpoint_name"]
            save_checkpoint(best_path, epoch, modules, optimizer, stats=avg, config=cfg)

        if epoch in set(cfg["train"].get("save_epoch_checkpoints", [])):
            milestone_path = save_dir / f"{cfg['experiment']['experiment_name']}_epoch_{epoch}.pth"
            save_checkpoint(milestone_path, epoch, modules, optimizer, stats=avg, config=cfg)

    writer.close()
    print("FlyingChairs pretraining finished.")


if __name__ == "__main__":
    main()
