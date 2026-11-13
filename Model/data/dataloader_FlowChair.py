"""
dataloader_FlowChair.py

FlyingChairs dataloader for pairwise optical-flow pretraining.

Expected common FlyingChairs structure:
    FlyingChairs_release/
        data/
            00001_img1.ppm
            00001_img2.ppm
            00001_flow.flo
            ...

Output contract matches your KITTI dataloader enough for Trainer-style code:
    imgs: [2, 3, H, W]
    flow: [2, H, W]
    valid: [H, W]
    img_src/img_tgt
    src_idx_in_seq/tgt_idx_in_seq
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


TAG_FLOAT = 202021.25  # .flo magic number: 'PIEH'


def read_rgb_img(path: str | Path) -> np.ndarray:
    """Read image as RGB float32 in [0, 1], shape [H, W, 3]."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0


def read_flo(path: str | Path) -> np.ndarray:
    """Read Middlebury .flo file, return float32 flow [H, W, 2]."""
    path = Path(path)
    with path.open("rb") as f:
        tag = np.fromfile(f, np.float32, count=1)[0]
        if tag != TAG_FLOAT:
            raise RuntimeError(f"Wrong .flo tag in {path}; got {tag}")
        w = int(np.fromfile(f, np.int32, count=1)[0])
        h = int(np.fromfile(f, np.int32, count=1)[0])
        data = np.fromfile(f, np.float32, count=2 * w * h)
    if data.size != 2 * w * h:
        raise RuntimeError(f"Incomplete .flo file: {path}")
    return data.reshape(h, w, 2).astype(np.float32)


def resize_img_and_flow(
    img1: np.ndarray,
    img2: np.ndarray,
    flow: np.ndarray,
    size_hw: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resize images and correctly scale flow vectors."""
    out_h, out_w = size_hw
    h, w = flow.shape[:2]
    if (h, w) == (out_h, out_w):
        return img1, img2, flow

    img1_r = cv2.resize(img1, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    img2_r = cv2.resize(img2, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    flow_r = cv2.resize(flow, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    flow_r[..., 0] *= float(out_w) / float(w)
    flow_r[..., 1] *= float(out_h) / float(h)
    return img1_r, img2_r, flow_r


def random_crop_img_flow(
    img1: np.ndarray,
    img2: np.ndarray,
    flow: np.ndarray,
    crop_size: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random crop aligned img1/img2/flow."""
    crop_h, crop_w = crop_size
    h, w = flow.shape[:2]
    if crop_h > h or crop_w > w:
        raise RuntimeError(f"Crop size {crop_size} larger than image size {(h, w)}")
    if crop_h == h and crop_w == w:
        return img1, img2, flow
    top = random.randint(0, h - crop_h)
    left = random.randint(0, w - crop_w)
    return (
        img1[top:top + crop_h, left:left + crop_w],
        img2[top:top + crop_h, left:left + crop_w],
        flow[top:top + crop_h, left:left + crop_w],
    )


def center_crop_img_flow(
    img1: np.ndarray,
    img2: np.ndarray,
    flow: np.ndarray,
    crop_size: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    crop_h, crop_w = crop_size
    h, w = flow.shape[:2]
    if crop_h > h or crop_w > w:
        raise RuntimeError(f"Crop size {crop_size} larger than image size {(h, w)}")
    top = (h - crop_h) // 2
    left = (w - crop_w) // 2
    return (
        img1[top:top + crop_h, left:left + crop_w],
        img2[top:top + crop_h, left:left + crop_w],
        flow[top:top + crop_h, left:left + crop_w],
    )


class FlyingChairsDataset(Dataset):
    """
    Pairwise FlyingChairs dataset.

    Args:
        root: FlyingChairs_release root OR its data folder.
        split: 'train', 'val', or 'all'.
        split_file: optional RAFT-style chairs_split.txt. Usually lines contain 1=train, 2=val.
        crop_size: random crop size, e.g. (352, 512). Use multiples of 8 for your decoder.
        resize_size: optional resize before crop. Normally keep None.
        normalize: z-score normalize images.
        stats_in: optional stats json. If missing, compute and save.
        augment: random crop + simple color jitter/flip when True. Validation uses center crop.
        photometric_safe_aug: if True, avoid augmentations that break image-to-image photometric consistency.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        split_file: Optional[str | Path] = None,
        crop_size: Optional[Tuple[int, int]] = (352, 512),
        resize_size: Optional[Tuple[int, int]] = None,
        normalize: bool = True,
        stats_in: Optional[str | Path] = None,
        augment: bool = True,
        photometric_safe_aug: bool = True,
    ):
        super().__init__()
        self.root = Path(root)
        self.data_dir = self.root / "data" if (self.root / "data").exists() else self.root
        self.split = split
        self.split_file = Path(split_file) if split_file is not None else None
        self.crop_size = tuple(crop_size) if crop_size is not None else None
        self.resize_size = tuple(resize_size) if resize_size is not None else None
        self.normalize = normalize
        self.stats_in = Path(stats_in) if stats_in is not None else None
        self.augment = augment and split == "train"
        self.photometric_safe_aug = photometric_safe_aug

        if not self.data_dir.exists():
            raise FileNotFoundError(f"FlyingChairs data folder not found: {self.data_dir}")

        self.samples = self._build_samples()
        if len(self.samples) == 0:
            raise RuntimeError(f"No FlyingChairs samples found in {self.data_dir}")

        self.stats = self._load_stats()

    def _build_samples(self) -> List[Dict[str, str]]:
        flow_files = sorted(self.data_dir.glob("*_flow.flo"))
        samples = []
        for flow_path in flow_files:
            prefix = flow_path.name.replace("_flow.flo", "")
            img1 = self.data_dir / f"{prefix}_img1.ppm"
            img2 = self.data_dir / f"{prefix}_img2.ppm"
            if img1.exists() and img2.exists():
                samples.append({"prefix": prefix, "img1": str(img1), "img2": str(img2), "flow": str(flow_path)})

        if self.split_file is not None and self.split_file.exists() and self.split in {"train", "val"}:
            split_ids = [int(x.strip()) for x in self.split_file.read_text().splitlines() if x.strip()]
            # RAFT convention: 1=train, 2=val. If your file differs, adjust here.
            keep_value = 1 if self.split == "train" else 2
            samples = [s for s, sid in zip(samples, split_ids) if sid == keep_value]

        return samples

    def _load_stats(self) -> Dict[str, object]:
        if self.stats_in is not None and self.stats_in.exists():
            with self.stats_in.open("r") as f:
                return json.load(f)
        stats = self.compute_stats(max_images=2000)
        if self.stats_in is not None:
            self.stats_in.parent.mkdir(parents=True, exist_ok=True)
            with self.stats_in.open("w") as f:
                json.dump(stats, f, indent=2)
        return stats

    def compute_stats(self, max_images: int = 2000) -> Dict[str, object]:
        channel_sum = np.zeros(3, dtype=np.float64)
        channel_sq_sum = np.zeros(3, dtype=np.float64)
        pixel_count = 0
        img_paths = []
        for s in self.samples:
            img_paths.extend([s["img1"], s["img2"]])
        if len(img_paths) > max_images:
            rng = random.Random(1234)
            img_paths = rng.sample(img_paths, max_images)

        for p in img_paths:
            img = read_rgb_img(p)
            flat = img.reshape(-1, 3)
            channel_sum += flat.sum(axis=0)
            channel_sq_sum += (flat ** 2).sum(axis=0)
            pixel_count += flat.shape[0]

        mean = channel_sum / pixel_count
        var = channel_sq_sum / pixel_count - mean ** 2
        std = np.sqrt(np.maximum(var, 1e-8))
        return {"mean": mean.tolist(), "std": std.tolist(), "num_images_used": len(img_paths), "num_samples": len(self.samples)}

    def _normalize_img(self, img: np.ndarray) -> np.ndarray:
        mean = np.array(self.stats["mean"], dtype=np.float32)
        std = np.array(self.stats["std"], dtype=np.float32)
        return (img - mean) / std

    def _augment_pair(self, img1: np.ndarray, img2: np.ndarray, flow: np.ndarray):
        # Geometric horizontal flip. This preserves GT and photometric relation.
        if random.random() < 0.5:
            img1 = np.ascontiguousarray(img1[:, ::-1])
            img2 = np.ascontiguousarray(img2[:, ::-1])
            flow = np.ascontiguousarray(flow[:, ::-1])
            flow[..., 0] *= -1.0

        # Same brightness/contrast on both images, so photometric loss still makes sense.
        if random.random() < 0.5:
            scale = random.uniform(0.8, 1.2)
            shift = random.uniform(-0.05, 0.05)
            img1 = np.clip(img1 * scale + shift, 0.0, 1.0)
            img2 = np.clip(img2 * scale + shift, 0.0, 1.0)

        return img1, img2, flow

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        s = self.samples[idx]
        img1 = read_rgb_img(s["img1"])
        img2 = read_rgb_img(s["img2"])
        flow = read_flo(s["flow"])

        if self.resize_size is not None:
            img1, img2, flow = resize_img_and_flow(img1, img2, flow, self.resize_size)

        if self.augment:
            img1, img2, flow = self._augment_pair(img1, img2, flow)
            if self.crop_size is not None:
                img1, img2, flow = random_crop_img_flow(img1, img2, flow, self.crop_size)
        else:
            if self.crop_size is not None:
                img1, img2, flow = center_crop_img_flow(img1, img2, flow, self.crop_size)

        valid = np.isfinite(flow).all(axis=-1).astype(np.float32)
        flow = np.nan_to_num(flow, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        if self.normalize:
            img1 = self._normalize_img(img1)
            img2 = self._normalize_img(img2)

        img1_t = torch.from_numpy(img1).permute(2, 0, 1).contiguous().float()
        img2_t = torch.from_numpy(img2).permute(2, 0, 1).contiguous().float()
        imgs = torch.stack([img1_t, img2_t], dim=0)  # [2,3,H,W]

        flow_t = torch.from_numpy(flow).permute(2, 0, 1).contiguous().float()
        valid_t = torch.from_numpy(valid).contiguous().float()

        return {
            "imgs": imgs,
            "flow": flow_t,
            "valid": valid_t,
            "img_src": img1_t,
            "img_tgt": img2_t,
            "src_idx_in_seq": torch.tensor(0, dtype=torch.long),
            "tgt_idx_in_seq": torch.tensor(1, dtype=torch.long),
            "seq_id": s["prefix"],
            "flow_frame": torch.tensor(0, dtype=torch.long),
            "frame_indices": torch.tensor([0, 1], dtype=torch.long),
        }
