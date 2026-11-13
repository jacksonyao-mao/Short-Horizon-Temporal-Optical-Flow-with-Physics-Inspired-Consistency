"""Evaluation-only metrics. Fl values are outlier percentages, not F1 scores."""
import numpy as np
import torch
import torch.nn.functional as F


def require_finite(*tensors):
    if any(not torch.isfinite(x).all().item() for x in tensors):
        raise ValueError("Non-finite predictions/data; refusing to silently discard errors.")


def ratio(numerator, denominator, scale=1.0):
    return scale * numerator / denominator if denominator else None


def flow_statistics(pred, gt, valid, foreground):
    """Single sample: flow [2,H,W], masks [H,W]. Return sufficient statistics."""
    if pred.shape != gt.shape or pred.shape[0] != 2:
        raise ValueError("Expected matching [2,H,W] flow tensors.")
    if valid.shape != pred.shape[1:] or foreground.shape != valid.shape:
        raise ValueError("Flow and masks must have identical spatial dimensions.")
    require_finite(pred, gt, valid, foreground)
    epe = torch.linalg.vector_norm(pred - gt, dim=0)
    magnitude = torch.linalg.vector_norm(gt, dim=0)
    bad = (epe > 3.0) & (epe / (magnitude + 1e-9) > 0.05)
    valid = valid > 0.5
    fg = foreground > 0
    result = {}
    for region, mask in {"all": valid, "fg": valid & fg, "bg": valid & ~fg}.items():
        count = int(mask.sum().item())
        error_sum = float(epe[mask].double().sum().item())
        bad_count = int(bad[mask].sum().item())
        result.update({f"{region}_pixels": count, f"{region}_epe_sum": error_sum,
                       f"{region}_bad_pixels": bad_count,
                       f"epe_{region}": ratio(error_sum, count),
                       f"fl_{region}_percent": ratio(bad_count, count, 100.0)})
    return result


def sample_at_flow(tensor, flow):
    """Bilinear sample tensor at x+flow(x); return samples and in-bounds mask."""
    b, _, h, w = flow.shape
    yy, xx = torch.meshgrid(torch.arange(h, device=flow.device),
                            torch.arange(w, device=flow.device), indexing="ij")
    x = xx[None] + flow[:, 0]
    y = yy[None] + flow[:, 1]
    valid = (x >= 0) & (x <= w - 1) & (y >= 0) & (y <= h - 1)
    grid = torch.stack((2 * x / max(w - 1, 1) - 1,
                        2 * y / max(h - 1, 1) - 1), dim=-1)
    return F.grid_sample(tensor, grid, mode="bilinear", padding_mode="border",
                         align_corners=True), valid


def erode_valid(mask, radius):
    if radius == 0:
        return mask
    k = 2 * radius + 1
    padded = F.pad(mask[:, None].float(), (radius,) * 4, value=0)
    return F.avg_pool2d(padded, k, stride=1)[:, 0] >= 1 - 1e-7


def two_step_statistics(rgb, flows, photo_map_fn, radius=3):
    """RGB [T,3,H,W] in [0,1], flows [T-1,2,H,W]. Uniform spatial weighting.

    Both the intermediate and final sample must be in bounds. Erode the valid
    mask to keep every SSIM/Census neighborhood valid. No predicted occlusion,
    texture, confidence, or error-dependent weighting is used.
    """
    require_finite(rgb, flows)
    if rgb.ndim != 4 or flows.shape != (len(rgb) - 1, 2, *rgb.shape[-2:]):
        raise ValueError("Expected T RGB frames and T-1 matching flow maps.")
    if len(rgb) < 3 or radius < 0:
        raise ValueError("Two-step evaluation needs >=3 frames and nonnegative radius.")
    if min(rgb.shape[-2:]) <= 2 * radius:
        raise ValueError("Images are too small for the photometric neighborhood.")
    if rgb.min() < -1e-5 or rgb.max() > 1 + 1e-5:
        raise ValueError("Reconstruction metrics require RGB in [0,1].")
    result = {"two_step_l1_sum": 0.0, "two_step_photo_sum": 0.0,
              "two_step_pixels": 0, "two_step_candidate_pixels": 0}
    for t in range(len(rgb) - 2):
        first = flows[t:t+1]
        second_warped, intermediate_valid = sample_at_flow(flows[t+1:t+2], first)
        composed = first + second_warped
        target, final_valid = sample_at_flow(rgb[t+2:t+3], composed)
        mask = erode_valid(intermediate_valid & final_valid, radius)
        l1 = (rgb[t:t+1] - target).abs().mean(dim=1)
        photo = photo_map_fn(rgb[t:t+1], rgb[t+2:t+3], composed)
        require_finite(photo)
        if photo.shape != mask.shape:
            raise ValueError("Photometric callback must return [B,H,W].")
        result["two_step_l1_sum"] += float(l1[mask].double().sum().item())
        result["two_step_photo_sum"] += float(photo[mask].double().sum().item())
        result["two_step_pixels"] += int(mask.sum().item())
        result["two_step_candidate_pixels"] += mask.numel()
    result["two_step_l1"] = ratio(result["two_step_l1_sum"], result["two_step_pixels"])
    result["two_step_photo"] = ratio(result["two_step_photo_sum"], result["two_step_pixels"])
    result["two_step_coverage_percent"] = ratio(result["two_step_pixels"],
                                                result["two_step_candidate_pixels"], 100)
    return result


METRIC_COLUMNS = {
    **{f"epe_{r}": (f"{r}_epe_sum", f"{r}_pixels", 1.0) for r in ("all", "fg", "bg")},
    **{f"fl_{r}_percent": (f"{r}_bad_pixels", f"{r}_pixels", 100.0) for r in ("all", "fg", "bg")},
    "two_step_l1": ("two_step_l1_sum", "two_step_pixels", 1.0),
    "two_step_photo": ("two_step_photo_sum", "two_step_pixels", 1.0),
}


def summarize(rows):
    result = {}
    for metric, (num, den, scale) in METRIC_COLUMNS.items():
        result[metric] = ratio(sum(float(r[num]) for r in rows),
                               sum(int(r[den]) for r in rows), scale)
    image_epe = [float(r["all_epe_sum"]) / int(r["all_pixels"])
                 for r in rows if int(r["all_pixels"]) > 0]
    result["per_image_epe_mean"] = float(np.mean(image_epe)) if image_epe else None
    result["per_image_epe_median"] = float(np.median(image_epe)) if image_epe else None
    result["per_image_epe_p90"] = float(np.percentile(image_epe, 90)) if image_epe else None
    result["two_step_coverage_percent"] = ratio(
        sum(int(r["two_step_pixels"]) for r in rows),
        sum(int(r["two_step_candidate_pixels"]) for r in rows), 100)
    return result


def paired_bootstrap(rows_a, rows_b, repeats=5000, seed=42):
    """Paired percentile bootstrap of A-B, clustered by KITTI sequence ID."""
    if repeats < 100:
        raise ValueError("Use at least 100 bootstrap replicates.")
    def index(rows):
        out = {r["sample_id"]: r for r in rows}
        if len(out) != len(rows):
            raise ValueError("Duplicate sample IDs.")
        return out
    a, b = index(rows_a), index(rows_b)
    if not a or a.keys() != b.keys():
        raise ValueError("Paired comparison requires identical, nonempty sample ID sets.")
    scenes = sorted({r["scene_id"] for r in rows_a})
    if len(scenes) < 2:
        raise ValueError("Need at least two independent scenes for a bootstrap interval.")
    for key in a:
        if a[key]["scene_id"] != b[key]["scene_id"]:
            raise ValueError("Scene IDs do not match.")
        for region in ("all", "fg", "bg"):
            col = f"{region}_pixels"
            if int(a[key][col]) != int(b[key][col]):
                raise ValueError(f"Ground-truth region counts differ for {key}/{region}.")
    rng = np.random.default_rng(seed)
    # The same scene draws are used for every metric and both models.
    draws = rng.integers(0, len(scenes), size=(repeats, len(scenes)))
    result = {}
    for metric, (num, den, scale) in METRIC_COLUMNS.items():
        arrays = []
        for records in (a, b):
            arr = np.array([[sum(float(r[num]) for r in records.values() if r["scene_id"] == s),
                             sum(float(r[den]) for r in records.values() if r["scene_id"] == s)]
                            for s in scenes], dtype=np.float64)
            if not np.isfinite(arr).all() or (arr < 0).any():
                raise ValueError("Invalid metric sums/counts in comparison input.")
            arrays.append(arr)
        av, bv = (arr[draws].sum(axis=1) for arr in arrays)
        good = (av[:, 1] > 0) & (bv[:, 1] > 0)
        deltas = scale * (av[good, 0] / av[good, 1] - bv[good, 0] / bv[good, 1])
        totals = [arr.sum(axis=0) for arr in arrays]
        point = (scale * (totals[0][0] / totals[0][1] - totals[1][0] / totals[1][1])
                 if all(t[1] > 0 for t in totals) else None)
        result[metric] = {"difference_a_minus_b": point,
                          "ci95": np.percentile(deltas, [2.5, 97.5]).tolist() if len(deltas) else None,
                          "valid_replicates": int(good.sum()), "requested_replicates": repeats}
    return {"scene_count": len(scenes), "sample_count": len(a), "seed": seed,
            "method": "paired scene-cluster percentile bootstrap; negative favors A",
            "limitation": "Scene-sampling uncertainty only; not training-seed uncertainty.",
            "metrics": result}
