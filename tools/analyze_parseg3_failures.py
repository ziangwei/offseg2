# -*- coding: utf-8 -*-
"""Failure analysis for PARSeg3-style decode heads.

This script answers a narrow question before designing a new head: are the
remaining errors mostly resolution/boundary/small-object errors, or semantic
confusion/self-confident wrong predictions?

It is intentionally split into two layers:
  1. Pure NumPy helpers, covered by unit tests and importable without torch.
  2. A CLI that lazily imports torch/mmseg and runs a checkpoint on val images.

Example:
  python tools/analyze_parseg3_failures.py \
      local_configs/offseg2/Base/parseg3_ade20k_160k-512x512.py \
      work_dirs/parseg3_ade20k_160k-512x512/iter_160000.pth \
      --max-images 250
"""

import argparse
import datetime as _datetime
import os
from collections import defaultdict

import numpy as np


SUMMARY_KEYS = (
    "valid_px",
    "base_correct_px",
    "refine_correct_px",
    "final_correct_px",
    "base_wrong_px",
    "refine_wrong_px",
    "final_wrong_px",
    "bo_ro_px",
    "bo_rw_px",
    "bw_ro_px",
    "bw_rw_px",
    "base_wrong_final_correct_px",
    "base_correct_final_wrong_px",
    "refine_wrong_final_correct_px",
    "refine_correct_final_wrong_px",
    "base_self_confident_wrong_px",
    "base_top2_gt_hit_px",
    "final_wrong_top2_gt_hit_px",
    "boundary_wrong_px",
    "small_wrong_px",
    "resolution_wrong_px",
    "interior_large_wrong_px",
)


def _softmax_np(logits):
    logits = np.asarray(logits, dtype=np.float32)
    shifted = logits - logits.max(axis=0, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.maximum(exp.sum(axis=0, keepdims=True), 1e-12)


def _top2_np(probs):
    """Return top-1/top-2 class ids and probabilities for CxHxW probabilities."""
    probs = np.asarray(probs)
    if probs.ndim != 3:
        raise ValueError(f"expected probs with shape CxHxW, got {probs.shape}")

    num_classes = probs.shape[0]
    top1_idx = probs.argmax(axis=0).astype(np.int64)
    top1_prob = np.take_along_axis(probs, top1_idx[None, ...], axis=0)[0]

    if num_classes == 1:
        top2_idx = np.full_like(top1_idx, -1)
        top2_prob = np.zeros_like(top1_prob)
        return top1_idx, top2_idx, top1_prob, top2_prob

    candidate_ids = np.argpartition(probs, kth=num_classes - 2, axis=0)[-2:]
    candidate_probs = np.take_along_axis(probs, candidate_ids, axis=0)
    candidate_order = np.argsort(candidate_probs, axis=0)
    top2_rel = candidate_order[-2:-1]
    top2_idx = np.take_along_axis(candidate_ids, top2_rel, axis=0)[0].astype(np.int64)
    top2_prob = np.take_along_axis(candidate_probs, top2_rel, axis=0)[0]
    return top1_idx, top2_idx, top1_prob, top2_prob


def _as_2d_target(target):
    target = np.asarray(target)
    if target.ndim == 3 and target.shape[0] == 1:
        target = target[0]
    if target.ndim != 2:
        raise ValueError(f"expected target with shape HxW, got {target.shape}")
    return target.astype(np.int64, copy=False)


def _as_chw_logits(logits, target_shape, name):
    logits = np.asarray(logits, dtype=np.float32)
    if logits.ndim == 4 and logits.shape[0] == 1:
        logits = logits[0]
    if logits.ndim != 3:
        raise ValueError(f"expected {name} with shape CxHxW, got {logits.shape}")
    if tuple(logits.shape[-2:]) != tuple(target_shape):
        raise ValueError(
            f"{name} spatial shape {logits.shape[-2:]} does not match target {target_shape}"
        )
    return logits


def boundary_mask_from_labels(labels, ignore_index=255):
    """4-neighbor GT boundary mask, excluding ignore-label pairs."""
    labels = _as_2d_target(labels)
    valid = labels != ignore_index
    out = np.zeros(labels.shape, dtype=bool)

    if labels.shape[0] > 1:
        a = labels[:-1, :]
        b = labels[1:, :]
        pair = (a != b) & valid[:-1, :] & valid[1:, :]
        out[:-1, :] |= pair
        out[1:, :] |= pair

    if labels.shape[1] > 1:
        a = labels[:, :-1]
        b = labels[:, 1:]
        pair = (a != b) & valid[:, :-1] & valid[:, 1:]
        out[:, :-1] |= pair
        out[:, 1:] |= pair

    return out


def connected_component_small_mask(labels, ignore_index=255, max_px=64):
    """Mask GT connected components whose area is <= max_px.

    Components are computed per class with 4-neighbor connectivity. This is
    deliberately not class-frequency based: a class can be common globally while
    still containing tiny disconnected instances.
    """
    labels = _as_2d_target(labels)
    if max_px <= 0:
        return np.zeros(labels.shape, dtype=bool)

    h, w = labels.shape
    visited = np.zeros((h, w), dtype=bool)
    small = np.zeros((h, w), dtype=bool)

    for sy in range(h):
        for sx in range(w):
            if visited[sy, sx] or labels[sy, sx] == ignore_index:
                continue

            cls = labels[sy, sx]
            stack = [(sy, sx)]
            visited[sy, sx] = True
            coords = []

            while stack:
                y, x = stack.pop()
                coords.append((y, x))

                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if ny < 0 or ny >= h or nx < 0 or nx >= w:
                        continue
                    if visited[ny, nx] or labels[ny, nx] != cls:
                        continue
                    visited[ny, nx] = True
                    stack.append((ny, nx))

            if len(coords) <= max_px:
                for y, x in coords:
                    small[y, x] = True

    return small


def summarize_prediction_arrays(
    base_logits,
    refine_logits,
    final_logits,
    target,
    ignore_index=255,
    self_conf_threshold=0.90,
    margin_threshold=0.50,
    small_component_max_px=64,
):
    """Summarize one image worth of PARSeg3 predictions.

    Inputs are CxHxW logits, already resized to the target grid.
    """
    target = _as_2d_target(target)
    target_shape = target.shape
    base_logits = _as_chw_logits(base_logits, target_shape, "base_logits")
    refine_logits = _as_chw_logits(refine_logits, target_shape, "refine_logits")
    final_logits = _as_chw_logits(final_logits, target_shape, "final_logits")

    valid = target != ignore_index

    base_probs = _softmax_np(base_logits)
    refine_probs = _softmax_np(refine_logits)
    final_probs = _softmax_np(final_logits)

    base_pred, base_top2, base_conf, base_second = _top2_np(base_probs)
    refine_pred = refine_probs.argmax(axis=0).astype(np.int64)
    final_pred, final_top2, _, _ = _top2_np(final_probs)

    base_ok = (base_pred == target) & valid
    refine_ok = (refine_pred == target) & valid
    final_ok = (final_pred == target) & valid

    base_wrong = (~base_ok) & valid
    refine_wrong = (~refine_ok) & valid
    final_wrong = (~final_ok) & valid

    base_margin = base_conf - base_second
    base_self_confident_wrong = (
        base_wrong
        & (base_conf >= float(self_conf_threshold))
        & (base_margin >= float(margin_threshold))
    )

    base_gt_in_top2 = ((target == base_pred) | (target == base_top2)) & valid
    final_gt_in_top2 = ((target == final_pred) | (target == final_top2)) & valid

    boundary = boundary_mask_from_labels(target, ignore_index=ignore_index)
    small = connected_component_small_mask(
        target, ignore_index=ignore_index, max_px=small_component_max_px
    )
    resolution_region = boundary | small

    out = {key: 0 for key in SUMMARY_KEYS}
    out.update(
        valid_px=int(valid.sum()),
        base_correct_px=int(base_ok.sum()),
        refine_correct_px=int(refine_ok.sum()),
        final_correct_px=int(final_ok.sum()),
        base_wrong_px=int(base_wrong.sum()),
        refine_wrong_px=int(refine_wrong.sum()),
        final_wrong_px=int(final_wrong.sum()),
        bo_ro_px=int((base_ok & refine_ok).sum()),
        bo_rw_px=int((base_ok & refine_wrong).sum()),
        bw_ro_px=int((base_wrong & refine_ok).sum()),
        bw_rw_px=int((base_wrong & refine_wrong).sum()),
        base_wrong_final_correct_px=int((base_wrong & final_ok).sum()),
        base_correct_final_wrong_px=int((base_ok & final_wrong).sum()),
        refine_wrong_final_correct_px=int((refine_wrong & final_ok).sum()),
        refine_correct_final_wrong_px=int((refine_ok & final_wrong).sum()),
        base_self_confident_wrong_px=int(base_self_confident_wrong.sum()),
        base_top2_gt_hit_px=int((base_wrong & base_gt_in_top2).sum()),
        final_wrong_top2_gt_hit_px=int((final_wrong & final_gt_in_top2).sum()),
        boundary_wrong_px=int((final_wrong & boundary).sum()),
        small_wrong_px=int((final_wrong & small).sum()),
        resolution_wrong_px=int((final_wrong & resolution_region).sum()),
        interior_large_wrong_px=int((final_wrong & ~resolution_region).sum()),
    )
    return out


def merge_summaries(total, item):
    for key in SUMMARY_KEYS:
        total[key] += int(item.get(key, 0))


def _ratio(num, den):
    return 0.0 if den <= 0 else float(num) / float(den)


def _pct(num, den):
    return 100.0 * _ratio(num, den)


def _fmt_count_share(summary, key, denominator_key):
    return (
        f"{summary[key]} "
        f"({summary[key] / max(summary[denominator_key], 1):.4f} of {denominator_key})"
    )


def route_hints(summary):
    final_wrong = max(summary["final_wrong_px"], 1)
    base_wrong = max(summary["base_wrong_px"], 1)

    resolution_share = _ratio(summary["resolution_wrong_px"], final_wrong)
    self_conf_share = _ratio(summary["base_self_confident_wrong_px"], base_wrong)
    final_top2_share = _ratio(summary["final_wrong_top2_gt_hit_px"], final_wrong)
    interior_share = _ratio(summary["interior_large_wrong_px"], final_wrong)

    hints = []
    if resolution_share >= 0.50:
        hints.append(
            "resolution-first: boundary/small-object errors dominate; test UPLiFT/NAF/BRDG-style refinement before another prototype route."
        )
    elif interior_share >= 0.50:
        hints.append(
            "semantic-first: many errors are interior large-region errors; prototype/region reasoning may be worth testing."
        )
    else:
        hints.append(
            "mixed: neither resolution nor interior errors dominate; keep the next experiment narrowly isolated."
        )

    if self_conf_share >= 0.20:
        hints.append(
            "prototype-risk: high-confidence wrong base pixels are common; naive confidence-filtered CPC/stable prototypes are unsafe."
        )
    else:
        hints.append(
            "prototype-risk: self-confident wrong base pixels are limited under the current thresholds."
        )

    if final_top2_share >= 0.35:
        hints.append(
            "top2-signal: many final errors still contain GT in top-2; ambiguity-aware correction is plausible."
        )
    else:
        hints.append(
            "top2-signal: GT is often not in top-2, so simple top-k disambiguation is unlikely to carry the project."
        )

    return hints


def format_report(summary, meta):
    n = max(summary["valid_px"], 1)
    final_wrong = max(summary["final_wrong_px"], 1)
    base_wrong = max(summary["base_wrong_px"], 1)

    lines = []
    lines.append("[ID]")
    lines.append(f"CONFIG={meta.get('config', '?')}")
    lines.append(f"CKPT={meta.get('checkpoint', '?')} ITER={meta.get('iter', '?')}")
    lines.append(
        f"EVAL n_images={meta.get('n_images', '?')}/{meta.get('n_total', '?')} "
        f"date={_datetime.date.today()} mode=whole-image-diagnostic"
    )
    lines.append(
        "THRESHOLDS "
        f"self_conf={meta.get('self_conf_threshold')} "
        f"margin={meta.get('margin_threshold')} "
        f"small_component_max_px={meta.get('small_component_max_px')}"
    )

    lines.append("[HEADS]")
    lines.append(
        "acc "
        f"base={summary['base_correct_px'] / n:.4f} "
        f"refine={summary['refine_correct_px'] / n:.4f} "
        f"final={summary['final_correct_px'] / n:.4f}"
    )
    lines.append(
        "base/refine cells "
        f"bo_ro={summary['bo_ro_px'] / n:.4f} "
        f"bo_rw={summary['bo_rw_px'] / n:.4f} "
        f"bw_ro={summary['bw_ro_px'] / n:.4f} "
        f"bw_rw={summary['bw_rw_px'] / n:.4f}"
    )
    lines.append(
        "fusion movement "
        f"base_wrong->final_correct={_pct(summary['base_wrong_final_correct_px'], base_wrong):.2f}% "
        f"base_correct->final_wrong={_pct(summary['base_correct_final_wrong_px'], summary['base_correct_px']):.2f}% "
        f"refine_wrong->final_correct={_pct(summary['refine_wrong_final_correct_px'], max(summary['refine_wrong_px'], 1)):.2f}%"
    )

    lines.append("[FAILURE_SPLIT]")
    lines.append(f"final_wrong_px={summary['final_wrong_px']} ({summary['final_wrong_px'] / n:.4f} of valid)")
    lines.append(f"boundary_wrong_px={_fmt_count_share(summary, 'boundary_wrong_px', 'final_wrong_px')}")
    lines.append(f"small_wrong_px={_fmt_count_share(summary, 'small_wrong_px', 'final_wrong_px')}")
    lines.append(f"resolution_wrong_px={_fmt_count_share(summary, 'resolution_wrong_px', 'final_wrong_px')}")
    lines.append(f"interior_large_wrong_px={_fmt_count_share(summary, 'interior_large_wrong_px', 'final_wrong_px')}")

    lines.append("[CONFIDENCE_RISK]")
    lines.append(
        f"base_self_confident_wrong_px={summary['base_self_confident_wrong_px']} "
        f"({summary['base_self_confident_wrong_px'] / base_wrong:.4f} of base_wrong)"
    )
    lines.append(
        f"base_wrong_with_GT_in_top2={summary['base_top2_gt_hit_px']} "
        f"({summary['base_top2_gt_hit_px'] / base_wrong:.4f} of base_wrong)"
    )
    lines.append(
        f"final_wrong_with_GT_in_top2={summary['final_wrong_top2_gt_hit_px']} "
        f"({summary['final_wrong_top2_gt_hit_px'] / final_wrong:.4f} of final_wrong)"
    )

    lines.append("[ROUTE_HINT]")
    for hint in route_hints(summary):
        lines.append(f"- {hint}")

    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze PARSeg3 base/refine/final failures on a validation subset."
    )
    parser.add_argument("config", help="mmseg config path")
    parser.add_argument("checkpoint", help="checkpoint path")
    parser.add_argument("--max-images", type=int, default=250, help="number of evenly sampled val images")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", default=None, help="report path; default is next to checkpoint")
    parser.add_argument("--self-conf-threshold", type=float, default=0.90)
    parser.add_argument("--margin-threshold", type=float, default=0.50)
    parser.add_argument("--small-component-max-px", type=int, default=64)
    parser.add_argument("--progress-interval", type=int, default=25)
    return parser.parse_args()


def _get_iter_meta(torch_module, checkpoint):
    try:
        meta = torch_module.load(checkpoint, map_location="cpu").get("meta", {})
        return str(meta.get("iter", "?"))
    except Exception:
        return "?"


def _prepare_inputs_and_gt(data, model, device, ignore_index):
    import torch.nn.functional as F

    data = model.data_preprocessor(data, False)
    inputs = data["inputs"]
    if isinstance(inputs, (list, tuple)):
        inputs = inputs[0].unsqueeze(0)
    inputs = inputs.to(device)

    gt = data["data_samples"][0].gt_sem_seg.data.to(device)
    h, w = inputs.shape[-2:]
    if tuple(gt.shape[-2:]) != (h, w):
        gt = F.interpolate(gt.unsqueeze(0).float(), size=(h, w), mode="nearest").squeeze(0).long()

    pad_h = (32 - h % 32) % 32
    pad_w = (32 - w % 32) % 32
    if pad_h or pad_w:
        inputs = F.pad(inputs, (0, pad_w, 0, pad_h), value=0.0)
        gt = F.pad(gt, (0, pad_w, 0, pad_h), value=ignore_index)

    return inputs, gt


def _resize_logits(logits, size):
    import torch.nn.functional as F

    if tuple(logits.shape[-2:]) == tuple(size):
        return logits
    return F.interpolate(logits, size=size, mode="bilinear", align_corners=False)


def _to_numpy_logits(logits):
    return logits[0].detach().float().cpu().numpy()


def main():
    args = parse_args()

    import torch
    from mmengine.config import Config
    from mmengine.dataset import pseudo_collate
    from mmengine.registry import init_default_scope
    from mmseg.apis import init_model
    from mmseg.registry import DATASETS

    cfg = Config.fromfile(args.config)
    init_default_scope("mmseg")

    model = init_model(args.config, args.checkpoint, device=args.device)
    model.eval()
    ignore_index = int(getattr(model.decode_head, "ignore_index", 255))

    ds_cfg = cfg.val_dataloader.dataset if "val_dataloader" in cfg else cfg.test_dataloader.dataset
    dataset = DATASETS.build(ds_cfg)
    n_total = len(dataset)
    n_images = min(int(args.max_images), n_total)
    indices = np.linspace(0, n_total - 1, n_images).astype(int)

    total = defaultdict(int)

    with torch.no_grad():
        for k, idx in enumerate(indices):
            data = pseudo_collate([dataset[int(idx)]])
            inputs, gt = _prepare_inputs_and_gt(data, model, args.device, ignore_index)

            feats = model.extract_feat(inputs)
            outputs = model.decode_head.forward(list(feats))
            missing = {"base_head_logits", "refinement_head_logits", "final_logits"} - set(outputs)
            if missing:
                raise KeyError(f"decode_head output is missing keys: {sorted(missing)}")

            target_size = tuple(gt.shape[-2:])
            base = _resize_logits(outputs["base_head_logits"], target_size)
            refine = _resize_logits(outputs["refinement_head_logits"], target_size)
            final = _resize_logits(outputs["final_logits"], target_size)

            item = summarize_prediction_arrays(
                base_logits=_to_numpy_logits(base),
                refine_logits=_to_numpy_logits(refine),
                final_logits=_to_numpy_logits(final),
                target=gt[0].detach().cpu().numpy(),
                ignore_index=ignore_index,
                self_conf_threshold=args.self_conf_threshold,
                margin_threshold=args.margin_threshold,
                small_component_max_px=args.small_component_max_px,
            )
            merge_summaries(total, item)

            if args.progress_interval > 0 and (k + 1) % args.progress_interval == 0:
                print(f"... {k + 1}/{n_images} images")

    summary = {key: int(total[key]) for key in SUMMARY_KEYS}
    meta = dict(
        config=os.path.splitext(os.path.basename(args.config))[0],
        checkpoint=os.path.basename(args.checkpoint),
        iter=_get_iter_meta(torch, args.checkpoint),
        n_images=n_images,
        n_total=n_total,
        self_conf_threshold=args.self_conf_threshold,
        margin_threshold=args.margin_threshold,
        small_component_max_px=args.small_component_max_px,
    )
    report = format_report(summary, meta)
    print("\n" + report)

    out = args.out or os.path.join(
        os.path.dirname(os.path.abspath(args.checkpoint)),
        f"parseg3_failure_analysis_{os.path.splitext(os.path.basename(args.checkpoint))[0]}.txt",
    )
    with open(out, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
