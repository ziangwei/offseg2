# -*- coding: utf-8 -*-
"""Second-stage confusion analysis for PARSeg3-style decode heads.

This script should be run after ``analyze_parseg3_failures.py`` shows that most
errors are interior semantic errors. It ranks the concrete GT->prediction
confusion pairs and measures whether base/refine/final share the same wrong
answer.

Example for the current PARSeg3 run:
  python tools/analyze_parseg3_confusions.py \
      local_configs/offseg2/Base/parseg3_ade20k_160k-512x512.py \
      work_dirs/parseg3_ade20k_160k-512x512_4x4_try1/iter_160000.pth \
      --max-images 250 --device cuda:0
"""

import argparse
import datetime as _datetime
import os
from collections import Counter, defaultdict

import numpy as np


SCALAR_KEYS = (
    "valid_px",
    "base_wrong_px",
    "refine_wrong_px",
    "final_wrong_px",
    "final_wrong_top2_gt_hit_px",
    "base_wrong_gt_top2_px",
    "base_self_confident_wrong_px",
    "base_self_confident_wrong_gt_top2_px",
    "final_wrong_base_same_pred_px",
    "final_wrong_refine_same_pred_px",
    "final_wrong_all_heads_same_pred_px",
    "base_refine_same_wrong_px",
)

COUNTER_KEYS = (
    "pair_counts",
    "pair_final_top2_gt_hit",
    "pair_base_same_pred",
    "pair_refine_same_pred",
    "pair_all_heads_same_pred",
    "pair_base_self_confident_wrong",
    "class_valid",
    "class_final_wrong",
    "class_final_top2_gt_hit",
    "class_base_wrong",
    "class_base_self_confident_wrong",
)


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


def _top2_from_logits_np(logits, need_probs=False):
    """Return top-1/top-2 class ids and, optionally, their softmax probabilities."""
    logits = np.asarray(logits, dtype=np.float32)
    if logits.ndim != 3:
        raise ValueError(f"expected logits with shape CxHxW, got {logits.shape}")

    num_classes = logits.shape[0]
    top1 = logits.argmax(axis=0).astype(np.int64)

    if num_classes == 1:
        top2 = np.full_like(top1, -1)
        if not need_probs:
            return top1, top2, None, None
        return top1, top2, np.ones_like(logits[0], dtype=np.float32), np.zeros_like(logits[0], dtype=np.float32)

    candidate_ids = np.argpartition(logits, kth=num_classes - 2, axis=0)[-2:]
    candidate_logits = np.take_along_axis(logits, candidate_ids, axis=0)
    candidate_order = np.argsort(candidate_logits, axis=0)
    top1_rel = candidate_order[-1:]
    top2_rel = candidate_order[-2:-1]
    top1 = np.take_along_axis(candidate_ids, top1_rel, axis=0)[0].astype(np.int64)
    top2 = np.take_along_axis(candidate_ids, top2_rel, axis=0)[0].astype(np.int64)

    if not need_probs:
        return top1, top2, None, None

    top1_logits = np.take_along_axis(logits, top1[None, ...], axis=0)[0]
    top2_logits = np.take_along_axis(logits, top2[None, ...], axis=0)[0]
    shifted = logits - logits.max(axis=0, keepdims=True)
    denom = np.exp(shifted).sum(axis=0)
    max_logits = logits.max(axis=0)
    top1_prob = np.exp(top1_logits - max_logits) / np.maximum(denom, 1e-12)
    top2_prob = np.exp(top2_logits - max_logits) / np.maximum(denom, 1e-12)
    return top1, top2, top1_prob, top2_prob


def _empty_summary():
    out = {key: 0 for key in SCALAR_KEYS}
    for key in COUNTER_KEYS:
        out[key] = Counter()
    return out


def summarize_confusion_arrays(
    base_logits,
    refine_logits,
    final_logits,
    target,
    ignore_index=255,
    self_conf_threshold=0.90,
    margin_threshold=0.50,
):
    """Summarize one image worth of semantic confusion statistics."""
    target = _as_2d_target(target)
    target_shape = target.shape
    base_logits = _as_chw_logits(base_logits, target_shape, "base_logits")
    refine_logits = _as_chw_logits(refine_logits, target_shape, "refine_logits")
    final_logits = _as_chw_logits(final_logits, target_shape, "final_logits")

    valid = target != ignore_index
    base_pred, base_top2, base_conf, base_second = _top2_from_logits_np(base_logits, need_probs=True)
    refine_pred, _, _, _ = _top2_from_logits_np(refine_logits, need_probs=False)
    final_pred, final_top2, _, _ = _top2_from_logits_np(final_logits, need_probs=False)

    base_wrong = (base_pred != target) & valid
    refine_wrong = (refine_pred != target) & valid
    final_wrong = (final_pred != target) & valid
    base_gt_top2 = ((target == base_pred) | (target == base_top2)) & valid
    final_gt_top2 = ((target == final_pred) | (target == final_top2)) & valid

    base_self_confident_wrong = (
        base_wrong
        & (base_conf >= float(self_conf_threshold))
        & ((base_conf - base_second) >= float(margin_threshold))
    )

    same_base = final_wrong & (base_pred == final_pred)
    same_refine = final_wrong & (refine_pred == final_pred)
    same_all = same_base & same_refine
    base_refine_same_wrong = base_wrong & refine_wrong & (base_pred == refine_pred)

    out = _empty_summary()
    out["valid_px"] = int(valid.sum())
    out["base_wrong_px"] = int(base_wrong.sum())
    out["refine_wrong_px"] = int(refine_wrong.sum())
    out["final_wrong_px"] = int(final_wrong.sum())
    out["final_wrong_top2_gt_hit_px"] = int((final_wrong & final_gt_top2).sum())
    out["base_wrong_gt_top2_px"] = int((base_wrong & base_gt_top2).sum())
    out["base_self_confident_wrong_px"] = int(base_self_confident_wrong.sum())
    out["base_self_confident_wrong_gt_top2_px"] = int((base_self_confident_wrong & base_gt_top2).sum())
    out["final_wrong_base_same_pred_px"] = int(same_base.sum())
    out["final_wrong_refine_same_pred_px"] = int(same_refine.sum())
    out["final_wrong_all_heads_same_pred_px"] = int(same_all.sum())
    out["base_refine_same_wrong_px"] = int(base_refine_same_wrong.sum())

    for cls, count in zip(*np.unique(target[valid], return_counts=True)):
        out["class_valid"][int(cls)] += int(count)

    wrong_coords = np.argwhere(final_wrong)
    for y, x in wrong_coords:
        gt_cls = int(target[y, x])
        pred_cls = int(final_pred[y, x])
        pair = (gt_cls, pred_cls)
        out["pair_counts"][pair] += 1
        out["class_final_wrong"][gt_cls] += 1

        if final_gt_top2[y, x]:
            out["pair_final_top2_gt_hit"][pair] += 1
            out["class_final_top2_gt_hit"][gt_cls] += 1
        if same_base[y, x]:
            out["pair_base_same_pred"][pair] += 1
        if same_refine[y, x]:
            out["pair_refine_same_pred"][pair] += 1
        if same_all[y, x]:
            out["pair_all_heads_same_pred"][pair] += 1
        if base_self_confident_wrong[y, x]:
            out["pair_base_self_confident_wrong"][pair] += 1

    for cls, count in zip(*np.unique(target[base_wrong], return_counts=True)):
        out["class_base_wrong"][int(cls)] += int(count)
    for cls, count in zip(*np.unique(target[base_self_confident_wrong], return_counts=True)):
        out["class_base_self_confident_wrong"][int(cls)] += int(count)

    return out


def merge_summaries(total, item):
    for key in SCALAR_KEYS:
        total[key] += int(item.get(key, 0))
    for key in COUNTER_KEYS:
        total[key].update(item.get(key, {}))


def _ratio(num, den):
    return 0.0 if den <= 0 else float(num) / float(den)


def _class_name(class_names, idx):
    if class_names and 0 <= int(idx) < len(class_names):
        return str(class_names[int(idx)])
    return str(int(idx))


def top_confusion_pairs(summary, class_names=None, topk=20, min_count=1):
    pairs = []
    final_wrong = max(int(summary.get("final_wrong_px", 0)), 1)
    for (gt_cls, pred_cls), count in summary.get("pair_counts", {}).items():
        count = int(count)
        if count < min_count:
            continue
        pairs.append(
            dict(
                gt=int(gt_cls),
                pred=int(pred_cls),
                gt_name=_class_name(class_names, gt_cls),
                pred_name=_class_name(class_names, pred_cls),
                count=count,
                share=_ratio(count, final_wrong),
                top2_rate=_ratio(summary["pair_final_top2_gt_hit"][(gt_cls, pred_cls)], count),
                base_same_pred_rate=_ratio(summary["pair_base_same_pred"][(gt_cls, pred_cls)], count),
                refine_same_pred_rate=_ratio(summary["pair_refine_same_pred"][(gt_cls, pred_cls)], count),
                all_heads_same_pred_rate=_ratio(summary["pair_all_heads_same_pred"][(gt_cls, pred_cls)], count),
                self_conf_rate=_ratio(summary["pair_base_self_confident_wrong"][(gt_cls, pred_cls)], count),
            )
        )
    pairs.sort(key=lambda item: (-item["count"], item["gt"], item["pred"]))
    return pairs[:topk]


def top_failed_classes(summary, class_names=None, topk=20, min_count=1):
    rows = []
    for cls, valid_count in summary.get("class_valid", {}).items():
        valid_count = int(valid_count)
        wrong = int(summary.get("class_final_wrong", {}).get(cls, 0))
        if wrong < min_count:
            continue
        rows.append(
            dict(
                cls=int(cls),
                name=_class_name(class_names, cls),
                valid=valid_count,
                wrong=wrong,
                wrong_rate=_ratio(wrong, valid_count),
                top2_rate=_ratio(summary.get("class_final_top2_gt_hit", {}).get(cls, 0), wrong),
                self_conf_rate=_ratio(summary.get("class_base_self_confident_wrong", {}).get(cls, 0), wrong),
                base_wrong_rate=_ratio(summary.get("class_base_wrong", {}).get(cls, 0), valid_count),
            )
        )
    rows.sort(key=lambda item: (-item["wrong_rate"], -item["wrong"], item["cls"]))
    return rows[:topk]


def format_report(summary, meta, class_names=None, top_pairs=30, top_classes=30, min_pair_count=1):
    final_wrong = max(int(summary.get("final_wrong_px", 0)), 1)
    base_wrong = max(int(summary.get("base_wrong_px", 0)), 1)

    lines = []
    lines.append("[ID]")
    lines.append(f"CONFIG={meta.get('config', '?')}")
    lines.append(f"CKPT={meta.get('checkpoint', '?')} ITER={meta.get('iter', '?')}")
    lines.append(
        f"EVAL n_images={meta.get('n_images', '?')}/{meta.get('n_total', '?')} "
        f"date={_datetime.date.today()} mode=whole-image-confusion"
    )
    lines.append(
        "THRESHOLDS "
        f"self_conf={meta.get('self_conf_threshold')} "
        f"margin={meta.get('margin_threshold')} "
        f"min_pair_count={min_pair_count}"
    )

    lines.append("[GLOBAL]")
    lines.append(
        f"final_wrong={summary.get('final_wrong_px', 0)} "
        f"final_GT_in_top2={_ratio(summary.get('final_wrong_top2_gt_hit_px', 0), final_wrong):.4f} "
        f"base_wrong_GT_in_top2={_ratio(summary.get('base_wrong_gt_top2_px', 0), base_wrong):.4f}"
    )
    lines.append(
        f"base_self_confident_wrong={summary.get('base_self_confident_wrong_px', 0)} "
        f"({ _ratio(summary.get('base_self_confident_wrong_px', 0), base_wrong):.4f} of base_wrong) "
        f"self_conf_GT_in_top2={_ratio(summary.get('base_self_confident_wrong_gt_top2_px', 0), max(summary.get('base_self_confident_wrong_px', 0), 1)):.4f}"
    )
    lines.append(
        f"final_wrong_same_as_base={_ratio(summary.get('final_wrong_base_same_pred_px', 0), final_wrong):.4f} "
        f"same_as_refine={_ratio(summary.get('final_wrong_refine_same_pred_px', 0), final_wrong):.4f} "
        f"all_heads_same_wrong={_ratio(summary.get('final_wrong_all_heads_same_pred_px', 0), final_wrong):.4f}"
    )
    lines.append(
        f"base_refine_same_wrong={summary.get('base_refine_same_wrong_px', 0)} "
        f"({ _ratio(summary.get('base_refine_same_wrong_px', 0), base_wrong):.4f} of base_wrong)"
    )

    lines.append("[TOP_FINAL_CONFUSIONS]")
    lines.append("rank count share GT->PRED top2_rate base_same refine_same all_same self_conf_base")
    pairs = top_confusion_pairs(
        summary,
        class_names=class_names,
        topk=top_pairs,
        min_count=min_pair_count,
    )
    if not pairs:
        lines.append("(none)")
    for rank, item in enumerate(pairs, 1):
        lines.append(
            f"{rank:02d} {item['count']} {item['share']:.4f} "
            f"{item['gt']}:{item['gt_name']} -> {item['pred']}:{item['pred_name']} "
            f"{item['top2_rate']:.3f} {item['base_same_pred_rate']:.3f} "
            f"{item['refine_same_pred_rate']:.3f} {item['all_heads_same_pred_rate']:.3f} "
            f"{item['self_conf_rate']:.3f}"
        )

    lines.append("[CLASS_FAILURES]")
    lines.append("rank cls name wrong/valid wrong_rate top2_rate base_wrong_rate self_conf_base_rate")
    classes = top_failed_classes(summary, class_names=class_names, topk=top_classes)
    if not classes:
        lines.append("(none)")
    for rank, item in enumerate(classes, 1):
        lines.append(
            f"{rank:02d} {item['cls']} {item['name']} "
            f"{item['wrong']}/{item['valid']} {item['wrong_rate']:.4f} "
            f"{item['top2_rate']:.3f} {item['base_wrong_rate']:.3f} {item['self_conf_rate']:.3f}"
        )

    lines.append("[READING]")
    lines.append("- high top2_rate means a top-k reranker has an actual candidate to recover.")
    lines.append("- high all_same/self_conf_base means confidence pseudo-prototypes are likely polluted.")
    lines.append("- concentrated GT->PRED pairs justify a confusion-aware reassessment head.")
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rank PARSeg3 semantic confusion pairs and class-level failure modes."
    )
    parser.add_argument("config", help="mmseg config path")
    parser.add_argument("checkpoint", help="checkpoint path")
    parser.add_argument("--max-images", type=int, default=250, help="number of evenly sampled val images")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", default=None, help="report path; default is next to checkpoint")
    parser.add_argument("--self-conf-threshold", type=float, default=0.90)
    parser.add_argument("--margin-threshold", type=float, default=0.50)
    parser.add_argument("--top-pairs", type=int, default=30)
    parser.add_argument("--top-classes", type=int, default=30)
    parser.add_argument("--min-pair-count", type=int, default=50)
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


def _get_class_names(dataset):
    meta = getattr(dataset, "metainfo", None) or getattr(dataset, "METAINFO", None) or {}
    names = meta.get("classes", None) or meta.get("CLASSES", None)
    if names is None:
        return None
    return list(names)


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
    class_names = _get_class_names(dataset)
    n_total = len(dataset)
    n_images = min(int(args.max_images), n_total)
    if n_images <= 0:
        raise ValueError("--max-images must select at least one image")
    indices = np.linspace(0, n_total - 1, n_images).astype(int)

    total = _empty_summary()

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

            item = summarize_confusion_arrays(
                base_logits=_to_numpy_logits(base),
                refine_logits=_to_numpy_logits(refine),
                final_logits=_to_numpy_logits(final),
                target=gt[0].detach().cpu().numpy(),
                ignore_index=ignore_index,
                self_conf_threshold=args.self_conf_threshold,
                margin_threshold=args.margin_threshold,
            )
            merge_summaries(total, item)

            if args.progress_interval > 0 and (k + 1) % args.progress_interval == 0:
                print(f"... {k + 1}/{n_images} images")

    meta = dict(
        config=os.path.splitext(os.path.basename(args.config))[0],
        checkpoint=os.path.basename(args.checkpoint),
        iter=_get_iter_meta(torch, args.checkpoint),
        n_images=n_images,
        n_total=n_total,
        self_conf_threshold=args.self_conf_threshold,
        margin_threshold=args.margin_threshold,
    )
    report = format_report(
        total,
        meta,
        class_names=class_names,
        top_pairs=args.top_pairs,
        top_classes=args.top_classes,
        min_pair_count=args.min_pair_count,
    )
    print("\n" + report)

    out = args.out or os.path.join(
        os.path.dirname(os.path.abspath(args.checkpoint)),
        f"parseg3_confusion_analysis_{os.path.splitext(os.path.basename(args.checkpoint))[0]}.txt",
    )
    with open(out, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
