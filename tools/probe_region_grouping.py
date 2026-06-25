# -*- coding: utf-8 -*-
"""Region-grouping oracle probe (NO training, pure evaluation).

The question this answers, before we build any region-assignment head:

    If we stop deciding per-pixel and instead group pixels into regions and give
    each region ONE label, how high could mIoU go -- and which grouping SIGNAL
    gives the most headroom?

For every candidate grouping signal we:
  1. partition each image into regions,
  2. relabel each region with its GT-majority class  (an ORACLE upper bound),
  3. measure mIoU, mean region purity, and region count on a val subset.

Reference rows to compare against:
  - pixel-final : the model's current per-pixel mIoU on this subset      (FLOOR)
  - pred-cc     : connected components of the model prediction
                  = gain from merely making predictions region-consistent
  - slic-K      : appearance superpixels (class-agnostic grouping)
  - feat-K      : k-means on decoder features (feat_aligned)

How to read it:
  * If slic/feat oracle mIoU clearly BEATS pred-cc at a similar region count,
    that signal separates classes the model MERGES -> a region-assignment head
    built on it has real headroom. Pick the winner.
  * If pred-cc is already >> floor but slic/feat ~= pred-cc, the gain is just
    region-consistency (a cheap majority-vote / CRF post-process may get it).
  * If nothing beats the floor, region assignment has no headroom -> drop it.

NOTE: inference here is a single whole-image forward (like the analyze_* tools),
not slide inference, so 'pixel-final' won't equal the reported slide mIoU. All
rows use the SAME forward, so the RELATIVE comparison is what matters.

Pure-numpy helpers are importable without torch; the CLI lazily imports
torch / mmseg / skimage / sklearn and skips any signal whose dependency is
missing (it tells you what to pip install).

Example:
  python tools/probe_region_grouping.py \
      local_configs/offseg2/Base/parseg3_ade20k_160k-512x512.py \
      work_dirs/parseg3_ade20k_160k-512x512_4x4_try2/iter_160000.pth \
      --max-images 250 --slic 64,256 --feat 64,256
"""

import argparse
import os
from collections import defaultdict

import numpy as np


# --------------------------- pure-numpy core ---------------------------------

def oracle_relabel_flat(region_flat, gt_flat, num_classes, ignore_index=255):
    """Relabel each region by its GT-majority class.

    Returns (pred_flat, valid_px, purity_num, n_regions). Regions that cover no
    valid GT pixel are set to ignore_index (they do not affect mIoU).
    """
    region_flat = np.asarray(region_flat).reshape(-1)
    gt_flat = np.asarray(gt_flat).reshape(-1).astype(np.int64)
    valid = gt_flat != ignore_index

    _, ridx = np.unique(region_flat, return_inverse=True)
    n_regions = int(ridx.max()) + 1 if ridx.size else 0

    hist = np.zeros((max(n_regions, 1), num_classes), dtype=np.int64)
    np.add.at(hist, (ridx[valid], gt_flat[valid]), 1)
    region_label = hist.argmax(axis=1)
    has_valid = hist.sum(axis=1) > 0

    pred = region_label[ridx].astype(np.int64)
    pred[~has_valid[ridx]] = ignore_index

    purity_num = int((pred[valid] == gt_flat[valid]).sum())
    return pred, int(valid.sum()), purity_num, n_regions


def accumulate_iou(pred_flat, gt_flat, num_classes, inter, union, ignore_index=255):
    """Add one image to running per-class intersection / union arrays."""
    gt_flat = np.asarray(gt_flat).reshape(-1).astype(np.int64)
    pred_flat = np.asarray(pred_flat).reshape(-1).astype(np.int64)
    valid = gt_flat != ignore_index
    p = pred_flat[valid]
    g = gt_flat[valid]

    inter += np.bincount(g[p == g], minlength=num_classes)[:num_classes]
    pred_area = np.bincount(p, minlength=num_classes + 1)[:num_classes]  # pred==ignore dropped
    gt_area = np.bincount(g, minlength=num_classes)[:num_classes]
    union += pred_area + gt_area - np.bincount(g[p == g], minlength=num_classes)[:num_classes]


def mean_iou(inter, union):
    present = union > 0
    if not present.any():
        return 0.0
    return float((inter[present] / np.maximum(union[present], 1)).mean())


# ------------------------------- CLI -----------------------------------------

def _parse_int_list(text):
    return [int(x) for x in str(text).split(",") if x.strip()]


def parse_args():
    p = argparse.ArgumentParser(description="Oracle probe for region-grouping signals (no training).")
    p.add_argument("config", help="mmseg config path")
    p.add_argument("checkpoint", help="checkpoint path (probe the FLOOR model, e.g. parseg3)")
    p.add_argument("--max-images", type=int, default=250)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--slic", default="64,256", help="comma list of superpixel counts, '' to disable")
    p.add_argument("--feat", default="64,256", help="comma list of feature k-means clusters, '' to disable")
    p.add_argument("--out", default=None)
    p.add_argument("--progress-interval", type=int, default=25)
    return p.parse_args()


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


def _denorm_image_hwc(inputs, data_preprocessor):
    """Best-effort recovery of an RGB image in [0,1] for SLIC."""
    img = inputs[0].detach().float()
    mean = getattr(data_preprocessor, "mean", None)
    std = getattr(data_preprocessor, "std", None)
    if mean is not None and std is not None:
        img = img * std.to(img.device).view(-1, 1, 1) + mean.to(img.device).view(-1, 1, 1)
        img = (img / 255.0)
    else:
        img = img - img.amin()
        img = img / img.amax().clamp_min(1e-6)
    img = img.clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
    return img


def main():
    args = parse_args()

    import torch
    import torch.nn.functional as F
    from mmengine.config import Config
    from mmengine.dataset import pseudo_collate
    from mmengine.registry import init_default_scope
    from mmseg.apis import init_model
    from mmseg.registry import DATASETS

    slic_ks = _parse_int_list(args.slic)
    feat_ks = _parse_int_list(args.feat)

    # optional deps
    try:
        from skimage.segmentation import slic as _slic
        from skimage.measure import label as _cc_label
        have_skimage = True
    except Exception as exc:
        have_skimage = False
        print(f"[warn] scikit-image missing ({exc}); pred-cc and slic-* skipped. pip install scikit-image")

    _kmeans = None
    if feat_ks:
        try:
            from sklearn.cluster import MiniBatchKMeans as _kmeans
        except Exception as exc:
            print(f"[warn] scikit-learn missing ({exc}); feat-* skipped. pip install scikit-learn")

    cfg = Config.fromfile(args.config)
    init_default_scope("mmseg")
    model = init_model(args.config, args.checkpoint, device=args.device)
    model.eval()
    num_classes = int(model.decode_head.num_classes)
    ignore_index = int(getattr(model.decode_head, "ignore_index", 255))

    # capture feat_aligned via a hook on the decode head's `align` module
    feat_store = {}
    hook_handle = None
    if feat_ks and _kmeans is not None:
        align_mod = getattr(model.decode_head, "align", None)
        if align_mod is not None:
            hook_handle = align_mod.register_forward_hook(
                lambda m, i, o: feat_store.__setitem__("feat", o.detach())
            )
        else:
            print("[warn] decode_head has no .align; feat-* skipped")
            feat_ks = []

    ds_cfg = cfg.val_dataloader.dataset if "val_dataloader" in cfg else cfg.test_dataloader.dataset
    dataset = DATASETS.build(ds_cfg)
    n_total = len(dataset)
    n_images = min(int(args.max_images), n_total)
    indices = np.linspace(0, n_total - 1, n_images).astype(int)

    # signal -> running stats
    signals = ["pixel-final"]
    if have_skimage:
        signals.append("pred-cc")
        signals += [f"slic-{k}" for k in slic_ks]
    if feat_ks and _kmeans is not None:
        signals += [f"feat-{k}" for k in feat_ks]

    inter = {s: np.zeros(num_classes, dtype=np.int64) for s in signals}
    union = {s: np.zeros(num_classes, dtype=np.int64) for s in signals}
    purity_num = defaultdict(int)
    purity_den = defaultdict(int)
    region_count = defaultdict(int)
    region_imgs = defaultdict(int)

    with torch.no_grad():
        for k, idx in enumerate(indices):
            data = pseudo_collate([dataset[int(idx)]])
            inputs, gt = _prepare_inputs_and_gt(data, model, args.device, ignore_index)
            H, W = gt.shape[-2:]

            feats = model.extract_feat(inputs)
            outputs = model.decode_head.forward(list(feats))
            if "final_logits" not in outputs:
                raise KeyError("decode_head output has no 'final_logits'")

            final = outputs["final_logits"]
            if tuple(final.shape[-2:]) != (H, W):
                final = F.interpolate(final, size=(H, W), mode="bilinear", align_corners=False)
            pred = final[0].argmax(0).cpu().numpy().astype(np.int64)
            gt_np = gt[0].cpu().numpy().astype(np.int64)
            gt_flat = gt_np.reshape(-1)

            # pixel-final (floor): the prediction itself, no grouping
            accumulate_iou(pred.reshape(-1), gt_flat, num_classes, inter["pixel-final"], union["pixel-final"], ignore_index)

            region_maps = {}
            if have_skimage:
                region_maps["pred-cc"] = _cc_label(pred, connectivity=1)
                img_hwc = _denorm_image_hwc(inputs, model.data_preprocessor)
                for kk in slic_ks:
                    region_maps[f"slic-{kk}"] = _slic(
                        img_hwc, n_segments=kk, compactness=10.0, start_label=0, channel_axis=-1
                    )
            if feat_ks and _kmeans is not None and "feat" in feat_store:
                feat = feat_store["feat"][0]                     # [C, h, w]
                c, h, w = feat.shape
                X = feat.reshape(c, h * w).t().cpu().numpy()
                for kk in feat_ks:
                    km = _kmeans(n_clusters=min(kk, h * w), n_init=3, max_iter=50, batch_size=4096)
                    lab = km.fit_predict(X).reshape(h, w)
                    lab_up = F.interpolate(
                        torch.from_numpy(lab.astype(np.float32))[None, None], size=(H, W), mode="nearest"
                    )[0, 0].numpy().astype(np.int64)
                    region_maps[f"feat-{kk}"] = lab_up

            for s, rmap in region_maps.items():
                pred_o, vpx, pnum, nreg = oracle_relabel_flat(rmap, gt_flat, num_classes, ignore_index)
                accumulate_iou(pred_o, gt_flat, num_classes, inter[s], union[s], ignore_index)
                purity_num[s] += pnum
                purity_den[s] += vpx
                region_count[s] += nreg
                region_imgs[s] += 1

            if args.progress_interval > 0 and (k + 1) % args.progress_interval == 0:
                print(f"... {k + 1}/{n_images} images")

    if hook_handle is not None:
        hook_handle.remove()

    # ---- report ----
    lines = []
    lines.append("[REGION-GROUPING ORACLE PROBE]")
    lines.append(f"config={os.path.basename(args.config)} ckpt={os.path.basename(args.checkpoint)}")
    lines.append(f"n_images={n_images}/{n_total} num_classes={num_classes} (single whole-image forward)")
    lines.append("")
    lines.append(f"{'signal':<12} {'regions/img':>11} {'purity':>8} {'oracle_mIoU':>12}")
    floor = mean_iou(inter["pixel-final"], union["pixel-final"])
    lines.append(f"{'pixel-final':<12} {'-':>11} {'-':>8} {floor:>12.4f}   <- FLOOR (no grouping)")
    for s in signals:
        if s == "pixel-final":
            continue
        miou = mean_iou(inter[s], union[s])
        pur = purity_num[s] / max(purity_den[s], 1)
        reg = region_count[s] / max(region_imgs[s], 1)
        lines.append(f"{s:<12} {reg:>11.1f} {pur:>8.4f} {miou:>12.4f}")

    # interpretation
    lines.append("")
    lines.append("[READ]")
    miou_of = lambda s: mean_iou(inter[s], union[s]) if s in inter else None
    predcc = miou_of("pred-cc")
    best_app = None
    for s in signals:
        if s.startswith("slic-") or s.startswith("feat-"):
            m = miou_of(s)
            if best_app is None or m > best_app[1]:
                best_app = (s, m)
    if predcc is not None:
        lines.append(f"- pred-cc oracle {predcc:.4f} vs floor {floor:.4f}: gain from pure region-consistency = {predcc - floor:+.4f}")
    if best_app is not None and predcc is not None:
        delta = best_app[1] - predcc
        lines.append(f"- best appearance/feature signal {best_app[0]} oracle {best_app[1]:.4f}: beats pred-cc by {delta:+.4f}")
        if delta >= 0.03:
            lines.append("  => this signal separates classes the model merges; a region-assignment head on it has real headroom.")
        elif (predcc - floor) >= 0.03:
            lines.append("  => gain is mostly region-consistency, not better grouping; try cheap majority-vote/CRF before a new head.")
        else:
            lines.append("  => little headroom anywhere; region-assignment route is probably not worth it.")

    report = "\n".join(lines)
    print("\n" + report)

    out = args.out or os.path.join(
        os.path.dirname(os.path.abspath(args.checkpoint)),
        f"region_grouping_probe_{os.path.splitext(os.path.basename(args.checkpoint))[0]}.txt",
    )
    with open(out, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
