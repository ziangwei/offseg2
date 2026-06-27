# -*- coding: utf-8 -*-
"""Probe a learnable active-class prior without training a segmentation head.

This script freezes a trained segmentation checkpoint, trains a tiny class-wise
presence predictor from image-level logit statistics, then sweeps gating rules
on dense logits. It answers whether the +10 mIoU active-class oracle has a
learnable, cheap part worth turning into a real subsystem.

It evaluates three gate families in one pass and ranks them together:
  * HARD  (learned head): threshold / min-classes / penalty   (original)
  * SOFT  (learned head): add alpha*log p_presence to logits  -- recall-safe,
           this is what the real subsystem would actually do
  * SELF  (zero training): top-k by the base's OWN max-prob    -- a floor that
           shows how much the *learned* head adds over base self-signal
Then it prints the ceiling-capture fraction = best realizable delta / oracle.

Example:
  python tools/probe_active_class_predictor.py \
      local_configs/offseg2/Base/parseg3_ade20k_160k-512x512.py \
      work_dirs/parseg3_ade20k_160k-512x512_4x4_try1/iter_160000.pth \
      --fit-images 2000 --eval-images 2000
"""

import argparse
import copy
import math
from typing import Iterable, List, Sequence, Tuple


def _flatten_nested(values):
    for value in values:
        if isinstance(value, (list, tuple)):
            yield from _flatten_nested(value)
        else:
            yield value


def _shape_hw(logits):
    return len(logits[0]), len(logits[0][0])


def parse_float_list(text: str) -> List[float]:
    out = []
    for raw in str(text).split(","):
        raw = raw.strip()
        if not raw:
            continue
        out.append(float("inf") if raw.lower() == "inf" else float(raw))
    return out


def parse_int_list(text: str) -> List[int]:
    return [int(x) for x in str(text).split(",") if x.strip()]


def probe_dataloader_cfg(loader_cfg):
    """Return a dataloader config that feeds one image at a time to test_step."""
    cfg = copy.deepcopy(loader_cfg)
    cfg["batch_size"] = 1
    return cfg


def presence_from_label(label, num_classes: int, ignore_index: int = 255) -> List[float]:
    present = [0.0] * num_classes
    for cls in _flatten_nested(label):
        cls = int(cls)
        if cls != ignore_index and 0 <= cls < num_classes:
            present[cls] = 1.0
    return present


def select_active_classes(scores: Sequence[float], threshold: float, min_classes: int) -> List[bool]:
    scores = [float(x) for x in scores]
    active = [score >= threshold for score in scores]
    if min_classes > 0:
        k = min(int(min_classes), len(scores))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        for idx in order[:k]:
            active[idx] = True
    return active


def argmax_logits(logits) -> List[List[int]]:
    c = len(logits)
    h, w = _shape_hw(logits)
    pred = [[0 for _ in range(w)] for _ in range(h)]
    for y in range(h):
        for x in range(w):
            best_cls = 0
            best_val = float(logits[0][y][x])
            for cls in range(1, c):
                val = float(logits[cls][y][x])
                if val > best_val:
                    best_cls = cls
                    best_val = val
            pred[y][x] = best_cls
    return pred


def predict_with_active_gate(
    logits,
    scores: Sequence[float],
    threshold: float,
    min_classes: int,
    penalty: float,
) -> List[List[int]]:
    active = select_active_classes(scores, threshold, min_classes)
    c = len(logits)
    h, w = _shape_hw(logits)
    pred = [[0 for _ in range(w)] for _ in range(h)]
    for y in range(h):
        for x in range(w):
            best_cls = 0
            best_val = -float("inf")
            for cls in range(c):
                val = float(logits[cls][y][x])
                if not active[cls]:
                    val = -float("inf") if math.isinf(penalty) else val - penalty
                if val > best_val:
                    best_cls = cls
                    best_val = val
            pred[y][x] = best_cls
    return pred


def intersect_union_np(pred, label, num_classes: int, ignore_index: int = 255) -> Tuple[List[int], List[int]]:
    inter = [0] * num_classes
    pred_area = [0] * num_classes
    label_area = [0] * num_classes
    for p, g in zip(_flatten_nested(pred), _flatten_nested(label)):
        p = int(p)
        g = int(g)
        if g == ignore_index:
            continue
        if 0 <= p < num_classes:
            pred_area[p] += 1
        if 0 <= g < num_classes:
            label_area[g] += 1
        if p == g and 0 <= g < num_classes:
            inter[g] += 1
    union = [pred_area[i] + label_area[i] - inter[i] for i in range(num_classes)]
    return inter, union


def mean_iou_np(inter: Sequence[int], union: Sequence[int]) -> float:
    vals = [i / max(u, 1) for i, u in zip(inter, union) if u > 0]
    return sum(vals) / len(vals) if vals else 0.0


def _add_hist(dst, src):
    for i, value in enumerate(src):
        dst[i] += int(value)


def _candidate_name(threshold, min_classes, penalty):
    p = "inf" if math.isinf(penalty) else f"{penalty:g}"
    return f"hard t={threshold:g},k={min_classes},p={p}"


def parse_args():
    parser = argparse.ArgumentParser(description="Learnable active-class prior probe.")
    parser.add_argument("config", help="mmseg config path")
    parser.add_argument("checkpoint", help="checkpoint path for the frozen floor model")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fit-split", choices=["train", "val"], default="train")
    parser.add_argument("--fit-images", type=int, default=2000)
    parser.add_argument("--eval-images", type=int, default=-1)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--topk-frac", type=float, default=0.01)
    parser.add_argument("--thresholds", default="0.03,0.05,0.1,0.2,0.3,0.5")
    parser.add_argument("--min-classes", default="8,12,16,24,32")
    parser.add_argument("--penalties", default="2,5,10,inf")
    parser.add_argument("--soft-alphas", default="0.5,1,2,4",
                        help="soft log-prior gate: add alpha*log p_presence to logits (recall-safe)")
    parser.add_argument("--self-topks", default="8,12,16,24",
                        help="zero-training reference: top-k classes by base's own max-prob")
    parser.add_argument("--progress-interval", type=int, default=200)
    parser.add_argument("--out", default=None, help="optional text file for the summary")
    return parser.parse_args()


def _set_innermost_pipeline(dataset_cfg, pipeline):
    """Set .pipeline on the innermost dataset (handles wrappers like RepeatDataset)."""
    node = dataset_cfg
    while isinstance(node, dict) and "dataset" in node:
        node = node["dataset"]
    if isinstance(node, dict):
        node["pipeline"] = pipeline


def _fit_loader_from_cfg(cfg, split):
    """One-image-at-a-time loader.

    For 'train' we load TRAIN images through the TEST pipeline on purpose: train
    aug (RandomResize 0.5-2.0 + RandomCrop cat_max_ratio) produces sub-512,
    non-square crops, and the efficientformer backbone cannot run those through
    slide-inference (an attention reshape blows up). The test pipeline only
    ratio-resizes, so every image is backbone-safe -- the same path the oracle
    probe ran cleanly on. Train/val separation is preserved (train *images*).
    """
    from mmengine.runner import Runner

    if split == "val":
        loader = probe_dataloader_cfg(cfg.val_dataloader)
        loader["sampler"] = dict(type="DefaultSampler", shuffle=False)
        return Runner.build_dataloader(loader)

    loader = probe_dataloader_cfg(cfg.train_dataloader)
    loader["sampler"] = dict(type="DefaultSampler", shuffle=False)  # train uses InfiniteSampler
    _set_innermost_pipeline(loader["dataset"], copy.deepcopy(cfg.val_dataloader["dataset"]["pipeline"]))
    return Runner.build_dataloader(loader)


def _resize_logits_to_gt(logits, gt):
    import torch.nn.functional as F

    if logits.shape[-2:] != gt.shape[-2:]:
        logits = F.interpolate(logits[None], size=gt.shape[-2:], mode="bilinear", align_corners=False)[0]
    return logits


def _presence_from_label_torch(gt, num_classes, ignore_index):
    import torch

    present = torch.zeros(num_classes, dtype=torch.float32)
    cls = torch.unique(gt[gt != ignore_index]).cpu().long()
    cls = cls[(cls >= 0) & (cls < num_classes)]
    if cls.numel() > 0:
        present[cls] = 1.0
    return present


def _logit_presence_features(logits, topk_frac):
    import torch

    logits = logits.float()
    probs = torch.softmax(logits, dim=0)
    c, h, w = probs.shape
    hw = h * w
    k = max(1, int(hw * topk_frac))
    prob_flat = probs.flatten(1)
    logit_flat = logits.flatten(1)
    pred = logits.argmax(0).flatten()
    pred_area = torch.bincount(pred, minlength=c).float() / max(hw, 1)
    feats = torch.stack(
        [
            prob_flat.max(dim=1).values,
            prob_flat.mean(dim=1),
            prob_flat.topk(k, dim=1).values.mean(dim=1),
            logit_flat.max(dim=1).values,
            pred_area,
        ],
        dim=1,
    )
    return feats.cpu()


def _self_score(logits):
    """Base's own image-level presence signal: per-class max softmax over pixels."""
    import torch

    return torch.softmax(logits.float(), dim=0).flatten(1).max(dim=1).values


def _collect_presence_data(model, dataloader, max_images, device, num_classes, ignore_index, topk_frac, name, progress):
    import torch

    feats = []
    labels = []
    n = 0
    with torch.no_grad():
        for data in dataloader:
            results = model.test_step(data)
            for result in results:
                logits = result.seg_logits.data.float().to(device)
                gt = result.gt_sem_seg.data.squeeze(0).long().to(device)
                logits = _resize_logits_to_gt(logits, gt)
                feats.append(_logit_presence_features(logits, topk_frac))
                labels.append(_presence_from_label_torch(gt, num_classes, ignore_index))
                n += 1
                if progress > 0 and n % progress == 0:
                    print(f"[{name}] collected {n} images", flush=True)
                if max_images > 0 and n >= max_images:
                    return torch.stack(feats), torch.stack(labels)
    return torch.stack(feats), torch.stack(labels)


def _train_presence_head(x, y, epochs, lr, weight_decay, batch_size, device):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class ClasswiseLinearPresence(nn.Module):
        def __init__(self, num_classes, feat_dim):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(num_classes, feat_dim))
            self.bias = nn.Parameter(torch.zeros(num_classes))

        def forward(self, feats):
            return (feats * self.weight.unsqueeze(0)).sum(dim=-1) + self.bias.unsqueeze(0)

    x = x.to(device)
    y = y.to(device)
    mean = x.mean(dim=(0, 1), keepdim=True)
    std = x.std(dim=(0, 1), keepdim=True).clamp_min(1e-6)
    x = (x - mean) / std

    model = ClasswiseLinearPresence(x.shape[1], x.shape[2]).to(device)
    pos = y.sum(dim=0)
    neg = y.shape[0] - pos
    pos_weight = (neg / pos.clamp_min(1.0)).clamp(1.0, 50.0)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(epochs):
        order = torch.randperm(x.shape[0], device=device)
        total = 0.0
        seen = 0
        for start in range(0, x.shape[0], batch_size):
            idx = order[start:start + batch_size]
            logits = model(x[idx])
            loss = F.binary_cross_entropy_with_logits(logits, y[idx], pos_weight=pos_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * int(idx.numel())
            seen += int(idx.numel())
        if (epoch + 1) in {1, epochs} or (epoch + 1) % 20 == 0:
            print(f"[fit] epoch {epoch + 1:03d}/{epochs} loss={total / max(seen, 1):.4f}", flush=True)

    return model, mean, std


def _active_mask_torch(scores, threshold, min_classes):
    import torch

    mask = scores >= threshold
    if min_classes > 0:
        k = min(int(min_classes), scores.numel())
        top_idx = torch.topk(scores, k=k).indices
        mask[top_idx] = True
    if not bool(mask.any()):
        mask[torch.argmax(scores)] = True
    return mask


def _topk_mask_torch(scores, k):
    import torch

    mask = torch.zeros_like(scores, dtype=torch.bool)
    k = min(int(k), scores.numel())
    mask[torch.topk(scores, k=k).indices] = True
    return mask


def _intersect_union_torch(pred, label, num_classes, ignore_index, device):
    import torch

    mask = label != ignore_index
    pred = pred[mask].float()
    label = label[mask].float()
    inter = pred[pred == label]
    ai = torch.histc(inter, bins=num_classes, min=0, max=num_classes - 1).to(device)
    ap = torch.histc(pred, bins=num_classes, min=0, max=num_classes - 1).to(device)
    al = torch.histc(label, bins=num_classes, min=0, max=num_classes - 1).to(device)
    return ai, ap + al - ai


def _miou_torch(inter, union):
    valid = union > 0
    if not bool(valid.any()):
        return 0.0
    return float((inter[valid] / union[valid].clamp_min(1.0)).mean().item() * 100)


def _new_iu(num_classes, device):
    import torch

    return {"inter": torch.zeros(num_classes, device=device), "union": torch.zeros(num_classes, device=device),
            "tp": 0, "fp": 0, "fn": 0, "pruned": 0}


def _eval_sweep(
    model,
    dataloader,
    presence_head,
    mean,
    std,
    args,
    num_classes,
    ignore_index,
    thresholds,
    min_classes_list,
    penalties,
    soft_alphas,
    self_topks,
):
    import torch

    device = args.device
    hard_cands = [(t, k, p) for t in thresholds for k in min_classes_list for p in penalties]
    hard_stats = {c: _new_iu(num_classes, device) for c in hard_cands}
    soft_stats = {a: _new_iu(num_classes, device) for a in soft_alphas}
    self_stats = {k: _new_iu(num_classes, device) for k in self_topks}

    base_i, base_u = torch.zeros(num_classes, device=device), torch.zeros(num_classes, device=device)
    oracle_i, oracle_u = torch.zeros(num_classes, device=device), torch.zeros(num_classes, device=device)
    head = {"tp": 0, "fp": 0, "fn": 0}  # learned head quality @0.5 (macro)

    n = 0
    with torch.no_grad():
        for data in dataloader:
            results = model.test_step(data)
            for result in results:
                logits = result.seg_logits.data.float().to(device)
                gt = result.gt_sem_seg.data.squeeze(0).long().to(device)
                logits = _resize_logits_to_gt(logits, gt)
                present = _presence_from_label_torch(gt, num_classes, ignore_index).to(device).bool()

                feat = _logit_presence_features(logits, args.topk_frac).unsqueeze(0).to(device)
                pres_logit = presence_head((feat - mean.to(device)) / std.to(device)).squeeze(0)
                probs = torch.sigmoid(pres_logit)
                logp = torch.log(probs.clamp_min(1e-6))
                self_sc = _self_score(logits)

                # base
                ai, au = _intersect_union_torch(logits.argmax(0), gt, num_classes, ignore_index, device)
                base_i += ai; base_u += au

                # GT-active oracle (upper bound)
                gated = logits.clone(); gated[~present] = -float("inf")
                ai, au = _intersect_union_torch(gated.argmax(0), gt, num_classes, ignore_index, device)
                oracle_i += ai; oracle_u += au

                # learned-head quality @0.5
                act05 = probs >= 0.5
                head["tp"] += int((act05 & present).sum()); head["fp"] += int((act05 & ~present).sum())
                head["fn"] += int((~act05 & present).sum())

                # HARD (learned head): threshold / min-classes / penalty
                for cand in hard_cands:
                    threshold, min_classes, penalty = cand
                    active = _active_mask_torch(probs, threshold, min_classes)
                    g = logits.clone()
                    if math.isinf(penalty):
                        g[~active] = -float("inf")
                    else:
                        g[~active] = g[~active] - float(penalty)
                    ai, au = _intersect_union_torch(g.argmax(0), gt, num_classes, ignore_index, device)
                    st = hard_stats[cand]
                    st["inter"] += ai; st["union"] += au
                    st["tp"] += int((active & present).sum()); st["fp"] += int((active & ~present).sum())
                    st["fn"] += int((~active & present).sum()); st["pruned"] += int((~active).sum())

                # SOFT (learned head): add alpha*log p to logits -- recall-safe, the real mechanism
                for alpha in soft_alphas:
                    g = logits + (alpha * logp).view(-1, 1, 1)
                    ai, au = _intersect_union_torch(g.argmax(0), gt, num_classes, ignore_index, device)
                    soft_stats[alpha]["inter"] += ai; soft_stats[alpha]["union"] += au

                # SELF (zero training): top-k by base's own max-prob
                for k in self_topks:
                    active = _topk_mask_torch(self_sc, k)
                    g = logits.clone(); g[~active] = -float("inf")
                    ai, au = _intersect_union_torch(g.argmax(0), gt, num_classes, ignore_index, device)
                    st = self_stats[k]
                    st["inter"] += ai; st["union"] += au
                    st["tp"] += int((active & present).sum()); st["fp"] += int((active & ~present).sum())
                    st["fn"] += int((~active & present).sum()); st["pruned"] += int((~active).sum())

                n += 1
                if args.progress_interval > 0 and n % args.progress_interval == 0:
                    print(f"[eval] {n} images", flush=True)
                if args.eval_images > 0 and n >= args.eval_images:
                    return n, base_i, base_u, oracle_i, oracle_u, hard_stats, soft_stats, self_stats, head
    return n, base_i, base_u, oracle_i, oracle_u, hard_stats, soft_stats, self_stats, head


def _prf(st, n):
    tp, fp, fn = st["tp"], st["fp"], st["fn"]
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    return prec, rec, f1, st["pruned"] / max(n, 1)


def main():
    args = parse_args()

    import torch
    from mmengine.config import Config
    from mmseg.apis import init_model

    cfg = Config.fromfile(args.config)
    model = init_model(cfg, args.checkpoint, device=args.device)
    model.eval()

    num_classes = int(model.decode_head.num_classes)
    ignore_index = int(getattr(model.decode_head, "ignore_index", 255))
    thresholds = parse_float_list(args.thresholds)
    min_classes_list = parse_int_list(args.min_classes)
    penalties = parse_float_list(args.penalties)
    soft_alphas = parse_float_list(args.soft_alphas)
    self_topks = parse_int_list(args.self_topks)

    fit_loader = _fit_loader_from_cfg(cfg, args.fit_split)
    print(f"[fit] split={args.fit_split} images={args.fit_images} (train imgs via TEST pipeline)", flush=True)
    x_fit, y_fit = _collect_presence_data(
        model, fit_loader, args.fit_images, args.device,
        num_classes, ignore_index, args.topk_frac, "fit", args.progress_interval,
    )
    presence_head, mean, std = _train_presence_head(
        x_fit, y_fit, args.epochs, args.lr, args.weight_decay, args.batch_size, args.device,
    )
    presence_head.eval()

    val_loader = _fit_loader_from_cfg(cfg, "val")
    n, base_i, base_u, oracle_i, oracle_u, hard_stats, soft_stats, self_stats, head = _eval_sweep(
        model, val_loader, presence_head, mean, std, args,
        num_classes, ignore_index, thresholds, min_classes_list, penalties, soft_alphas, self_topks,
    )
    base_miou = _miou_torch(base_i, base_u)
    oracle_miou = _miou_torch(oracle_i, oracle_u)

    # rows: (miou, name, prec_or_None, rec_or_None, f1_or_None, pruned_per_img)
    rows = []
    for cand, st in hard_stats.items():
        prec, rec, f1, pruned = _prf(st, n)
        rows.append((_miou_torch(st["inter"], st["union"]), _candidate_name(*cand), prec, rec, f1, pruned))
    for alpha, st in soft_stats.items():
        rows.append((_miou_torch(st["inter"], st["union"]), f"soft a={alpha:g}", None, None, None, 0.0))
    for k, st in self_stats.items():
        prec, rec, f1, pruned = _prf(st, n)
        rows.append((_miou_torch(st["inter"], st["union"]), f"self topk={k}", prec, rec, f1, pruned))
    rows.sort(key=lambda item: item[0], reverse=True)

    oracle_delta = oracle_miou - base_miou
    best_delta = rows[0][0] - base_miou if rows else 0.0
    capture = (best_delta / oracle_delta * 100) if oracle_delta > 1e-6 else 0.0
    h_prec = head["tp"] / max(head["tp"] + head["fp"], 1)
    h_rec = head["tp"] / max(head["tp"] + head["fn"], 1)
    h_f1 = 2 * h_prec * h_rec / max(h_prec + h_rec, 1e-12)

    def fmt(v):
        return "   -  " if v is None else f"{v:>6.3f}"

    lines = []
    lines.append("=" * 92)
    lines.append(f"fit images           : {x_fit.shape[0]} ({args.fit_split} imgs, test pipeline)")
    lines.append(f"eval images          : {n}")
    lines.append(f"baseline mIoU        : {base_miou:.2f}   (self-check: should be ~48.2)")
    lines.append(f"GT active oracle     : {oracle_miou:.2f}  (ceiling delta = +{oracle_delta:.2f})")
    lines.append(f"presence head @0.5   : prec={h_prec:.3f} recall={h_rec:.3f} f1={h_f1:.3f}")
    lines.append("-" * 92)
    lines.append(f"{'rank':>4} {'gate':<22} {'mIoU':>7} {'delta':>8} {'prec':>7} {'recall':>7} {'f1':>7} {'pruned/img':>11}")
    for rank, (miou, name, prec, rec, f1, pruned) in enumerate(rows[:24], start=1):
        lines.append(
            f"{rank:>4} {name:<22} {miou:>7.2f} {miou - base_miou:>+8.2f} "
            f"{fmt(prec)} {fmt(rec)} {fmt(f1)} {pruned:>11.1f}"
        )
    lines.append("=" * 92)
    lines.append(f"BEST realizable      : +{best_delta:.2f}  = {capture:.0f}% of the +{oracle_delta:.2f} oracle ceiling")
    lines.append("Readout: 师兄 PAL+AGCF 整套才 ~1pt。best realizable 若明显 > 1 且占 ceiling 可观比例 → 这条轴")
    lines.append("        (图像条件可出现类先验) 够同量级,值得搭成子系统。soft 通常 > hard;若 soft≈self,说明")
    lines.append("        学出来的头没比 base 自带信号多带信息,要换更强的 presence 特征/输入。")

    summary = "\n".join(lines)
    print(summary)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(summary + "\n")


if __name__ == "__main__":
    main()
