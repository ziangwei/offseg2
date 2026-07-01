# -*- coding: utf-8 -*-
"""CGR probe: confusion-group regional re-decision without training.

This is a go/no-go probe for the CGR idea before writing a new decode head.
It freezes a PARSeg3 checkpoint, builds class prototypes from TRAIN GT pixels
in the refinement feature space, then evaluates whether pure feature geometry
can re-decide known confusion groups on VAL.

Key constraints:
  * prototype fitting uses GT labels + decoder features only;
  * base logits are used only to route pixels into a small confusion group;
  * re-decision compares only classes inside that group;
  * optional feature-affinity smoothing makes the decision region-consistent.

Example:
  python tools/probe_cgr_redecision.py \
      local_configs/offseg2/Base/parseg3_ade20k_160k-512x512.py \
      work_dirs/parseg3_ade20k_160k-512x512_4x4_try1/iter_160000.pth \
      --fit-images 2000 --eval-images 2000 --num-prototypes 3
"""

import argparse
import copy
import json
import os
from collections import defaultdict, namedtuple
from typing import Dict, Iterable, List, Sequence


ResolvedGroup = namedtuple("ResolvedGroup", ["name", "class_ids", "class_names", "missing"])


_ALIASES = {
    "windowpane": ["windowpane", "window", "windows"],
    "wall": ["wall", "wall-other", "walls"],
    "floor": ["floor", "floor-other"],
    "sidewalk": ["sidewalk", "pavement"],
    "sofa": ["sofa", "couch"],
    "armchair": ["armchair", "chair armchair"],
}


def default_confusion_groups():
    """Groups from the current PARSeg3 diagnostic notes."""
    return [
        dict(
            name="wall_family",
            classes=["wall", "ceiling", "door", "windowpane", "cabinet", "mirror", "curtain"],
        ),
        dict(name="building_tree", classes=["building", "tree"]),
        dict(name="road_sidewalk", classes=["road", "sidewalk"]),
        dict(name="rug_floor", classes=["rug", "floor"]),
        dict(name="armchair_sofa", classes=["armchair", "sofa"]),
    ]


def _norm_name(name):
    return str(name).strip().lower().replace("_", " ").replace("-", " ")


def _lookup_names(canonical):
    names = [canonical]
    names.extend(_ALIASES.get(canonical, []))
    out = []
    seen = set()
    for name in names:
        key = _norm_name(name)
        if key not in seen:
            out.append(key)
            seen.add(key)
    return out


def resolve_confusion_groups(groups, classes, min_size=2):
    """Resolve class-name groups to dataset class ids.

    Missing classes are recorded for reporting, but a group is kept when at
    least ``min_size`` classes resolve. This keeps the script usable across ADE,
    Cityscapes, and COCO-Stuff variants.
    """
    class_lookup = {_norm_name(name): idx for idx, name in enumerate(classes)}
    resolved = []
    for spec in groups:
        ids = []
        names = []
        missing = []
        for canonical in spec["classes"]:
            found = None
            for key in _lookup_names(canonical):
                if key in class_lookup:
                    found = class_lookup[key]
                    break
            if found is None:
                missing.append(canonical)
                continue
            if found not in ids:
                ids.append(found)
                names.append(classes[found])
        if len(ids) >= min_size:
            order = sorted(range(len(ids)), key=lambda i: ids[i])
            resolved.append(
                ResolvedGroup(
                    name=spec["name"],
                    class_ids=[ids[i] for i in order],
                    class_names=[names[i] for i in order],
                    missing=missing,
                )
            )
    return resolved


def _load_group_specs(path):
    if not path:
        return default_confusion_groups()
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        return [dict(name=name, classes=classes) for name, classes in payload.items()]
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description="Training-free CGR confusion-group re-decision probe.")
    parser.add_argument("config", help="mmseg config path")
    parser.add_argument("checkpoint", help="frozen PARSeg checkpoint")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fit-split", choices=["train", "val"], default="train")
    parser.add_argument("--eval-split", choices=["val", "train"], default="val")
    parser.add_argument("--fit-images", type=int, default=2000)
    parser.add_argument("--eval-images", type=int, default=2000)
    parser.add_argument(
        "--probe-size",
        default=None,
        help="fixed H,W or single integer for whole-image probe forward; default uses config crop_size",
    )
    parser.add_argument("--groups-json", default=None, help="optional {group: [class,...]} JSON")
    parser.add_argument("--num-prototypes", type=int, default=3)
    parser.add_argument("--max-samples-per-class", type=int, default=8192)
    parser.add_argument("--per-image-class-samples", type=int, default=128)
    parser.add_argument("--kmeans-iters", type=int, default=12)
    parser.add_argument("--trigger-topk", type=int, default=2, help="route if base top-k intersects a group")
    parser.add_argument("--min-proto-margin", type=float, default=0.0)
    parser.add_argument("--smooth-iters", type=int, default=1)
    parser.add_argument("--smooth-sigma", type=float, default=0.15)
    parser.add_argument("--prototype-temp", type=float, default=0.07)
    parser.add_argument("--prototype-cache", default=None, help="optional torch cache for fitted prototypes")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--progress-interval", type=int, default=200)
    parser.add_argument("--out", default=None, help="optional report path")
    return parser.parse_args()


def _set_innermost_pipeline(dataset_cfg, pipeline):
    node = dataset_cfg
    while isinstance(node, dict) and "dataset" in node:
        node = node["dataset"]
    if isinstance(node, dict):
        node["pipeline"] = pipeline


def _parse_probe_size(text):
    if text is None or text == "":
        return None
    parts = [int(x) for x in str(text).replace("x", ",").split(",") if x.strip()]
    if len(parts) == 1:
        return (parts[0], parts[0])
    if len(parts) == 2:
        return (parts[0], parts[1])
    raise ValueError(f"invalid --probe-size: {text!r}")


def _probe_size_from_cfg(cfg, requested):
    parsed = _parse_probe_size(requested)
    if parsed is not None:
        return parsed
    if "crop_size" in cfg:
        return tuple(int(x) for x in cfg.crop_size)
    data_preprocessor = cfg.get("model", {}).get("data_preprocessor", {})
    if "size" in data_preprocessor:
        return tuple(int(x) for x in data_preprocessor["size"])
    return (512, 512)


def _innermost_dataset_cfg(dataset_cfg):
    node = dataset_cfg
    while isinstance(node, dict) and "dataset" in node:
        node = node["dataset"]
    return node


def _fixed_probe_pipeline(cfg, probe_size):
    """Fixed-size pipeline for EfficientFormer-safe single forward."""
    ann = dict(type="LoadAnnotations")
    val_ds = _innermost_dataset_cfg(cfg.val_dataloader["dataset"])
    for step in val_ds.get("pipeline", []):
        if isinstance(step, dict) and step.get("type") == "LoadAnnotations":
            ann = copy.deepcopy(step)
            break
    return [
        dict(type="LoadImageFromFile"),
        dict(type="Resize", scale=tuple(probe_size), keep_ratio=False),
        ann,
        dict(type="PackSegInputs"),
    ]


def _loader_from_cfg(cfg, split, probe_size):
    """One-image-at-a-time loader using fixed-size eval transforms.

    Train split deliberately uses eval-style fixed transforms. This preserves
    train/val image separation while avoiding variable crop sizes, the batch
    size assertion from earlier active-class probes, and EfficientFormer ASUB
    reshape failures from ratio-resized whole images.
    """
    from mmengine.runner import Runner

    if split == "val":
        loader = copy.deepcopy(cfg.val_dataloader)
        loader["batch_size"] = 1
        loader["sampler"] = dict(type="DefaultSampler", shuffle=False)
        _set_innermost_pipeline(loader["dataset"], _fixed_probe_pipeline(cfg, probe_size))
        return Runner.build_dataloader(loader)

    loader = copy.deepcopy(cfg.train_dataloader)
    loader["batch_size"] = 1
    loader["sampler"] = dict(type="DefaultSampler", shuffle=False)
    _set_innermost_pipeline(loader["dataset"], _fixed_probe_pipeline(cfg, probe_size))
    return Runner.build_dataloader(loader)


def _register_refinement_hook(model, store):
    head = model.decode_head
    target = None
    target_name = "decode_head.align"
    if hasattr(head, "prototype_attribute_refinement"):
        module = head.prototype_attribute_refinement
        target = getattr(module, "refinement_feat_proj", None)
        target_name = "decode_head.prototype_attribute_refinement.refinement_feat_proj"
    if target is None:
        target = getattr(head, "align", None)
        target_name = "decode_head.align"
    if target is None:
        raise AttributeError("decode_head has neither refinement_feat_proj nor align to hook")

    def _hook(_module, _inputs, output):
        store["refinement_feats"] = output.detach()

    return target.register_forward_hook(_hook), target_name


def _forward_once(model, data, store):
    data = model.data_preprocessor(data, False)
    inputs = data["inputs"]
    feats = model.extract_feat(inputs)
    outputs = model.decode_head.forward(list(feats))
    if not isinstance(outputs, dict) or "final_logits" not in outputs:
        raise KeyError("decode_head.forward must return a dict with 'final_logits'")
    if "refinement_feats" not in store:
        raise KeyError("refinement feature hook did not fire")
    return outputs["final_logits"], store["refinement_feats"], data["data_samples"]


def _gt_at_feature_size(gt, feat_hw):
    import torch.nn.functional as F

    return F.interpolate(gt[None, None].float(), size=feat_hw, mode="nearest")[0, 0].long()


def _select_evenly(indices, limit):
    import torch

    if indices.numel() <= limit:
        return indices
    pos = torch.linspace(0, indices.numel() - 1, steps=limit, device=indices.device).long()
    return indices[pos]


def _collect_prototype_samples(
    model,
    dataloader,
    groups,
    max_images,
    device,
    max_samples_per_class,
    per_image_class_samples,
    progress_interval,
    store,
):
    import torch
    import torch.nn.functional as F

    wanted = sorted({class_id for group in groups for class_id in group.class_ids})
    prototype_banks = {class_id: [] for class_id in wanted}
    counts = defaultdict(int)
    seen = 0

    with torch.no_grad():
        for data in dataloader:
            store.clear()
            _final_logits, feats, samples = _forward_once(model, data, store)
            feats = F.normalize(feats.float(), p=2, dim=1, eps=1e-6)

            for batch_idx, sample in enumerate(samples):
                feat = feats[batch_idx]
                gt = sample.gt_sem_seg.data.squeeze(0).long().to(device)
                gt_f = _gt_at_feature_size(gt, feat.shape[-2:]).reshape(-1)
                feat_flat = feat.permute(1, 2, 0).reshape(-1, feat.shape[0])

                for class_id in wanted:
                    if counts[class_id] >= max_samples_per_class:
                        continue
                    where = (gt_f == class_id).nonzero(as_tuple=True)[0]
                    if where.numel() == 0:
                        continue
                    remain = max_samples_per_class - counts[class_id]
                    take = min(int(per_image_class_samples), int(remain), int(where.numel()))
                    picked = _select_evenly(where, take)
                    prototype_banks[class_id].append(feat_flat[picked].cpu())
                    counts[class_id] += int(picked.numel())

            seen += 1
            if progress_interval > 0 and seen % progress_interval == 0:
                filled = sum(1 for class_id in wanted if counts[class_id] >= max_samples_per_class)
                print(f"[fit] {seen} images, capped classes {filled}/{len(wanted)}", flush=True)
            if max_images > 0 and seen >= max_images:
                break

    merged = {}
    for class_id, chunks in prototype_banks.items():
        if chunks:
            merged[class_id] = torch.cat(chunks, dim=0)
        else:
            merged[class_id] = torch.empty(0, 0)
    return merged, dict(counts), seen


def run_kmeans(samples, num_prototypes, iters, device):
    import torch
    import torch.nn.functional as F

    if samples.numel() == 0:
        return samples
    x = F.normalize(samples.float().to(device), p=2, dim=1, eps=1e-6)
    k = min(max(1, int(num_prototypes)), x.shape[0])
    init_idx = torch.linspace(0, x.shape[0] - 1, steps=k, device=x.device).long()
    centers = x[init_idx].clone()
    for _ in range(max(1, int(iters))):
        assign = torch.matmul(x, centers.t()).argmax(dim=1)
        new_centers = []
        for j in range(k):
            mask = assign == j
            if bool(mask.any()):
                new_centers.append(x[mask].mean(dim=0))
            else:
                new_centers.append(centers[j])
        centers = F.normalize(torch.stack(new_centers, dim=0), p=2, dim=1, eps=1e-6)
    return centers.cpu()


def _fit_prototypes(model, cfg, args, groups, store):
    import torch

    if args.prototype_cache and os.path.exists(args.prototype_cache) and not args.rebuild_cache:
        payload = torch.load(args.prototype_cache, map_location="cpu")
        print(f"[fit] loaded prototype cache: {args.prototype_cache}", flush=True)
        return payload["prototypes"], payload.get("counts", {}), payload.get("fit_images", 0)

    loader = _loader_from_cfg(cfg, args.fit_split, args.probe_size)
    prototype_banks, counts, fit_images = _collect_prototype_samples(
        model=model,
        dataloader=loader,
        groups=groups,
        max_images=args.fit_images,
        device=args.device,
        max_samples_per_class=args.max_samples_per_class,
        per_image_class_samples=args.per_image_class_samples,
        progress_interval=args.progress_interval,
        store=store,
    )
    prototypes = {}
    for class_id, samples in prototype_banks.items():
        if samples.numel() == 0:
            continue
        prototypes[class_id] = run_kmeans(samples, args.num_prototypes, args.kmeans_iters, args.device)

    if args.prototype_cache:
        os.makedirs(os.path.dirname(os.path.abspath(args.prototype_cache)), exist_ok=True)
        torch.save(dict(prototypes=prototypes, counts=counts, fit_images=fit_images), args.prototype_cache)
        print(f"[fit] saved prototype cache: {args.prototype_cache}", flush=True)
    return prototypes, counts, fit_images


def feature_affinity_smooth(group_logits, feats, iters=1, sigma=0.15):
    """Smooth class scores over 4-neighbors weighted by feature cosine affinity."""
    import torch

    if iters <= 0:
        return group_logits
    sigma = max(float(sigma), 1e-6)
    logits = group_logits
    feat = feats
    shifts = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for _ in range(int(iters)):
        acc = logits.clone()
        denom = torch.ones((1, logits.shape[-2], logits.shape[-1]), device=logits.device, dtype=logits.dtype)
        for dy, dx in shifts:
            if dy < 0:
                yc = slice(0, dy)
                yn = slice(-dy, None)
            elif dy > 0:
                yc = slice(dy, None)
                yn = slice(0, -dy)
            else:
                yc = slice(None)
                yn = slice(None)

            if dx < 0:
                xc = slice(0, dx)
                xn = slice(-dx, None)
            elif dx > 0:
                xc = slice(dx, None)
                xn = slice(0, -dx)
            else:
                xc = slice(None)
                xn = slice(None)

            center = feat[:, yc, xc]
            neigh = feat[:, yn, xn]
            weight = torch.exp(((center * neigh).sum(dim=0, keepdim=True) - 1.0) / sigma).to(logits.dtype)
            acc[:, yc, xc] = acc[:, yc, xc] + logits[:, yn, xn] * weight
            denom[:, yc, xc] = denom[:, yc, xc] + weight
        logits = acc / denom.clamp_min(1e-6)
    return logits


def _group_redecision(final_logits, refinement_feat, groups, prototypes, args):
    import torch
    import torch.nn.functional as F

    logits = final_logits.float()
    feat = F.normalize(refinement_feat.float(), p=2, dim=0, eps=1e-6)
    pred = logits.argmax(dim=0)
    new_pred = pred.clone()
    replace_mask = torch.zeros_like(pred, dtype=torch.bool)

    topk = min(max(1, int(args.trigger_topk)), logits.shape[0])
    topk_idx = logits.topk(k=topk, dim=0).indices
    flat_feat = feat.reshape(feat.shape[0], -1).t()
    temp = max(float(args.prototype_temp), 1e-6)

    group_changed = {}
    for group in groups:
        class_ids = [class_id for class_id in group.class_ids if class_id in prototypes]
        if len(class_ids) < 2:
            continue
        ids = torch.tensor(class_ids, device=logits.device, dtype=torch.long)
        routed = (topk_idx[..., None] == ids.view(1, 1, 1, -1)).any(dim=0).any(dim=-1)
        if not bool(routed.any()):
            group_changed[group.name] = 0
            continue

        scores = []
        for class_id in class_ids:
            proto = F.normalize(prototypes[class_id].to(logits.device).float(), p=2, dim=1, eps=1e-6)
            sim = torch.matmul(flat_feat, proto.t()).max(dim=1).values
            scores.append(sim.view_as(pred) / temp)
        group_scores = torch.stack(scores, dim=0)
        group_scores = feature_affinity_smooth(group_scores, feat, args.smooth_iters, args.smooth_sigma)
        best = group_scores.argmax(dim=0)
        candidate = torch.tensor(class_ids, device=logits.device, dtype=torch.long)[best]

        if len(class_ids) > 1 and args.min_proto_margin > 0:
            vals = group_scores.topk(k=2, dim=0).values
            margin_ok = (vals[0] - vals[1]) >= float(args.min_proto_margin)
        else:
            margin_ok = torch.ones_like(routed)

        apply = routed & margin_ok
        new_pred[apply] = candidate[apply]
        replace_mask |= apply
        group_changed[group.name] = int(apply.sum().item())
    return new_pred, replace_mask, group_changed


def _resize_logits_to_gt(logits, gt_hw):
    import torch.nn.functional as F

    if tuple(logits.shape[-2:]) == tuple(gt_hw):
        return logits
    return F.interpolate(logits[None], size=gt_hw, mode="bilinear", align_corners=False)[0]


def _upsample_label(label, target_hw):
    import torch.nn.functional as F

    return F.interpolate(label[None, None].float(), size=target_hw, mode="nearest")[0, 0].long()


def _upsample_mask(mask, target_hw):
    import torch.nn.functional as F

    return F.interpolate(mask[None, None].float(), size=target_hw, mode="nearest")[0, 0] > 0.5


def _intersect_union(pred, label, num_classes, ignore_index):
    import torch

    valid = label != ignore_index
    pred = pred[valid].long()
    label = label[valid].long()
    same = pred == label
    inter = torch.bincount(label[same], minlength=num_classes)[:num_classes].float()
    pred_area = torch.bincount(pred.clamp(0, num_classes), minlength=num_classes + 1)[:num_classes].float()
    label_area = torch.bincount(label, minlength=num_classes)[:num_classes].float()
    return inter, pred_area + label_area - inter


def _miou(inter, union):
    valid = union > 0
    if not bool(valid.any()):
        return 0.0
    return float((inter[valid] / union[valid].clamp_min(1.0)).mean().item() * 100.0)


def _eval_cgr(model, cfg, args, groups, prototypes, num_classes, ignore_index, store):
    import torch

    loader = _loader_from_cfg(cfg, args.eval_split, args.probe_size)
    device = args.device
    base_inter = torch.zeros(num_classes, device=device)
    base_union = torch.zeros(num_classes, device=device)
    cgr_inter = torch.zeros(num_classes, device=device)
    cgr_union = torch.zeros(num_classes, device=device)
    changed_px = 0
    valid_px = 0
    corrected_px = 0
    damaged_px = 0
    group_changed = defaultdict(int)
    seen = 0

    with torch.no_grad():
        for data in loader:
            store.clear()
            final_logits, feats, samples = _forward_once(model, data, store)
            for batch_idx, sample in enumerate(samples):
                gt = sample.gt_sem_seg.data.squeeze(0).long().to(device)
                target_hw = tuple(gt.shape[-2:])
                full_logits = _resize_logits_to_gt(final_logits[batch_idx], target_hw)
                base_pred = full_logits.argmax(dim=0)

                small_pred, small_mask, small_group_changed = _group_redecision(
                    final_logits[batch_idx], feats[batch_idx], groups, prototypes, args
                )
                redecision = _upsample_label(small_pred, target_hw)
                mask = _upsample_mask(small_mask, target_hw)
                cgr_pred = base_pred.clone()
                cgr_pred[mask] = redecision[mask]

                bi, bu = _intersect_union(base_pred, gt, num_classes, ignore_index)
                ci, cu = _intersect_union(cgr_pred, gt, num_classes, ignore_index)
                base_inter += bi
                base_union += bu
                cgr_inter += ci
                cgr_union += cu

                valid = gt != ignore_index
                changed = (base_pred != cgr_pred) & valid
                corrected = (base_pred != gt) & (cgr_pred == gt) & valid
                damaged = (base_pred == gt) & (cgr_pred != gt) & valid
                changed_px += int(changed.sum().item())
                valid_px += int(valid.sum().item())
                corrected_px += int(corrected.sum().item())
                damaged_px += int(damaged.sum().item())
                for name, value in small_group_changed.items():
                    group_changed[name] += int(value)

                seen += 1
                if args.progress_interval > 0 and seen % args.progress_interval == 0:
                    print(f"[eval] {seen} images", flush=True)
                if args.eval_images > 0 and seen >= args.eval_images:
                    return dict(
                        images=seen,
                        base_inter=base_inter,
                        base_union=base_union,
                        cgr_inter=cgr_inter,
                        cgr_union=cgr_union,
                        changed_px=changed_px,
                        valid_px=valid_px,
                        corrected_px=corrected_px,
                        damaged_px=damaged_px,
                        group_changed=dict(group_changed),
                    )

    return dict(
        images=seen,
        base_inter=base_inter,
        base_union=base_union,
        cgr_inter=cgr_inter,
        cgr_union=cgr_union,
        changed_px=changed_px,
        valid_px=valid_px,
        corrected_px=corrected_px,
        damaged_px=damaged_px,
        group_changed=dict(group_changed),
    )


def _format_report(args, groups, hook_name, counts, fit_images, prototypes, stats):
    base_miou = _miou(stats["base_inter"], stats["base_union"])
    cgr_miou = _miou(stats["cgr_inter"], stats["cgr_union"])
    delta = cgr_miou - base_miou
    changed_rate = 100.0 * stats["changed_px"] / max(stats["valid_px"], 1)
    ratio = stats["corrected_px"] / max(stats["damaged_px"], 1)

    lines = []
    lines.append("[CGR TRAINING-FREE RE-DECISION PROBE]")
    lines.append(f"config={os.path.basename(args.config)}")
    lines.append(f"checkpoint={os.path.basename(args.checkpoint)}")
    lines.append(f"feature_hook={hook_name}")
    lines.append(
        f"fit={args.fit_split}:{fit_images} eval={args.eval_split}:{stats['images']} "
        f"probe_size={args.probe_size[0]}x{args.probe_size[1]} "
        f"K={args.num_prototypes} trigger_topk={args.trigger_topk} "
        f"smooth={args.smooth_iters} margin={args.min_proto_margin:g}"
    )
    lines.append("")
    lines.append("[groups]")
    for group in groups:
        proto_info = []
        for class_id, name in zip(group.class_ids, group.class_names):
            proto_info.append(f"{class_id}:{name}(n={counts.get(class_id, 0)},p={len(prototypes.get(class_id, []))})")
        miss = f" missing={group.missing}" if group.missing else ""
        lines.append(f"- {group.name}: " + ", ".join(proto_info) + miss)
    lines.append("")
    lines.append(f"baseline_mIoU : {base_miou:.4f}")
    lines.append(f"cgr_mIoU      : {cgr_miou:.4f}")
    lines.append(f"delta         : {delta:+.4f}")
    lines.append(f"changed_px    : {stats['changed_px']} / {stats['valid_px']} ({changed_rate:.3f}%)")
    lines.append(f"correction    : {stats['corrected_px']}")
    lines.append(f"damage        : {stats['damaged_px']}")
    lines.append(f"corr/damage   : {ratio:.3f}")
    if stats["group_changed"]:
        lines.append("")
        lines.append("[changed feature-pixels by group]")
        for name, value in sorted(stats["group_changed"].items()):
            lines.append(f"- {name}: {value}")
    lines.append("")
    lines.append("[read]")
    if delta >= 0.20:
        lines.append("CGR has enough direct geometry signal to justify writing a trainable head.")
    elif delta >= 0.05 and ratio > 1.0:
        lines.append("CGR is positive but small; next step should be a very light gate, not a heavy decoder.")
    else:
        lines.append("CGR does not yet beat the base enough; inspect group damage before writing a full model.")
    return "\n".join(lines)


def main():
    args = parse_args()

    from mmengine.config import Config
    from mmseg.apis import init_model

    cfg = Config.fromfile(args.config)
    args.probe_size = _probe_size_from_cfg(cfg, args.probe_size)
    model = init_model(cfg, args.checkpoint, device=args.device)
    model.eval()

    num_classes = int(model.decode_head.num_classes)
    ignore_index = int(getattr(model.decode_head, "ignore_index", 255))
    meta = getattr(model, "dataset_meta", None) or {}
    classes = list(meta.get("classes", [])) or [str(i) for i in range(num_classes)]

    specs = _load_group_specs(args.groups_json)
    groups = resolve_confusion_groups(specs, classes)
    if not groups:
        raise RuntimeError("No confusion group resolved to at least two dataset classes.")

    store = {}
    hook, hook_name = _register_refinement_hook(model, store)
    print(f"[hook] {hook_name}", flush=True)
    print("[groups] " + "; ".join(f"{g.name}={g.class_ids}" for g in groups), flush=True)

    try:
        prototypes, counts, fit_images = _fit_prototypes(model, cfg, args, groups, store)
        missing_proto = sorted({c for g in groups for c in g.class_ids if c not in prototypes})
        if missing_proto:
            print(f"[warn] classes with no prototype samples: {missing_proto}", flush=True)
        stats = _eval_cgr(model, cfg, args, groups, prototypes, num_classes, ignore_index, store)
    finally:
        hook.remove()

    report = _format_report(args, groups, hook_name, counts, fit_images, prototypes, stats)
    print("\n" + report)

    out = args.out or os.path.join(
        os.path.dirname(os.path.abspath(args.checkpoint)),
        f"cgr_redecision_probe_{os.path.splitext(os.path.basename(args.checkpoint))[0]}.txt",
    )
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
