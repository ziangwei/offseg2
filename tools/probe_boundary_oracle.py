# -*- coding: utf-8 -*-
"""Boundary vs interior oracle: size the high-res / image-guided route's ceiling.

Read-only frozen pass over ADE val. Splits errors by distance-to-boundary and asks
how much mIoU a PERFECT fix in each region would give:
  * boundary oracle (r=3/5/8): set pixels within r px of a GT class boundary to GT
    -> upper bound for any boundary/high-res/detail method (your NAF/UPLiFT route).
  * interior oracle: set the non-boundary wrong pixels to GT
    -> upper bound for the semantic/decision route (for contrast).
If the boundary delta is worth chasing (and clearly > what decision-side could give),
the high-res route is justified before spending a 20h run.

Usage:
  python tools/probe_boundary_oracle.py \
      local_configs/offseg2/Base/parseg3_ade20k_160k-512x512.py <base_ckpt>.pth
"""
import argparse

import torch
import torch.nn.functional as F
from mmengine.config import Config
from mmengine.runner import Runner

from mmseg.apis import init_model


def parse_args():
    p = argparse.ArgumentParser(description="Boundary vs interior oracle (read-only).")
    p.add_argument("config")
    p.add_argument("checkpoint")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-images", type=int, default=-1)
    p.add_argument("--radii", default="3,5,8")
    p.add_argument("--interior-radius", type=int, default=5)
    return p.parse_args()


def _val_loader(cfg):
    loader = dict(cfg.val_dataloader)
    loader["batch_size"] = 1
    loader["sampler"] = dict(type="DefaultSampler", shuffle=False)
    return Runner.build_dataloader(loader)


def _iu(pred, label, num_classes, ignore_index, device):
    mask = label != ignore_index
    pred = pred[mask].float(); label = label[mask].float()
    inter = pred[pred == label]
    ai = torch.histc(inter, bins=num_classes, min=0, max=num_classes - 1).to(device)
    ap = torch.histc(pred, bins=num_classes, min=0, max=num_classes - 1).to(device)
    al = torch.histc(label, bins=num_classes, min=0, max=num_classes - 1).to(device)
    return ai, ap + al - ai


def _miou(i, u):
    v = u > 0
    return float((i[v] / u[v].clamp_min(1.0)).mean().item() * 100) if bool(v.any()) else 0.0


def _boundary(gt, valid, radius):
    g = gt.clone().float(); g[~valid] = -1.0
    g = g[None, None]; k = 2 * radius + 1
    mx = F.max_pool2d(g, k, 1, radius); mn = -F.max_pool2d(-g, k, 1, radius)
    return (mx != mn).squeeze(0).squeeze(0) & valid


@torch.no_grad()
def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    model = init_model(cfg, args.checkpoint, device=args.device); model.eval()
    dev = args.device
    C = int(model.decode_head.num_classes)
    ign = int(getattr(model.decode_head, "ignore_index", 255))
    radii = [int(x) for x in args.radii.split(",") if x.strip()]
    loader = _val_loader(cfg)

    acc = {k: [torch.zeros(C, device=dev), torch.zeros(C, device=dev)] for k in
           (["base", "interior"] + [f"bnd{r}" for r in radii])}
    band_px = {r: 0 for r in radii}
    total_px = 0
    n = 0; stop = False
    for data in loader:
        if stop:
            break
        for r in model.test_step(data):
            logits = r.seg_logits.data.float().to(dev)
            gt = r.gt_sem_seg.data.squeeze(0).long().to(dev)
            if logits.shape[-2:] != gt.shape[-2:]:
                logits = F.interpolate(logits[None], size=gt.shape[-2:], mode="bilinear", align_corners=False)[0]
            pred = logits.argmax(0)
            valid = gt != ign
            total_px += int(valid.sum())

            ai, au = _iu(pred, gt, C, ign, dev); acc["base"][0] += ai; acc["base"][1] += au
            for rad in radii:
                band = _boundary(gt, valid, rad)
                band_px[rad] += int(band.sum())
                pb = torch.where(band, gt, pred)
                ai, au = _iu(pb, gt, C, ign, dev); acc[f"bnd{rad}"][0] += ai; acc[f"bnd{rad}"][1] += au
            band_i = _boundary(gt, valid, args.interior_radius)
            pi = torch.where(valid & (~band_i), gt, pred)   # fix interior (non-boundary) pixels
            ai, au = _iu(pi, gt, C, ign, dev); acc["interior"][0] += ai; acc["interior"][1] += au

            n += 1
            if n % 200 == 0:
                print(f"[probe] {n} images", flush=True)
            if args.max_images > 0 and n >= args.max_images:
                stop = True; break

    base = _miou(*acc["base"])
    print("=" * 70)
    print(f"images: {n}   baseline mIoU: {base:.2f}  (self-check ~48.2)")
    print("-" * 70)
    print("BOUNDARY oracle (fix pixels within r px of a GT boundary):")
    for rad in radii:
        m = _miou(*acc[f"bnd{rad}"])
        print(f"  r={rad}: mIoU {m:6.2f}   delta +{m - base:5.2f}   (band = {100*band_px[rad]/max(total_px,1):.1f}% of px)")
    mi = _miou(*acc["interior"])
    print(f"INTERIOR oracle (fix non-boundary wrong px, r={args.interior_radius}): mIoU {mi:6.2f}   delta +{mi - base:5.2f}")
    print("=" * 70)
    print("Readout: boundary delta = ceiling of your high-res / image-guided route.")
    print("        compare to interior delta (semantic). big boundary delta => route is worth building.")


if __name__ == "__main__":
    main()
