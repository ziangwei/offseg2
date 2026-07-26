# -*- coding: utf-8 -*-
"""Boundary SNAP oracle: split the +16.4 boundary ceiling into relocatable vs semantic.

probe_boundary_oracle.py said: fix every pixel within r px of a GT boundary and mIoU
jumps ~+16. That number is an upper bound for ANY boundary method, but it is reached
by handing the model the answer. A geometry method (attraction / offset / boundary
field, SegFix-style) cannot invent a class -- it can only RELOCATE a prediction that
already exists nearby. So the honest ceiling for that whole family is:

    for each wrong boundary-band pixel p, does the correct class gt[p] already appear
    in the INTERIOR prediction within R px of p?  If yes -> a perfect offset field
    could snap it and get p right. If no -> the model does not know that class there,
    and no amount of geometry will help.

This probe measures exactly that split:
  * base            : plain mIoU (self-check ~48.2 on parseg3 try1)
  * bnd{r}          : full boundary oracle (reference, the +16 number)
  * snap{r}_{R}     : relocation-only oracle = ceiling of the boundary-field route
  * residual        : bnd{r} - snap{r}_{R} = the part that is semantic, i.e. the axis
                      already declared dead

Read-out: if snap delta is >= ~4 mIoU the boundary-field route is worth 2-3 runs.
If it is <= ~1.5, a perfect field predictor caps the method there and the route is
not worth writing -- kill it before spending 30h/run.

Runs read-only through model.test_step, i.e. the SAME slide_inference path as eval.
Do not bypass it (that mistake cost a probe already).

Usage:
  python tools/probe_boundary_snap_oracle.py \
      local_configs/offseg2/Base/parseg3_ade20k_160k-512x512.py \
      work_dirs/parseg3_ade20k_160k-512x512_4x4_try1/iter_160000.pth
Options:
  --radii 3,5,8         GT-boundary band widths (must match probe_boundary_oracle)
  --search-radii 4,8,16 how far a perfect offset field is allowed to reach
  --max-images 500      quick pass
"""
import argparse
import itertools

import torch
import torch.nn.functional as F
from mmengine.config import Config
from mmengine.runner import Runner

from mmseg.apis import init_model


def parse_args():
    p = argparse.ArgumentParser(description="Boundary snap (relocation) oracle, read-only.")
    p.add_argument("config")
    p.add_argument("checkpoint")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-images", type=int, default=-1)
    p.add_argument("--radii", default="3,5,8", help="GT boundary band radii")
    p.add_argument("--search-radii", default="4,8,16", help="max reach of the offset field")
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
    """Same definition as probe_boundary_oracle.py so the two numbers are comparable."""
    g = gt.clone().float(); g[~valid] = -1.0
    g = g[None, None]; k = 2 * radius + 1
    mx = F.max_pool2d(g, k, 1, radius); mn = -F.max_pool2d(-g, k, 1, radius)
    return (mx != mn).squeeze(0).squeeze(0) & valid


def _dilate(mask_bool, radius):
    k = 2 * radius + 1
    return F.max_pool2d(mask_bool.float()[None, None], k, 1, radius).squeeze(0).squeeze(0) > 0.5


def _snap_reachable(pred, gt, band, interior, search_r):
    """Bool map over band pixels: gt[p] is available in the interior prediction within
    search_r px of p, so a perfect offset field could relocate it there."""
    reach = torch.zeros_like(band)
    cls = torch.unique(gt[band])
    for c in cls.tolist():
        donor = (pred == c) & interior          # where the model ALREADY says c, off-boundary
        if not bool(donor.any()):
            continue
        near = _dilate(donor, search_r)
        reach |= band & (gt == c) & near
    return reach


@torch.no_grad()
def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    model = init_model(cfg, args.checkpoint, device=args.device); model.eval()
    dev = args.device
    C = int(model.decode_head.num_classes)
    ign = int(getattr(model.decode_head, "ignore_index", 255))
    radii = [int(x) for x in args.radii.split(",") if x.strip()]
    sradii = [int(x) for x in args.search_radii.split(",") if x.strip()]
    combos = list(itertools.product(radii, sradii))
    loader = _val_loader(cfg)

    keys = ["base"] + [f"bnd{r}" for r in radii] + [f"snap{r}_{R}" for r, R in combos]
    acc = {k: [torch.zeros(C, device=dev), torch.zeros(C, device=dev)] for k in keys}
    band_px = {r: 0 for r in radii}
    band_err = {r: 0 for r in radii}                 # wrong pixels inside the band
    snap_fix = {(r, R): 0 for r, R in combos}        # of those, how many are relocatable
    total_px = 0
    n = 0; stop = False

    for data in loader:
        if stop:
            break
        for res in model.test_step(data):
            logits = res.seg_logits.data.float().to(dev)
            gt = res.gt_sem_seg.data.squeeze(0).long().to(dev)
            if logits.shape[-2:] != gt.shape[-2:]:
                logits = F.interpolate(logits[None], size=gt.shape[-2:],
                                       mode="bilinear", align_corners=False)[0]
            pred = logits.argmax(0)
            valid = gt != ign
            wrong = valid & (pred != gt)
            total_px += int(valid.sum())

            ai, au = _iu(pred, gt, C, ign, dev)
            acc["base"][0] += ai; acc["base"][1] += au

            bands = {}
            for r in radii:
                band = _boundary(gt, valid, r)
                bands[r] = band
                band_px[r] += int(band.sum())
                band_err[r] += int((band & wrong).sum())
                pb = torch.where(band, gt, pred)          # full boundary oracle
                ai, au = _iu(pb, gt, C, ign, dev)
                acc[f"bnd{r}"][0] += ai; acc[f"bnd{r}"][1] += au

            for r, R in combos:
                band = bands[r]
                interior = valid & (~band)
                reach = _snap_reachable(pred, gt, band, interior, R)
                ps = torch.where(reach, gt, pred)         # relocation-only oracle
                snap_fix[(r, R)] += int((reach & wrong).sum())
                ai, au = _iu(ps, gt, C, ign, dev)
                acc[f"snap{r}_{R}"][0] += ai; acc[f"snap{r}_{R}"][1] += au

            n += 1
            if n % 200 == 0:
                print(f"[probe] {n} images", flush=True)
            if args.max_images > 0 and n >= args.max_images:
                stop = True; break

    base = _miou(*acc["base"])
    print("=" * 78)
    print(f"images: {n}   baseline mIoU: {base:.2f}   (self-check ~48.2 on parseg3 try1)")
    print("-" * 78)
    for r in radii:
        mb = _miou(*acc[f"bnd{r}"])
        print(f"band r={r}  ({100 * band_px[r] / max(total_px, 1):5.1f}% of px, "
              f"{band_err[r]} wrong)   FULL boundary oracle: {mb:6.2f}  (+{mb - base:5.2f})")
        for R in sradii:
            ms = _miou(*acc[f"snap{r}_{R}"])
            frac = 100 * snap_fix[(r, R)] / max(band_err[r], 1)
            print(f"    search R={R:2d}   SNAP oracle {ms:6.2f}  (+{ms - base:5.2f})"
                  f"   relocatable {frac:5.1f}% of band errors"
                  f"   semantic residual +{mb - ms:5.2f}")
    print("-" * 78)
    print("SNAP delta is the ceiling of any boundary/offset/attraction-field method.")
    print("The semantic residual is the decision axis and is NOT reachable this way.")
    print("=" * 78)


if __name__ == "__main__":
    main()
