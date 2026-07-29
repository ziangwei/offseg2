# -*- coding: utf-8 -*-
"""Spatial-prior probe: measure THREE candidate decision-side terms at once,
training-free, on existing checkpoints. Predictions are PRE-REGISTERED in this
header BEFORE the first run; the read-out rules are fixed. No post-hoc story.

Context. The CCM factorial closed the "re-measure the same evidence" family:
capacity +0.08, depth -0.61, pooled scene -0.34 around the 46.8 gen-1 point.
The one unfilled cell in the mechanism map is spatial extent of class
hypotheses (SSA-Seg SPPA / CGRSeg RCM direction). Before building any head,
this probe asks whether the axis carries signal AT ALL, by adding to the
final logits, per image, per class c at pixel (i,j):

    gpos * ( log r_c[i] + log m_c[j] )    position structure: row/col marginal
                                          distributions of the model's own
                                          posterior mass for class c
    gmass * log pi_c                      mass prior: class c's share of the
                                          image's posterior mass

and sweeping (gpos, gmass) on a grid, all combos evaluated in ONE pass over
val (logits cached per image, 35 cheap argmaxes each).

THREE hypotheses, one probe:
  H-pos  (gpos > 0 carries):  per-image position structure helps -> BUILD the
         v1 head (~150 params: zero-init per-class gamma on the log-prior).
  H-mass (gmass > 0 carries): presence prior in disguise -- adjacent to the
         DEAD active-class axis (learned presence realised +0.03). Weak claim;
         do NOT build a decoder chapter on it.
  H-cal  (gmass < 0 carries): macro-IoU logit adjustment (rare classes get
         boosted). Inference-side calibration row for the thesis, zero params,
         NOT a decoder chapter.

PRE-REGISTERED predictions (written 2026-07-28, before any number):
  1. Deltas will be SMALL. Best combo on ccm2t1 (46.88): +0.1 ~ +0.4.
     If best < +0.15 -> the whole spatial/prior axis on this base is DEAD;
     the fallback line is a true second decision path + decorrelation, and
     no amount of gamma tuning reopens this cell.
  2. gpos alone: positive but modest; its gains concentrate in BOUNDARY band
     and spatially incoherent FP speckle, NOT interior present-conf mass.
  3. gmass > 0: near zero or NEGATIVE for mIoU (macro metric punishes
     suppressing rare true classes) -- despite absent-FP being 42.6% of
     errors. If this prediction is wrong and gmass>0 gives > +0.3, that is
     evidence the absent-FP pool is reachable by a trivial prior and every
     heavier mechanism (incl. LCR's 78%-absent gain) is overbuilt.
  4. Rich-get-richer check: position prior must NOT reduce absent-FP much
     (normalisation erases total mass; an FP blob supports itself). If
     absent-FP DOES drop under gpos alone, my mechanism reasoning is wrong
     even if mIoU moves -- flag it.
  5. On PARSeg3 try1 (48.17) the best delta will be SMALLER than on ccm2t1
     (stronger model, less incoherence to clean up).

Read-out table per checkpoint: mIoU for the full grid; for the best combo,
error decomposition (absent-FP vs present-conf; boundary r=5 vs interior) and
top-10 per-class IoU deltas.

Usage (1 GPU, ~40-60 min per checkpoint, 2000 val images):
  python tools/probe_spatial_prior.py \
      local_configs/offseg2/Base/offsegccm2t1_ade20k_160k-512x512.py \
      work_dirs/offsegccm2t1_ade20k_160k-512x512_4x4_try1/iter_160000.pth
  python tools/probe_spatial_prior.py \
      local_configs/offseg2/Base/parseg3_ade20k_160k-512x512.py \
      work_dirs/parseg3_ade20k_160k-512x512_4x4_try1/iter_160000.pth
Options: --max-images 500 for a quick pass; --gpos / --gmass to override grids.
Goes through model.test_step (slide inference preserved). Profiles here are
computed on the STITCHED full-image posterior; the deployed head would use
per-crop profiles -- if anything the probe overestimates, which only matters
if it reads out near the +0.15 threshold (disclosed).
"""
import argparse
import itertools

import torch
import torch.nn.functional as F
from mmengine.config import Config
from mmengine.runner import Runner

from mmseg.apis import init_model

EPS = 1e-4


def parse_args():
    p = argparse.ArgumentParser(description="Three-hypothesis spatial prior probe.")
    p.add_argument("config")
    p.add_argument("checkpoint")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-images", type=int, default=-1)
    p.add_argument("--gpos", default="0,0.1,0.2,0.4,0.8")
    p.add_argument("--gmass", default="-0.4,-0.2,-0.1,0,0.1,0.2,0.4")
    p.add_argument("--boundary-radius", type=int, default=5)
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
    gpos_list = [float(x) for x in args.gpos.split(",")]
    gmass_list = [float(x) for x in args.gmass.split(",")]
    combos = list(itertools.product(gpos_list, gmass_list))
    loader = _val_loader(cfg)

    acc = {c: [torch.zeros(C, device=dev), torch.zeros(C, device=dev)] for c in combos}
    # decomposition counters for every combo (cheap): [absent_fp, present_wrong,
    # boundary_wrong, interior_wrong]
    dec = {c: torch.zeros(4, device=dev) for c in combos}
    n = 0; stop = False

    for data in loader:
        if stop:
            break
        for r in model.test_step(data):
            logits = r.seg_logits.data.float().to(dev)            # [K,H,W]
            gt = r.gt_sem_seg.data.squeeze(0).long().to(dev)
            if logits.shape[-2:] != gt.shape[-2:]:
                logits = F.interpolate(logits[None], size=gt.shape[-2:],
                                       mode="bilinear", align_corners=False)[0]
            valid = gt != ign
            K, H, W = logits.shape
            p = torch.softmax(logits, dim=0)                      # [K,H,W]

            mass = p.sum(dim=(1, 2))                              # [K]
            pi = (mass / mass.sum().clamp_min(1e-6)).clamp_min(EPS)
            row = p.sum(dim=2); row = row / row.sum(1, keepdim=True).clamp_min(1e-6)
            col = p.sum(dim=1); col = col / col.sum(1, keepdim=True).clamp_min(1e-6)
            b_pos = (torch.log(row.clamp_min(EPS))[:, :, None]
                     + torch.log(col.clamp_min(EPS))[:, None, :])  # [K,H,W]
            b_mass = torch.log(pi)[:, None, None]                  # [K,1,1]

            present = torch.zeros(C, dtype=torch.bool, device=dev)
            present[torch.unique(gt[valid])] = True
            band = _boundary(gt, valid, args.boundary_radius)

            for c in combos:
                gp, gm = c
                pred = (logits + gp * b_pos + gm * b_mass).argmax(0)
                ai, au = _iu(pred, gt, C, ign, dev)
                acc[c][0] += ai; acc[c][1] += au
                wrong = valid & (pred != gt)
                absent_fp = wrong & (~present[pred])
                dec[c] += torch.stack([
                    absent_fp.sum(), (wrong & present[pred]).sum(),
                    (wrong & band).sum(), (wrong & ~band).sum()]).float()

            n += 1
            if n % 200 == 0:
                print(f"[probe] {n} images", flush=True)
            if args.max_images > 0 and n >= args.max_images:
                stop = True; break

    base = _miou(*acc[(0.0, 0.0)])
    print("=" * 78)
    print(f"images: {n}   baseline (gpos=0, gmass=0) mIoU: {base:.2f}")
    print("-" * 78)
    print(f"{'gpos':>6} {'gmass':>6} {'mIoU':>7} {'delta':>7}   "
          f"{'absentFP':>9} {'presentW':>9} {'bndW':>9} {'intW':>9}")
    best = None
    for gp in gpos_list:
        for gm in gmass_list:
            m = _miou(*acc[(gp, gm)])
            d = dec[(gp, gm)]
            print(f"{gp:>6.2f} {gm:>6.2f} {m:>7.2f} {m - base:>+7.2f}   "
                  f"{int(d[0]):>9d} {int(d[1]):>9d} {int(d[2]):>9d} {int(d[3]):>9d}")
            if best is None or m > best[0]:
                best = (m, gp, gm)
        print()
    m, gp, gm = best
    print("-" * 78)
    print(f"BEST: mIoU {m:.2f} (delta {m - base:+.2f}) at gpos={gp}, gmass={gm}")
    print("Pre-registered read-out rules (see header): best<+0.15 -> axis DEAD;")
    print("gpos carries -> build v1 head; gmass>0 carries -> presence prior in")
    print("disguise (no decoder chapter); gmass<0 carries -> macro calibration row.")
    # top per-class deltas at best combo
    ib, ub = acc[(0.0, 0.0)]; im, um = acc[(gp, gm)]
    ioub = ib / ub.clamp_min(1.0); ioum = im / um.clamp_min(1.0)
    delta = (ioum - ioub) * 100
    vals, idx = torch.sort(delta, descending=True)
    top = [(int(i), float(v)) for v, i in zip(vals[:10], idx[:10])]
    bot = [(int(i), float(v)) for v, i in zip(vals[-10:], idx[-10:])]
    print(f"top +: {top}")
    print(f"top -: {bot}")


if __name__ == "__main__":
    main()
