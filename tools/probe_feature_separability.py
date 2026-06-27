# -*- coding: utf-8 -*-
"""#3b de-risk: are the confusable pairs linearly separable in the FROZEN feature?

Hooks PARSeg3's shared decoder feature (`self.align` output, 256-d at stride-4)
and asks, for the top confusable class pairs, whether a simple linear probe can
separate A vs B in that frozen feature space.

  * HIGH separability (A-vs-B acc >> chance) -> the feature already distinguishes
    them; the base just draws the boundary wrong -> an end-to-end margin/separation
    loss can fix it (lighter (b)).
  * LOW separability (~chance) -> the frozen feature entangles them -> you must
    change the representation end-to-end (heavier (b), the real contribution).

Runs single-forward on 512x512 crops (backbone-safe, no slide). Read-only w.r.t.
the checkpoint; trains only tiny linear probes on cached features.

Usage (server, single GPU):
  python tools/probe_feature_separability.py \
      local_configs/offseg2/Base/parseg3_ade20k_160k-512x512.py \
      <base_ckpt>.pth --pair-images 400 --feat-images 600 --topn 12
"""
import argparse
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.config import Config
from mmengine.runner import Runner

from mmseg.apis import init_model


def parse_args():
    p = argparse.ArgumentParser(description="Frozen-feature pairwise separability probe.")
    p.add_argument("config")
    p.add_argument("checkpoint")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--pair-images", type=int, default=400, help="imgs to find top confusable pairs")
    p.add_argument("--feat-images", type=int, default=600, help="imgs to cache features from")
    p.add_argument("--topn", type=int, default=12)
    p.add_argument("--per-class-cap", type=int, default=4000)
    p.add_argument("--probe-steps", type=int, default=300)
    return p.parse_args()


def _loader_512(cfg):
    pipeline = [
        dict(type="LoadImageFromFile"),
        dict(type="Resize", scale=(512, 512), keep_ratio=False),
        dict(type="LoadAnnotations", reduce_zero_label=True),
        dict(type="PackSegInputs"),
    ]
    loader = copy.deepcopy(cfg.val_dataloader)
    loader["batch_size"] = 1
    loader["sampler"] = dict(type="DefaultSampler", shuffle=False)
    ds = loader["dataset"]
    while isinstance(ds, dict) and "dataset" in ds:
        ds = ds["dataset"]
    ds["pipeline"] = pipeline
    return Runner.build_dataloader(loader)


def _gt_at(feat, gt):
    return F.interpolate(gt[None, None].float(), size=feat.shape[-2:], mode="nearest")[0, 0].long()


def _linear_probe_acc(xa, xb, steps, device):
    x = torch.cat([xa, xb], 0)
    y = torch.cat([torch.zeros(len(xa)), torch.ones(len(xb))]).to(device)
    mu, sd = x.mean(0, keepdim=True), x.std(0, keepdim=True).clamp_min(1e-6)
    x = (x - mu) / sd
    perm = torch.randperm(len(x), device=device)
    x, y = x[perm], y[perm]
    n_tr = int(0.7 * len(x))
    xtr, ytr, xte, yte = x[:n_tr], y[:n_tr], x[n_tr:], y[n_tr:]
    w = nn.Linear(x.shape[1], 1).to(device)
    opt = torch.optim.Adam(w.parameters(), lr=0.05)
    pw = torch.tensor([len(xa) / max(len(xb), 1)], device=device)
    for _ in range(steps):
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(w(xtr).squeeze(1), ytr, pos_weight=pw)
        loss.backward(); opt.step()
    with torch.no_grad():
        pred = (torch.sigmoid(w(xte).squeeze(1)) >= 0.5).float()
        acc = float((pred == yte).float().mean())
    chance = max(len(xa), len(xb)) / (len(xa) + len(xb))
    return acc, chance


@torch.no_grad()
def _collect(model, loader, store, max_images, device, want_classes, cap, C, ignore_index, mode):
    """mode='pairs' -> accumulate present-conf confusion; mode='feats' -> cache features for want_classes."""
    conf = torch.zeros(C * C, dtype=torch.float64, device=device)
    banks = {c: [] for c in (want_classes or [])}
    counts = {c: 0 for c in (want_classes or [])}
    n = 0
    for data in loader:
        data = model.data_preprocessor(data, False)
        inputs = data["inputs"]
        feats = model.extract_feat(inputs)
        out = model.decode_head.forward(feats)
        feat = store["f"]  # (B,256,h,w)
        final = out["final_logits"] if isinstance(out, dict) else out
        for i, ds in enumerate(data["data_samples"]):
            gt = ds.gt_sem_seg.data.squeeze(0).long().to(device)
            fmap = feat[i]
            gt_f = _gt_at(fmap, gt)
            valid = gt_f != ignore_index
            if mode == "pairs":
                fl = F.interpolate(final[i][None], size=gt_f.shape, mode="bilinear", align_corners=False)[0]
                pred = fl.argmax(0)
                pm = torch.zeros(C, dtype=torch.bool, device=device)
                pr = torch.unique(gt_f[valid]); pm[pr[(pr >= 0) & (pr < C)]] = True
                pc = (pred != gt_f) & valid & pm[pred]
                if int(pc.sum()):
                    conf += torch.bincount(gt_f[pc] * C + pred[pc], minlength=C * C).double()
            else:
                fflat = fmap.permute(1, 2, 0).reshape(-1, fmap.shape[0])  # (hw,256)
                gflat = gt_f.reshape(-1)
                for c in want_classes:
                    if counts[c] >= cap:
                        continue
                    sel = (gflat == c).nonzero(as_tuple=True)[0]
                    if sel.numel() == 0:
                        continue
                    take = sel[torch.randperm(sel.numel(), device=device)[:min(120, sel.numel())]]
                    banks[c].append(fflat[take].cpu())
                    counts[c] += int(take.numel())
        n += 1
        if n % 200 == 0:
            print(f"[{mode}] {n} images", flush=True)
        if max_images > 0 and n >= max_images:
            break
    return conf, banks


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    model = init_model(cfg, args.checkpoint, device=args.device)
    model.eval()
    dev = args.device
    C = int(model.decode_head.num_classes)
    ignore_index = int(getattr(model.decode_head, "ignore_index", 255))
    meta = getattr(model, "dataset_meta", None) or {}
    classes = list(meta.get("classes", [])) or [str(i) for i in range(C)]

    store = {}
    handle = model.decode_head.align.register_forward_hook(lambda m, i, o: store.__setitem__("f", o.detach()))

    conf, _ = _collect(model, _loader_512(cfg), store, args.pair_images, dev, None, 0, C, ignore_index, "pairs")
    topv, topi = conf.topk(min(args.topn, C * C))
    pairs = [(int(topi[r]) // C, int(topi[r]) % C) for r in range(topv.numel()) if float(topv[r]) > 0]
    want = sorted({c for pr in pairs for c in pr})
    print(f"top pairs: {[(classes[g][:12], classes[p][:12]) for g, p in pairs]}", flush=True)

    _, banks = _collect(model, _loader_512(cfg), store, args.feat_images, dev, want, args.per_class_cap, C, ignore_index, "feats")
    banks = {c: (torch.cat(v, 0) if v else torch.zeros(0, 256)) for c, v in banks.items()}
    handle.remove()

    print("=" * 84)
    print(f"#3b FROZEN-FEATURE PAIRWISE SEPARABILITY (linear probe, A vs B)")
    print(f"   {'gt -> pred (confused pair)':<40}{'acc':>8}{'chance':>8}{'gain':>8}")
    accs = []
    for g, p in pairs:
        xa, xb = banks.get(g), banks.get(p)
        if xa is None or xb is None or len(xa) < 50 or len(xb) < 50:
            continue
        acc, chance = _linear_probe_acc(xa.to(dev), xb.to(dev), args.probe_steps, dev)
        accs.append(acc - chance)
        name = f"{classes[g][:16]} -> {classes[p][:16]}"
        print(f"   {name:<40}{100*acc:>7.1f}%{100*chance:>7.1f}%{100*(acc-chance):>+7.1f}")
    print("=" * 84)
    if accs:
        avg = sum(accs) / len(accs)
        print(f"mean separability gain over chance: {100*avg:+.1f} pts")
    print("Readout: acc >> chance (e.g. >85%) -> feature separates them, base draws boundary wrong")
    print("        -> end-to-end margin/separation loss can fix (lighter). ~chance -> feature entangles")
    print("        -> must change representation end-to-end (the real, heavier contribution).")


if __name__ == "__main__":
    main()
