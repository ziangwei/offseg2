# -*- coding: utf-8 -*-
"""WHY probe: which classes get confused, where, and how hard the re-rank is.

Read-only frozen pass over ADE val. For PRESENT-CONF pixels (wrong, predicted
class IS in the image -- the genuine semantic confusion that holds the mIoU):
  #1 CONFUSION PAIRS : top (gt -> pred) class pairs + concentration (do a few
                       pairs dominate?). Reported for interior pixels too.
  #2 BOUNDARY/INTERIOR: split errors into near-a-class-boundary vs object
                        interior. Interior = genuine semantic; boundary = noise.
  #3a RE-RANK HARDNESS: for present-conf where GT is the runner-up (top-2),
                        the logit margin (wrong_top1 - correct_gt). Small margin
                        => an end-to-end nudge can flip it; large => deep.

Usage (server, single GPU):
  python tools/probe_confusion_boundary.py \
      local_configs/offseg2/Base/parseg3_ade20k_160k-512x512.py \
      <base_ckpt>.pth --topn 25 --boundary-radius 3
"""
import argparse

import torch
import torch.nn.functional as F
from mmengine.config import Config
from mmengine.runner import Runner

from mmseg.apis import init_model


def parse_args():
    p = argparse.ArgumentParser(description="Confusion pairs + boundary/interior + re-rank hardness (read-only).")
    p.add_argument("config")
    p.add_argument("checkpoint")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-images", type=int, default=-1)
    p.add_argument("--topn", type=int, default=25)
    p.add_argument("--boundary-radius", type=int, default=3)
    return p.parse_args()


def _val_loader(cfg):
    loader = dict(cfg.val_dataloader)
    loader["batch_size"] = 1
    loader["sampler"] = dict(type="DefaultSampler", shuffle=False)
    return Runner.build_dataloader(loader)


def _boundary(gt, valid, radius):
    g = gt.clone().float()
    g[~valid] = -1.0
    g = g[None, None]
    k = 2 * radius + 1
    mx = F.max_pool2d(g, k, 1, radius)
    mn = -F.max_pool2d(-g, k, 1, radius)
    return (mx != mn).squeeze(0).squeeze(0) & valid


@torch.no_grad()
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
    loader = _val_loader(cfg)

    conf_all = torch.zeros(C * C, dtype=torch.float64, device=dev)      # present-conf (gt->pred)
    conf_int = torch.zeros(C * C, dtype=torch.float64, device=dev)      # present-conf & interior
    cnt = {"pc": 0, "pc_boundary": 0, "pc_interior": 0, "wrong": 0, "wrong_boundary": 0}
    confsum = {"pc_boundary": 0.0, "pc_interior": 0.0}
    margin_hist = torch.zeros(40, device=dev)                           # 0..4 logits, 0.1 bins
    margin_lt = {0.5: 0, 1.0: 0, 2.0: 0}
    margin_n = 0

    n = 0
    stop = False
    for data in loader:
        if stop:
            break
        for result in model.test_step(data):
            logits = result.seg_logits.data.float().to(dev)
            gt = result.gt_sem_seg.data.squeeze(0).long().to(dev)
            if logits.shape[-2:] != gt.shape[-2:]:
                logits = F.interpolate(logits[None], size=gt.shape[-2:], mode="bilinear", align_corners=False)[0]

            pred = logits.argmax(0)
            valid = gt != ignore_index
            present_mask = torch.zeros(C, dtype=torch.bool, device=dev)
            pres = torch.unique(gt[valid]); pres = pres[(pres >= 0) & (pres < C)]
            present_mask[pres] = True

            wrong = (pred != gt) & valid
            present_conf = wrong & present_mask[pred]
            boundary = _boundary(gt, valid, args.boundary_radius)
            pc_b = present_conf & boundary
            pc_i = present_conf & (~boundary)

            cnt["wrong"] += int(wrong.sum()); cnt["wrong_boundary"] += int((wrong & boundary).sum())
            cnt["pc"] += int(present_conf.sum())
            cnt["pc_boundary"] += int(pc_b.sum()); cnt["pc_interior"] += int(pc_i.sum())

            conf = logits.softmax(0).max(0).values
            confsum["pc_boundary"] += float(conf[pc_b].sum()); confsum["pc_interior"] += float(conf[pc_i].sum())

            if int(present_conf.sum()):
                idx = (gt[present_conf] * C + pred[present_conf])
                conf_all += torch.bincount(idx, minlength=C * C).double()
            if int(pc_i.sum()):
                idx = (gt[pc_i] * C + pred[pc_i])
                conf_int += torch.bincount(idx, minlength=C * C).double()

            top2v, top2i = logits.topk(2, dim=0)
            runner_is_gt = present_conf & (top2i[1] == gt)         # correct class is the runner-up
            if int(runner_is_gt.sum()):
                m = (top2v[0] - top2v[1])[runner_is_gt]
                margin_hist += torch.histc(m.clamp(0, 3.999), bins=40, min=0, max=4)
                margin_n += int(runner_is_gt.sum())
                for t in margin_lt:
                    margin_lt[t] += int((m < t).sum())

            n += 1
            if n % 200 == 0:
                print(f"[probe] {n} images", flush=True)
            if args.max_images > 0 and n >= args.max_images:
                stop = True
                break

    pc = max(cnt["pc"], 1)
    print("=" * 84)
    print(f"images                 : {n}")
    print(f"present-conf pixels     : {cnt['pc']:,}  ({100*cnt['pc']/max(cnt['wrong'],1):.1f}% of all wrong)")
    print("-" * 84)
    print("#2 BOUNDARY vs INTERIOR (present-conf)")
    print(f"  boundary : {100*cnt['pc_boundary']/pc:5.1f}%   mean_conf={confsum['pc_boundary']/max(cnt['pc_boundary'],1):.3f}")
    print(f"  interior : {100*cnt['pc_interior']/pc:5.1f}%   mean_conf={confsum['pc_interior']/max(cnt['pc_interior'],1):.3f}")
    print(f"  (all wrong that sit on a boundary: {100*cnt['wrong_boundary']/max(cnt['wrong'],1):.1f}%)")
    print("-" * 84)
    print(f"#3a RE-RANK HARDNESS (present-conf where correct class is the runner-up; N={margin_n:,})")
    print(f"   margin = wrong_top1_logit - correct_logit")
    for t in (0.5, 1.0, 2.0):
        print(f"   margin < {t:<3}: {100*margin_lt[t]/max(margin_n,1):5.1f}%")
    print("-" * 84)
    print(f"#1 TOP CONFUSION PAIRS (interior present-conf; share of interior present-conf)")
    ci = conf_int
    total_int = float(ci.sum())
    topv, topi = ci.topk(min(args.topn, C * C))
    cum = 0.0
    print(f"   {'rank':>4}  {'gt -> pred':<42}{'count':>12}{'share':>8}{'cum':>8}")
    for r in range(topv.numel()):
        c = float(topv[r])
        if c <= 0:
            break
        g, p = int(topi[r]) // C, int(topi[r]) % C
        cum += c
        gname = classes[g][:18] if g < len(classes) else str(g)
        pname = classes[p][:18] if p < len(classes) else str(p)
        print(f"   {r+1:>4}  {gname+' -> '+pname:<42}{int(c):>12,}{100*c/max(total_int,1):>7.1f}%{100*cum/max(total_int,1):>7.1f}%")
    print("=" * 84)
    print(f"Readout: top-{args.topn} pairs cover {100*cum/max(total_int,1):.1f}% of interior present-conf.")
    print("        concentrated + interior-heavy + small re-rank margin => clean target for an")
    print("        end-to-end separation ('fang') loss on those confusable pairs. diffuse/boundary-heavy")
    print("        or large margin => harder / leans toward injecting new info.")


if __name__ == "__main__":
    main()
