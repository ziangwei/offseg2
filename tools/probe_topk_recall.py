# -*- coding: utf-8 -*-
"""Top-k recall probe: is the right answer already in the base's top-k? (a vs b arbiter)

Read-only pass over ADE val on a frozen checkpoint. For every WRONG pixel (split
into ABSENT-FP vs PRESENT-CONF, same as the Figure-1 probe) it asks whether the
GT class is among the base's top-2/3/5/10 logits. It also computes the
"perfect top-k re-rank" oracle mIoU: pick GT whenever GT is in the top-k, else
keep argmax -- the exact ceiling a disambiguation / 'prevent' mechanism could
reach if it only re-ranks within the base's own top-k candidates.

Decision rule:
  * GT mostly IN top-2/3 (esp. for PRESENT-CONF)  -> the answer is encoded, just
    out-ranked -> a re-rank / representation ('fang') mechanism has a measurable
    ceiling (the oracle delta below sizes it).  -> direction (b)
  * GT often NOT in top-5 -> the representation doesn't encode it -> you must
    inject new information.  -> direction (a)

Usage (server, single GPU):
  python tools/probe_topk_recall.py \
      local_configs/offseg2/Base/parseg3_ade20k_160k-512x512.py \
      <base_ckpt>.pth
"""
import argparse

import torch
import torch.nn.functional as F
from mmengine.config import Config
from mmengine.runner import Runner

from mmseg.apis import init_model

KS = [2, 3, 5, 10]
ORACLE_KS = [2, 3, 5]


def parse_args():
    p = argparse.ArgumentParser(description="Top-k recall on wrong pixels + top-k re-rank oracle (read-only).")
    p.add_argument("config")
    p.add_argument("checkpoint")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-images", type=int, default=-1)
    return p.parse_args()


def _val_loader(cfg):
    loader = dict(cfg.val_dataloader)
    loader["batch_size"] = 1
    loader["sampler"] = dict(type="DefaultSampler", shuffle=False)
    return Runner.build_dataloader(loader)


def _iu(pred, label, num_classes, ignore_index, device):
    mask = label != ignore_index
    pred = pred[mask].float()
    label = label[mask].float()
    inter = pred[pred == label]
    ai = torch.histc(inter, bins=num_classes, min=0, max=num_classes - 1).to(device)
    ap = torch.histc(pred, bins=num_classes, min=0, max=num_classes - 1).to(device)
    al = torch.histc(label, bins=num_classes, min=0, max=num_classes - 1).to(device)
    return ai, ap + al - ai


def _miou(inter, union):
    valid = union > 0
    if not bool(valid.any()):
        return 0.0
    return float((inter[valid] / union[valid].clamp_min(1.0)).mean().item() * 100)


@torch.no_grad()
def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    model = init_model(cfg, args.checkpoint, device=args.device)
    model.eval()

    dev = args.device
    num_classes = int(model.decode_head.num_classes)
    ignore_index = int(getattr(model.decode_head, "ignore_index", 255))
    maxk = min(max(KS), num_classes)
    loader = _val_loader(cfg)

    base_i = torch.zeros(num_classes, device=dev)
    base_u = torch.zeros(num_classes, device=dev)
    orc_i = {k: torch.zeros(num_classes, device=dev) for k in ORACLE_KS}
    orc_u = {k: torch.zeros(num_classes, device=dev) for k in ORACLE_KS}

    tot = {"wrong": 0, "absent_fp": 0, "present_conf": 0}
    ink = {grp: {k: 0 for k in KS} for grp in ("wrong", "absent_fp", "present_conf")}

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
            present_mask = torch.zeros(num_classes, dtype=torch.bool, device=dev)
            pres = torch.unique(gt[valid])
            pres = pres[(pres >= 0) & (pres < num_classes)]
            present_mask[pres] = True

            wrong = (pred != gt) & valid
            absent_fp = wrong & (~present_mask[pred])
            present_conf = wrong & present_mask[pred]

            topk_idx = logits.topk(maxk, dim=0).indices
            eqs = topk_idx == gt.unsqueeze(0)

            tot["wrong"] += int(wrong.sum())
            tot["absent_fp"] += int(absent_fp.sum())
            tot["present_conf"] += int(present_conf.sum())
            for k in KS:
                ink_k = eqs[:k].any(0) & valid
                ink["wrong"][k] += int((wrong & ink_k).sum())
                ink["absent_fp"][k] += int((absent_fp & ink_k).sum())
                ink["present_conf"][k] += int((present_conf & ink_k).sum())

            ai, au = _iu(pred, gt, num_classes, ignore_index, dev)
            base_i += ai; base_u += au
            for k in ORACLE_KS:
                ink_k = eqs[:k].any(0)
                oracle_pred = torch.where(ink_k, gt, pred)
                ai, au = _iu(oracle_pred, gt, num_classes, ignore_index, dev)
                orc_i[k] += ai; orc_u[k] += au

            n += 1
            if n % 200 == 0:
                print(f"[probe] {n} images", flush=True)
            if args.max_images > 0 and n >= args.max_images:
                stop = True
                break

    base_miou = _miou(base_i, base_u)

    def rec(grp, k):
        return 100 * ink[grp][k] / max(tot[grp], 1)

    print("=" * 76)
    print(f"images                : {n}")
    print(f"baseline mIoU         : {base_miou:.2f}   (self-check ~48.2)")
    print("-" * 76)
    print("TOP-k RECALL on wrong pixels  (is GT in the base's top-k?)")
    print(f"  {'group':<16}{'@2':>8}{'@3':>8}{'@5':>8}{'@10':>8}")
    for grp, label in [("wrong", "all wrong"), ("present_conf", "PRESENT-CONF"), ("absent_fp", "ABSENT-FP")]:
        print(f"  {label:<16}" + "".join(f"{rec(grp, k):>7.1f}%" for k in KS))
    print("-" * 76)
    print("PERFECT TOP-k RE-RANK ORACLE  (pick GT when in top-k, else argmax)")
    for k in ORACLE_KS:
        mk = _miou(orc_i[k], orc_u[k])
        print(f"  top-{k}: mIoU {mk:6.2f}   delta +{mk - base_miou:5.2f}")
    print("=" * 76)
    print("Readout: PRESENT-CONF @2/@3 high + top-2/3 oracle delta >>1 -> answer encoded but out-ranked;")
    print("        re-rank/representation (fang) has a real ceiling = direction (b). If @5 still low ->")
    print("        representation does not encode it -> must inject new info (a). Oracle = perfect-pick bound.")


if __name__ == "__main__":
    main()
