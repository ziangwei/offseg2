# -*- coding: utf-8 -*-
"""Figure-1 probe: are the frozen base's residual errors confident & self-consistent?

Read-only pass over ADE val on a frozen checkpoint. For every valid pixel it:
  1) splits pixels into  correct / wrong  by argmax vs GT,
  2) splits every WRONG pixel into two kinds:
       - ABSENT-FP    : predicted class is NOT in this image's GT class set
                        (exactly what the active-class oracle removes),
       - PRESENT-CONF : predicted class IS present, just the wrong one
                        (genuine inter-class confusion the oracle can't fix),
  3) records the base's max-softmax confidence (and top1-top2 margin) per group,
  4) builds a reliability table (confidence bin -> accuracy).

Why: the active-class oracle showed +10.22 mIoU headroom but 0% realizable. This
probe tests the explanation -- that the ABSENT-FP errors are CONFIDENT, so no
signal derived from the base can rule them out -- and measures how much of the
error is genuine PRESENT-CONF confusion (the target for a 'prevent' approach).

Usage (server, single GPU):
  python tools/probe_error_confidence.py \
      local_configs/offseg2/Base/parseg3_ade20k_160k-512x512.py \
      <base_ckpt>.pth --out work_dirs/fig1_error_confidence.json
"""
import argparse
import json

import torch
import torch.nn.functional as F
from mmengine.config import Config
from mmengine.runner import Runner

from mmseg.apis import init_model

GROUPS = ["correct", "absent_fp", "present_conf"]


def parse_args():
    p = argparse.ArgumentParser(description="Error-confidence / error-decomposition probe (read-only).")
    p.add_argument("config")
    p.add_argument("checkpoint")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-images", type=int, default=-1)
    p.add_argument("--nbins", type=int, default=20)
    p.add_argument("--out", default=None, help="optional json dump of histograms/reliability for plotting")
    return p.parse_args()


def _val_loader(cfg):
    loader = dict(cfg.val_dataloader)
    loader["batch_size"] = 1
    loader["sampler"] = dict(type="DefaultSampler", shuffle=False)
    return Runner.build_dataloader(loader)


@torch.no_grad()
def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    model = init_model(cfg, args.checkpoint, device=args.device)
    model.eval()

    dev = args.device
    nb = int(args.nbins)
    num_classes = int(model.decode_head.num_classes)
    ignore_index = int(getattr(model.decode_head, "ignore_index", 255))
    loader = _val_loader(cfg)

    cnt = {g: 0 for g in GROUPS}
    conf_sum = {g: 0.0 for g in GROUPS}
    margin_sum = {g: 0.0 for g in GROUPS}
    conf_hist = {g: torch.zeros(nb) for g in GROUPS}
    rel_total = torch.zeros(nb)
    rel_correct = torch.zeros(nb)
    wrong_total = 0
    wrong_ge = {0.5: 0, 0.7: 0, 0.9: 0}

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

            probs = torch.softmax(logits, dim=0)
            top2 = probs.topk(2, dim=0).values
            conf = top2[0]
            margin = top2[0] - top2[1]
            pred = probs.argmax(0)

            valid = gt != ignore_index
            present_mask = torch.zeros(num_classes, dtype=torch.bool, device=dev)
            pres = torch.unique(gt[valid])
            pres = pres[(pres >= 0) & (pres < num_classes)]
            present_mask[pres] = True
            pred_present = present_mask[pred]

            correct = (pred == gt) & valid
            wrong = (pred != gt) & valid
            absent_fp = wrong & (~pred_present)
            present_conf = wrong & pred_present

            for g, msk in [("correct", correct), ("absent_fp", absent_fp), ("present_conf", present_conf)]:
                k = int(msk.sum())
                cnt[g] += k
                if k:
                    c = conf[msk]
                    conf_sum[g] += float(c.sum())
                    margin_sum[g] += float(margin[msk].sum())
                    conf_hist[g] += torch.histc(c.cpu(), bins=nb, min=0.0, max=1.0)

            cv = conf[valid]
            bin_idx = (cv * nb).long().clamp(0, nb - 1)
            rel_total += torch.bincount(bin_idx.cpu(), minlength=nb).float()
            rel_correct += torch.bincount(bin_idx[correct[valid]].cpu(), minlength=nb).float()

            wc = conf[wrong]
            wrong_total += int(wrong.sum())
            for t in wrong_ge:
                wrong_ge[t] += int((wc >= t).sum())

            n += 1
            if n % 200 == 0:
                print(f"[probe] {n} images", flush=True)
            if args.max_images > 0 and n >= args.max_images:
                stop = True
                break

    total_valid = sum(cnt.values())
    tv = max(total_valid, 1)
    wt = max(wrong_total, 1)

    def mean_conf(g):
        return conf_sum[g] / max(cnt[g], 1)

    def mean_margin(g):
        return margin_sum[g] / max(cnt[g], 1)

    lines = []
    lines.append("=" * 78)
    lines.append(f"images                : {n}")
    lines.append(f"valid pixels          : {total_valid:,}")
    lines.append(f"pixel accuracy        : {100 * cnt['correct'] / tv:.2f}%")
    lines.append("-" * 78)
    lines.append("ERROR DECOMPOSITION (share of all valid pixels)")
    lines.append(f"  correct                          : {100 * cnt['correct'] / tv:6.2f}%")
    lines.append(f"  wrong - ABSENT-FP (oracle-fixes) : {100 * cnt['absent_fp'] / tv:6.2f}%   "
                 f"mean_conf={mean_conf('absent_fp'):.3f}  mean_margin={mean_margin('absent_fp'):.3f}")
    lines.append(f"  wrong - PRESENT-CONF (true conf) : {100 * cnt['present_conf'] / tv:6.2f}%   "
                 f"mean_conf={mean_conf('present_conf'):.3f}  mean_margin={mean_margin('present_conf'):.3f}")
    lines.append(f"  (correct mean_conf={mean_conf('correct'):.3f})")
    lines.append("-" * 78)
    lines.append("ARE ERRORS CONFIDENT?  (fraction of ALL wrong pixels with conf >= t)")
    lines.append(f"  conf>=0.5 : {100 * wrong_ge[0.5] / wt:5.1f}%     "
                 f"conf>=0.7 : {100 * wrong_ge[0.7] / wt:5.1f}%     conf>=0.9 : {100 * wrong_ge[0.9] / wt:5.1f}%")
    share_absent = 100 * cnt["absent_fp"] / max(cnt["absent_fp"] + cnt["present_conf"], 1)
    lines.append(f"  of all errors: ABSENT-FP {share_absent:.1f}%  vs  PRESENT-CONF {100 - share_absent:.1f}%")
    lines.append("-" * 78)
    lines.append("RELIABILITY  (confidence bin -> accuracy; gap at high conf = confident errors)")
    for b in range(nb):
        tot = float(rel_total[b])
        if tot <= 0:
            continue
        acc = float(rel_correct[b]) / tot
        lines.append(f"  [{b / nb:.2f},{(b + 1) / nb:.2f})  n={int(tot):>9,}  acc={100 * acc:6.2f}%")
    lines.append("=" * 78)
    lines.append("Readout: ABSENT-FP 高 mean_conf + 高 conf>=0.7 → 自信误报,任何 base 派生先验都压不掉")
    lines.append("        (解释 active-class 0% realizable)。PRESENT-CONF = 真·语义混淆,是'防/表示层'")
    lines.append("        要打的部分。高 conf bin 的 acc 明显 < 100% = 自信错确实存在。")

    summary = "\n".join(lines)
    print(summary)

    if args.out:
        payload = {
            "images": n,
            "total_valid": total_valid,
            "counts": cnt,
            "mean_conf": {g: mean_conf(g) for g in GROUPS},
            "mean_margin": {g: mean_margin(g) for g in GROUPS},
            "wrong_total": wrong_total,
            "wrong_ge": {str(k): v for k, v in wrong_ge.items()},
            "nbins": nb,
            "conf_hist": {g: conf_hist[g].tolist() for g in GROUPS},
            "reliability_total": rel_total.tolist(),
            "reliability_correct": rel_correct.tolist(),
        }
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
