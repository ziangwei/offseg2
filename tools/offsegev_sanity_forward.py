# -*- coding: utf-8 -*-
"""Pre-flight checks for the EV round (no dataset, no checkpoint, CPU ok).

Run this BEFORE spending five 24h slots. It checks the things that would
silently invalidate the whole round:

  1. identity at init -- every RCM gamma and every PCE level scalar starts at
     0, so slots 1/2 must be bit-identical to stock OffSeg and slots 3/4/5
     bit-identical to OffSegCCM at step 0. If this fails, the "structural
     part + identity start" rule is broken and no delta is attributable.
  2. the switches actually switch -- with gammas forced non-zero the output
     must CHANGE, and only at the sites the flag claims. A flag that does
     nothing would produce five identical numbers and waste the round.
  3. shapes survive the FreqFusion path at the real crop (512), including the
     b*4 reshape trick OffSeg uses between fusion steps.
  4. parameter and FLOPs budget per slot, against the senior's +2.75M/+8.38G.

Usage:
  python tools/offsegev_sanity_forward.py
  python tools/offsegev_sanity_forward.py --device cuda:0 --crop 512
"""
import argparse

import torch

from mmseg.models.decode_heads.OffSegRCM import RCM, \
    RectangularSelfCalibrationAttention  # noqa: F401  (import-time check)
from mmseg.models.decode_heads.OffSegEV import OffSegEV

HAM_NORM = dict(type='GN', num_groups=32, requires_grad=True)
IN_CH = [32, 64, 144, 288]
NEW_CH = [32, 64, 128, 256]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='cpu')
    p.add_argument('--crop', type=int, default=512)
    p.add_argument('--batch', type=int, default=2)
    return p.parse_args()


def _ok(name, cond, extra=''):
    print(f"  [{'OK ' if cond else 'FAIL'}] {name} {extra}")
    return bool(cond)


def build(ev_ccm, ev_pce, ev_sfr):
    return OffSegEV(
        in_channels=IN_CH, new_channels=NEW_CH, in_index=[0, 1, 2, 3],
        channels=256, dropout_ratio=0.1, num_classes=150,
        norm_cfg=HAM_NORM, align_corners=False,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False,
                         loss_weight=1.0),
        ev_ccm=ev_ccm, ev_pce=ev_pce, ev_sfr=ev_sfr, rcm_depth=1)


def make_inputs(batch, crop, device):
    s = crop // 4
    return [torch.randn(batch, IN_CH[i], s // (2 ** i), s // (2 ** i),
                        device=device) for i in range(4)]


def excite(head):
    """Force every identity gate open so the switches must bite."""
    with torch.no_grad():
        if head.ev_pce:
            for m in head.pce:
                m.gamma.fill_(0.1)
            for g in head.pce_gamma:
                g.fill_(0.1)
        if head.ev_sfr:
            for m in head.sfr:
                m.gamma.fill_(0.1)


def main():
    args = parse_args()
    dev = torch.device(args.device)
    torch.manual_seed(0)
    x = make_inputs(args.batch, args.crop, dev)
    passed = []

    slots = [('slot1 OffSeg+PCE', False, True, False),
             ('slot2 OffSeg+SFR', False, False, True),
             ('slot3 CCM+PCE', True, True, False),
             ('slot4 CCM+SFR', True, False, True),
             ('slot5 CCM+PCE+SFR', True, True, True)]

    print('\n1) identity at init, and the switches actually switch')
    budget = []
    for name, ccm, pce, sfr in slots:
        torch.manual_seed(0)
        ref = build(ccm, False, False).to(dev).eval()   # same head, no evidence
        torch.manual_seed(0)
        head = build(ccm, pce, sfr).to(dev).eval()

        with torch.no_grad():
            a = ref(x)['final_logits']
            b = head(x)['final_logits']
        same = torch.equal(a, b)
        passed.append(_ok(f'{name}: identical to its no-evidence twin at init',
                          same))

        excite(head)
        with torch.no_grad():
            c = head(x)['final_logits']
        passed.append(_ok(f'{name}: output changes once gammas are opened',
                          not torch.equal(a, c)))
        passed.append(_ok(f'{name}: shape {tuple(c.shape)} preserved',
                          c.shape == a.shape))

        base = sum(p.numel() for p in ref.parameters())
        full = sum(p.numel() for p in head.parameters())
        budget.append((name, (full - base) / 1e6))

    print('\n2) losses wire up (one CE without CCM, two with)')
    for name, ccm, pce, sfr in slots:
        head = build(ccm, pce, sfr).to(dev).eval()
        with torch.no_grad():
            out = head(x)
        keys = ['loss_ce'] + (['loss_stage1'] if ccm else [])
        needles = ([] + (['acc_pce_gamma'] if pce else [])
                   + (['acc_sfr_gamma'] if sfr else [])
                   + (['acc_ccm_gain'] if ccm else []))
        has_s1 = (out['stage1_logits'] is not None) == ccm
        passed.append(_ok(f'{name}: expects {keys} + needles {needles}',
                          has_s1))

    print('\n3) head parameter budget (decode head only, vs stock OffSeg head)')
    print(f"    {'slot':<20}{'added':>10}")
    for name, d in budget:
        print(f"    {name:<20}{d:>9.2f}M")
    print(f"    {'senior PARSeg3':<20}{2.75:>9.2f}M   (+8.38 GFLOPs)")

    print(f"\n{sum(passed)}/{len(passed)} checks passed")
    if not all(passed):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
