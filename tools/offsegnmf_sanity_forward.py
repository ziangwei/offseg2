# -*- coding: utf-8 -*-
"""Dataset-free pre-flight checks for OffSeg-NMF and OffSeg-CCM-NMF.

Usage:
  python tools/offsegnmf_sanity_forward.py --crop 128 --batch 1
  python tools/offsegnmf_sanity_forward.py --device cuda:0 --crop 512
"""
import argparse

import torch

from mmseg.models.decode_heads.OffSegCCM import OffSegCCM
from mmseg.models.decode_heads.OffSegNMF import OffSegCCMNMF, OffSegNMF
from mmseg.models.decode_heads.offseg_head import OffSegHead


NORM_CFG = dict(type='GN', num_groups=32, requires_grad=True)
IN_CHANNELS = [32, 64, 144, 288]
NEW_CHANNELS = [32, 64, 128, 256]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--crop', type=int, default=128)
    parser.add_argument('--batch', type=int, default=1)
    return parser.parse_args()


def common_kwargs():
    return dict(
        in_channels=IN_CHANNELS,
        new_channels=NEW_CHANNELS,
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=NORM_CFG,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))


def nmf_kwargs():
    return dict(
        nmf_ham_channels=256,
        nmf_rank=32,
        nmf_train_steps=3,
        nmf_eval_steps=3,
        nmf_pool_stride=2)


def make_inputs(batch, crop, device):
    stride4 = crop // 4
    return [
        torch.randn(
            batch,
            IN_CHANNELS[i],
            stride4 // (2 ** i),
            stride4 // (2 ** i),
            device=device)
        for i in range(4)
    ]


def count_params(module):
    return sum(parameter.numel() for parameter in module.parameters())


def check(name, condition, detail=''):
    mark = 'OK' if condition else 'FAIL'
    print(f'  [{mark:4s}] {name} {detail}')
    return bool(condition)


def build_pair(base_class, nmf_class, device):
    torch.manual_seed(0)
    base = base_class(**common_kwargs()).to(device).eval()
    torch.manual_seed(0)
    nmf = nmf_class(**common_kwargs(), **nmf_kwargs()).to(device).eval()

    # The base subgraph has the same construction order.  Loading it
    # explicitly makes the identity check robust to future init changes.
    compatible = {
        key: value for key, value in base.state_dict().items()
        if key in nmf.state_dict() and nmf.state_dict()[key].shape == value.shape
    }
    nmf.load_state_dict(compatible, strict=False)
    return base, nmf


def main():
    args = parse_args()
    device = torch.device(args.device)
    torch.manual_seed(1)
    inputs = make_inputs(args.batch, args.crop, device)
    passed = []

    print('\n1) exact identity at initialisation')
    pairs = [
        ('OffSeg -> OffSegNMF', OffSegHead, OffSegNMF),
        ('CCM -> CCM+NMF', OffSegCCM, OffSegCCMNMF),
    ]
    built = []
    for name, base_class, nmf_class in pairs:
        base, nmf = build_pair(base_class, nmf_class, device)
        with torch.no_grad():
            base_out = base(inputs)
            nmf_out = nmf(inputs)
            if isinstance(base_out, dict):
                base_out = base_out['final_logits']
                nmf_out = nmf_out['final_logits']
        passed.append(check(
            f'{name}: logits are bit-identical',
            torch.equal(base_out, nmf_out),
            f'shape={tuple(nmf_out.shape)}'))
        built.append((name, base, nmf, base_out))

    print('\n2) branch is live once the identity gate opens')
    for name, _, nmf, base_out in built:
        with torch.no_grad():
            nmf.nmf_preconditioner.gamma.fill_(0.1)
            changed = nmf(inputs)
            if isinstance(changed, dict):
                changed = changed['final_logits']
        passed.append(check(
            f'{name}: output changes', not torch.equal(base_out, changed)))

    print('\n3) gradient and parameter budget')
    _, base, nmf, _ = built[0]
    nmf.train()
    nmf.nmf_preconditioner.gamma.data.zero_()
    logits = nmf(inputs)
    logits.float().square().mean().backward()
    gamma_grad = nmf.nmf_preconditioner.gamma.grad
    gamma_grad_text = ('None' if gamma_grad is None
                       else f'{gamma_grad.item():.6g}')
    passed.append(check(
        'identity gate receives a finite non-zero gradient',
        gamma_grad is not None
        and torch.isfinite(gamma_grad).all()
        and gamma_grad.abs().sum() > 0,
        f'grad={gamma_grad_text}'))

    added = count_params(nmf) - count_params(base)
    passed.append(check(
        'NMF branch stays below 0.14M learned parameters',
        added < 140_000,
        f'added={added / 1e6:.4f}M'))

    print(f'\n{sum(passed)}/{len(passed)} checks passed')
    if not all(passed):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
