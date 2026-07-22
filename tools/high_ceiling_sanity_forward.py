"""Verify PARSeg3-equivalent initialization for FA-U-Mix and PCQ."""

import argparse

import torch
from mmengine.config import Config

from mmseg.models.decode_heads.PARSegFAUMix import PARSegFAUMix  # noqa: F401
from mmseg.models.decode_heads.PARSegPCQ import PARSegPCQ  # noqa: F401
from mmseg.registry import MODELS
from mmseg.utils import register_all_modules


DEFAULT_BASE = (
    'local_configs/offseg2/Base/'
    'parseg3_ade20k_160k-512x512.py')
DEFAULT_FA = (
    'local_configs/offseg2/Base/'
    'parseg3_faumix_ade20k_160k-512x512.py')
DEFAULT_PCQ = (
    'local_configs/offseg2/Base/'
    'parseg3_pcq_ade20k_160k-512x512.py')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-config', default=DEFAULT_BASE)
    parser.add_argument('--fa-config', default=DEFAULT_FA)
    parser.add_argument('--pcq-config', default=DEFAULT_PCQ)
    parser.add_argument('--image-size', type=int, default=128)
    parser.add_argument('--atol', type=float, default=1e-5)
    parser.add_argument(
        '--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


def max_abs(left, right):
    return (left.float() - right.float()).abs().max().item()


def parameter_count(module):
    return sum(parameter.numel() for parameter in module.parameters())


def check_missing(incompatible, allowed_prefix):
    unexpected = list(incompatible.unexpected_keys)
    missing = list(incompatible.missing_keys)
    invalid_missing = [
        key for key in missing if not key.startswith(allowed_prefix)
    ]
    if unexpected or invalid_missing or not missing:
        raise RuntimeError(
            f'Invalid checkpoint compatibility for {allowed_prefix}: '
            f'missing={missing}, unexpected={unexpected}')


def decode(model, image):
    features = model.backbone(image)
    return features, model.decode_head(features)


def main():
    args = parse_args()
    register_all_modules()
    configs = {
        'base': Config.fromfile(args.base_config),
        'faumix': Config.fromfile(args.fa_config),
        'pcq': Config.fromfile(args.pcq_config),
    }
    models = {
        name: MODELS.build(config.model)
        for name, config in configs.items()
    }

    base_state = models['base'].state_dict()
    fa_incompatible = models['faumix'].load_state_dict(
        base_state, strict=False)
    pcq_incompatible = models['pcq'].load_state_dict(
        base_state, strict=False)
    check_missing(fa_incompatible, 'decode_head.fa_umix.')
    check_missing(
        pcq_incompatible,
        'decode_head.offset_learning.query_updater.',
    )

    for model in models.values():
        model.to(args.device).eval()
    image = torch.randn(
        1, 3, args.image_size, args.image_size, device=args.device)
    with torch.inference_mode():
        results = {
            name: decode(model, image)
            for name, model in models.items()
        }

    output_keys = (
        'base_head_logits',
        'refinement_head_logits',
        'final_logits',
    )
    for name in ('faumix', 'pcq'):
        feature_errors = [
            max_abs(base_feature, variant_feature)
            for base_feature, variant_feature in zip(
                results['base'][0], results[name][0])
        ]
        output_errors = {
            key: max_abs(results['base'][1][key], results[name][1][key])
            for key in output_keys
        }
        maximum_error = max(feature_errors + list(output_errors.values()))
        print(f'{name} encoder max-abs: {feature_errors}')
        print(f'{name} decoder max-abs: {output_errors}')
        if maximum_error > args.atol:
            raise RuntimeError(
                f'{name} differs from PARSeg3 by {maximum_error}, '
                f'above atol={args.atol}')

    fa_gate = models['faumix'].decode_head.fa_umix.faumix_gate
    pcq_gate = (models['pcq'].decode_head.offset_learning.query_updater
                .pcq_gates)
    if torch.count_nonzero(fa_gate).item() != 0:
        raise RuntimeError('FA-U-Mix gate is not zero initialized')
    if torch.count_nonzero(pcq_gate).item() != 0:
        raise RuntimeError('PCQ gates are not zero initialized')

    base_parameters = parameter_count(models['base'])
    for name in ('faumix', 'pcq'):
        added = parameter_count(models[name]) - base_parameters
        print(f'{name} added parameters: {added:,}')

    # Prove that the branches are live rather than zero-output dead paths.
    with torch.no_grad():
        fa_gate.fill_(0.01)
        pcq_gate.fill_(0.01)
        activated = {
            name: decode(models[name], image)[1]
            for name in ('faumix', 'pcq')
        }
    for name in ('faumix', 'pcq'):
        change = max_abs(
            results['base'][1]['final_logits'],
            activated[name]['final_logits'],
        )
        print(f'{name} activated final-logit change: {change:.6g}')
        if not change > 0:
            raise RuntimeError(f'{name} branch remains inactive after gating')

    print('High-ceiling decoder initialization sanity check passed.')


if __name__ == '__main__':
    main()
