"""Verify that initialized HC-S2 is equivalent to the PARSeg3 reference.

Run this once in the full server environment before launching the 160k job.
It checks shared checkpoint keys, all four encoder outputs, final logits, and
the very small number of newly introduced connection parameters.
"""

import argparse

import torch
from mmengine.config import Config

from mmseg.models.backbones.efficientformer_v2_hc import (  # noqa: F401
    efficientformerv2_s2_hc2_feat)
from mmseg.registry import MODELS
from mmseg.utils import register_all_modules


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--base-config',
        default='local_configs/offseg2/Base/'
                'parseg3_ade20k_160k-512x512.py')
    parser.add_argument(
        '--hc-config',
        default='local_configs/offseg2/Base/'
                'parseg3_hc2_s34_ade20k_160k-512x512.py')
    parser.add_argument('--image-size', type=int, default=128)
    parser.add_argument('--atol', type=float, default=1e-5)
    parser.add_argument(
        '--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


def max_abs(left, right):
    return (left.float() - right.float()).abs().max().item()


def main():
    args = parse_args()
    register_all_modules()
    base_cfg = Config.fromfile(args.base_config)
    hc_cfg = Config.fromfile(args.hc_config)

    base_model = MODELS.build(base_cfg.model)
    hc_model = MODELS.build(hc_cfg.model)
    incompatible = hc_model.load_state_dict(base_model.state_dict(), strict=False)
    expected_missing_prefix = 'backbone.hc_units.'
    expected_missing = {
        f'backbone.{key}'
        for key in hc_model.backbone.state_dict()
        if key.startswith('hc_units.')
    }
    bad_missing = [
        key for key in incompatible.missing_keys
        if not key.startswith(expected_missing_prefix)
    ]
    if (bad_missing or set(incompatible.missing_keys) != expected_missing or
            incompatible.unexpected_keys):
        raise RuntimeError(
            f'Checkpoint mismatch: missing={incompatible.missing_keys}, '
            f'unexpected={incompatible.unexpected_keys}')

    base_model.to(args.device).eval()
    hc_model.to(args.device).eval()
    image = torch.randn(
        1, 3, args.image_size, args.image_size, device=args.device)

    with torch.inference_mode():
        base_features = base_model.backbone(image)
        hc_features = hc_model.backbone(image)
        feature_errors = [
            max_abs(base, hc)
            for base, hc in zip(base_features, hc_features)
        ]
        base_logits = base_model.decode_head(base_features)['final_logits']
        hc_logits = hc_model.decode_head(hc_features)['final_logits']
        logit_error = max_abs(base_logits, hc_logits)

    hc_parameters = sum(
        parameter.numel()
        for parameter in hc_model.backbone.hc_units.parameters())
    hc_unit_count = len(hc_model.backbone.hc_units)
    if hc_unit_count != 28 or hc_parameters != 224:
        raise RuntimeError(
            f'Expected 28 HC units / 224 parameters, got '
            f'{hc_unit_count} / {hc_parameters}')
    print(f'HC units: {hc_unit_count}')
    print(f'HC parameters: {hc_parameters}')
    print(f'Encoder max-abs errors: {feature_errors}')
    print(f'Final-logit max-abs error: {logit_error}')

    if max(feature_errors + [logit_error]) > args.atol:
        raise RuntimeError(
            f'HC initialization is not residual-equivalent within {args.atol}')
    print('HC equivalence sanity check passed.')


if __name__ == '__main__':
    main()
