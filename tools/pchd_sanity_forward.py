"""Check full-model PCHD-Hyper/fixed equivalence before server training."""

import argparse

import torch
from mmengine.config import Config

from mmseg.models.decode_heads.PARSegPCHD import PARSegPCHD  # noqa: F401
from mmseg.registry import MODELS
from mmseg.utils import register_all_modules


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--hyper-config',
        default='local_configs/offseg2/Base/'
                'parseg3_pchd4_hyper_ade20k_160k-512x512.py')
    parser.add_argument(
        '--fixed-config',
        default='local_configs/offseg2/Base/'
                'parseg3_pchd4_fixed_ade20k_160k-512x512.py')
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
    hyper_cfg = Config.fromfile(args.hyper_config)
    fixed_cfg = Config.fromfile(args.fixed_config)

    hyper_model = MODELS.build(hyper_cfg.model)
    fixed_model = MODELS.build(fixed_cfg.model)
    incompatible = fixed_model.load_state_dict(
        hyper_model.state_dict(), strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f'Unexpected state mismatch: {incompatible}')

    hyper_model.to(args.device).eval()
    fixed_model.to(args.device).eval()
    image = torch.randn(
        1, 3, args.image_size, args.image_size, device=args.device)

    with torch.inference_mode():
        hyper_features = hyper_model.backbone(image)
        fixed_features = fixed_model.backbone(image)
        hyper_outputs = hyper_model.decode_head(hyper_features)
        fixed_outputs = fixed_model.decode_head(fixed_features)

    feature_errors = [
        max_abs(left, right)
        for left, right in zip(hyper_features, fixed_features)
    ]
    output_errors = {
        key: max_abs(hyper_outputs[key], fixed_outputs[key])
        for key in ('base_head_logits', 'refinement_head_logits',
                    'final_logits')
    }
    connection_parameters = sum(
        parameter.numel()
        for parameter in hyper_model.decode_head.pchd.connections.parameters())
    fixed_connection_parameters = sum(
        parameter.numel()
        for parameter in fixed_model.decode_head.pchd.connections.parameters())
    expected_parameters = 4 * 3 * 4 * 4
    if connection_parameters != expected_parameters:
        raise RuntimeError(
            f'Expected {expected_parameters} trainable connection scalars, '
            f'got {connection_parameters}')
    if fixed_connection_parameters != 0:
        raise RuntimeError(
            'Fixed control unexpectedly has trainable connection parameters')

    print(f'PCHD blocks: {len(hyper_model.decode_head.pchd.connections)}')
    print(f'Hyper connection parameters: {connection_parameters}')
    print(f'Fixed connection parameters: {fixed_connection_parameters}')
    print(f'Encoder max-abs errors: {feature_errors}')
    print(f'Decoder max-abs errors: {output_errors}')
    if max(feature_errors + list(output_errors.values())) > args.atol:
        raise RuntimeError(
            f'Hyper/fixed initialization differs by more than {args.atol}')
    print('PCHD hyper/fixed equivalence sanity check passed.')


if __name__ == '__main__':
    main()
