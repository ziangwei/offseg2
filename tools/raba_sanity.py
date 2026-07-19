"""GPU sanity check for the end-to-end RABA head.

Run this before launching the 160k job::

    python tools/raba_sanity.py \
        local_configs/offseg2/Base/raba_ade20k_160k-512x512.py
"""

import argparse
from pathlib import Path

import torch
from mmengine import Config
from mmengine.registry import init_default_scope
from mmengine.structures import PixelData
from mmengine.utils import import_modules_from_strings

from mmseg.registry import MODELS
from mmseg.structures import SegDataSample


def parse_args():
    parser = argparse.ArgumentParser(description='Sanity-check the RABA head')
    parser.add_argument(
        'config',
        nargs='?',
        default='local_configs/offseg2/Base/'
        'raba_ade20k_160k-512x512.py')
    parser.add_argument('--feature-size', type=int, default=32)
    parser.add_argument(
        '--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


def make_semantic_sample(size, device):
    """Four large regions plus ignore pixels; never exceeds query count."""
    target = torch.zeros((1, size, size), dtype=torch.long, device=device)
    half = size // 2
    target[:, :half, half:] = 1
    target[:, half:, :half] = 2
    target[:, half:, half:] = 3
    target[:, :2, :] = 255
    sample = SegDataSample(
        gt_sem_seg=PixelData(data=target),
        metainfo=dict(
            img_shape=(size, size),
            ori_shape=(size, size),
            pad_shape=(size, size)))
    return sample


def assert_gradient(parameter, name):
    if parameter.grad is None:
        raise AssertionError(f'{name} did not receive a gradient')
    if not torch.isfinite(parameter.grad).all():
        raise AssertionError(f'{name} has a non-finite gradient')
    if parameter.grad.abs().sum().item() == 0:
        raise AssertionError(f'{name} received an all-zero gradient')
    return float(parameter.grad.float().norm().detach())


def main():
    args = parse_args()
    config_path = Path(args.config).resolve()
    cfg = Config.fromfile(config_path)
    import_modules_from_strings(**cfg.custom_imports)
    init_default_scope('mmseg')

    from mmdet import __version__ as mmdet_version
    from mmdet.registry import MODELS as MMDET_MODELS
    from mmseg.models.decode_heads.region_attribute_bialign_head import (
        P3FreqFusionPixelDecoder, RegionAttributeBiAlignHead)

    assert MODELS.module_dict['P3FreqFusionPixelDecoder'] is \
        P3FreqFusionPixelDecoder
    assert MMDET_MODELS.module_dict['P3FreqFusionPixelDecoder'] is \
        P3FreqFusionPixelDecoder

    head_cfg = cfg.model.decode_head.copy()
    head = MODELS.build(head_cfg)
    if not isinstance(head, RegionAttributeBiAlignHead):
        raise AssertionError(f'unexpected head type: {type(head)}')
    if not isinstance(head.pixel_decoder, P3FreqFusionPixelDecoder):
        raise AssertionError(
            f'unexpected pixel decoder type: {type(head.pixel_decoder)}')
    head.init_weights()
    device = torch.device(args.device)
    head = head.to(device).train()

    base = args.feature_size
    if base < 16 or base % 8:
        raise ValueError('--feature-size must be >=16 and divisible by 8')
    inputs = [
        torch.randn(1, 32, base, base, device=device),
        torch.randn(1, 64, base // 2, base // 2, device=device),
        torch.randn(1, 144, base // 4, base // 4, device=device),
        torch.randn(1, 288, base // 8, base // 8, device=device),
    ]

    with torch.no_grad():
        cls_outputs, mask_outputs = head(inputs, [])
    expected_stages = head.num_transformer_decoder_layers + 1
    assert len(cls_outputs) == len(mask_outputs) == expected_stages
    assert cls_outputs[-1].shape == (1, head.num_queries,
                                     head.num_classes + 1)
    assert mask_outputs[-1].shape == (1, head.num_queries, base, base)

    image_size = base * 4
    sample = make_semantic_sample(image_size, device)
    losses = head.loss(inputs, [sample], None)
    if set(losses) != {'loss_cls', 'loss_mask', 'loss_dice'}:
        raise AssertionError(f'unexpected loss keys: {sorted(losses)}')
    total_loss = sum(losses.values())
    if not torch.isfinite(total_loss):
        raise AssertionError(f'non-finite total loss: {total_loss.item()}')
    total_loss.backward()
    gradient_parameters = {
        'query_feat': head.query_feat.weight,
        'mask_embed': head.mask_embed[0].weight,
        'mask_offset': head.offset_learning.cls_offset_proj.weight,
        'region_projection': head.region_projection.weight,
        'class_centers': head.region_classifier.class_centers,
        'attribute_deltas': head.region_classifier.attribute_deltas,
        'class_offset': head.region_classifier.class_offset_proj.weight,
        'region_offset': head.region_classifier.region_offset_proj.weight,
        'no_object': head.region_classifier.no_object_repr,
        'memory_s32': head.pixel_decoder.memory_projections[0].conv.weight,
        'memory_s16': head.pixel_decoder.memory_projections[1].conv.weight,
        'memory_s8': head.pixel_decoder.memory_projections[2].conv.weight,
        'freqfusion': head.pixel_decoder.freqfusions[0].content_encoder.weight,
    }
    gradient_norms = {
        name: assert_gradient(parameter, name)
        for name, parameter in gradient_parameters.items()
    }

    # Reproduce slide inference metadata: img_shape describes the current crop
    # while pad_shape may still describe the complete padded image.
    head.eval()
    slide_meta = dict(
        # A tuple verifies that slide detection uses test_cfg, not a fragile
        # metadata-type convention.
        img_shape=(image_size, image_size),
        pad_shape=(image_size + 32, image_size + 32),
        ori_shape=(image_size + 32, image_size + 32))
    with torch.no_grad():
        prediction = head.predict(inputs, [slide_meta], dict(mode='slide'))
    assert prediction.shape == (1, head.num_classes, image_size, image_size)

    print(f'RABA sanity passed | torch={torch.__version__} | '
          f'mmdet={mmdet_version} | device={device}')
    print('shapes:', cls_outputs[-1].shape, mask_outputs[-1].shape,
          prediction.shape)
    loss_values = {key: float(value.detach()) for key, value in losses.items()}
    print('losses:', loss_values)
    print('gradient_norms:', gradient_norms)


if __name__ == '__main__':
    main()
