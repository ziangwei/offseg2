"""Validate target-dataset recipes and real Route head at 19/171 classes."""
import copy
from pathlib import Path
import sys
import torch
from proto_variants_sanity import load_components


def configs():
    from mmengine.config import Config
    folder = Path(__file__).resolve().parents[1] / 'local_configs/offseg2/Base'
    source = Config.fromfile(str(folder / 'offsegccmiacs_protoroute_r4_responsibility_ade20k_160k-512x512.py'), import_custom_modules=False).to_dict()
    for dataset, classes, size, steps, interval, batch, seed in (
            ('stuff164k', 171, 512, 80000, 4000, 4, 2000199364),
            ('cityscapes', 19, 1024, 160000, 8000, 2, 1370346084)):
        suffix = f'{dataset}_{steps // 1000}k-{size}x{size}.py'
        cfg = Config.fromfile(str(folder / ('offsegccmiacs_protoroute_r4_responsibility_' + suffix)), import_custom_modules=False).to_dict()
        h = cfg['model']['decode_head']
        assert h['type'] == 'OffSegCCMIACSProtoRoute' and h['num_classes'] == classes
        source_head = copy.deepcopy(source['model']['decode_head'])
        source_head['num_classes'] = classes
        assert h == source_head, (dataset, {k: (h.get(k), source_head.get(k)) for k in set(h) | set(source_head) if h.get(k) != source_head.get(k)})
        assert cfg['model']['backbone']['type'] == 'efficientformerv2_s2_feat'
        assert cfg['model']['backbone']['init_cfg'] == source['model']['backbone']['init_cfg']
        assert tuple(cfg['model']['data_preprocessor']['size']) == (size, size)
        assert cfg['train_dataloader']['batch_size'] == batch
        assert cfg['train_cfg']['max_iters'] == steps and cfg['train_cfg']['val_interval'] == interval
        assert cfg['param_scheduler'][-1]['end'] == steps
        assert cfg['randomness']['seed'] == seed
        assert cfg['load_from'] is None and cfg['resume'] is False
        hook = cfg['default_hooks']['checkpoint']
        assert hook['interval'] == interval and hook['save_best'] == 'mIoU'
        assert hook['max_keep_ckpts'] == 2 and hook['save_last']
        assert cfg['optim_wrapper']['paramwise_cfg']['custom_keys']['proto_n0_raw'] == dict(lr_mult=10., decay_mult=0.)
        control_prefix = 'offsegccmiacs_proto_r4_responsibility_' if dataset == 'stuff164k' else 'offsegccmiacs_r4_responsibility_'
        control = Config.fromfile(str(folder / (control_prefix + suffix)), import_custom_modules=False).to_dict()
        for k in ('train_dataloader', 'val_dataloader', 'test_dataloader', 'param_scheduler', 'train_cfg', 'val_evaluator', 'test_evaluator'):
            assert cfg[k] == control[k], (dataset, k)
        assert cfg['model']['test_cfg'] == control['model']['test_cfg']
        assert cfg['work_dir'] not in (source['work_dir'], control['work_dir'])
        print('PASS resolved original Route head + target recipe:', dataset, 'data root:', cfg['train_dataloader']['dataset']['data_root'])


def main():
    torch.set_num_threads(1)
    _, variants, _ = load_components()
    for classes in (19, 171):
        def build():
            return variants.OffSegCCMIACSProtoRoute(
                in_channels=[12], new_channels=[12], channels=12, num_classes=classes,
                ccm_rank=4, ccm_hidden=8, iacs_assignment='posterior', proto_warmup=2)
        model = build()
        x = torch.randn(2, 12, 16, 16)
        labels = torch.randint(0, classes, (2, 16, 16))
        a = model(x)
        assert a['final_logits'].shape == (2, classes, 16, 16)
        assert a['proto_lambda'] == 0
        b = model(x)
        assert b['proto_lambda'] > 0 and (model.proto_seen > 0).any()
        assert model.prototypes.shape == (classes, 12)
        assert model.acs.raw_basis.shape == (classes, 12, 4)
        losses = model.loss_by_feat(b, labels)
        assert {k for k in losses if k.startswith('loss')} == {'loss_ccm', 'loss_stage1'}
        (losses['loss_ccm'] + losses['loss_stage1']).backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        torch.optim.AdamW(model.parameters(), lr=6e-4).step()
        model.eval()
        restored = build().eval()
        restored.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
        before = {k: v.clone() for k, v in model.named_buffers()}
        with torch.no_grad():
            torch.testing.assert_close(model(x)['final_logits'], restored(x)['final_logits'], rtol=0, atol=0)
        assert all(torch.equal(v, before[k]) for k, v in model.named_buffers())
        print('PASS target class count, warmup, memory, original losses, finite gradients and restored eval:', classes)


if __name__ == '__main__':
    configs() if '--configs-only' in sys.argv else main()
