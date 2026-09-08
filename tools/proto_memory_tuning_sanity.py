"""CPU and resolved-config checks for three ProtoRoute memory tuning arms.

Executes the real head using the existing trunk/framework stubs. No GPU
backbone, FreqFusion kernel, dataset or multi-process DDP run is involved.
"""
import copy
import io
import math
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

from proto_variants_sanity import close, load_components, make

ROOT = Path(__file__).resolve().parents[1]
ARMS = (
    ('n0800', 'proto_n0_init', 800.0),
    ('slowmem', 'proto_momentum', .001),
    ('fastmem', 'proto_momentum', .1),
)
SUFFIX = '_r4_responsibility_ade20k_160k-512x512.py'


def check_configs():
    from mmengine.config import Config
    folder = ROOT / 'local_configs/offseg2/Base'
    control = Config.fromfile(
        str(folder / ('offsegccmiacs_protoroute' + SUFFIX)),
        import_custom_modules=False).to_dict()
    dirs = {control['work_dir']}
    for name, key, value in ARMS:
        stem = 'offsegccmiacs_protoroute_' + name + SUFFIX
        cfg = Config.fromfile(str(folder / stem),
                              import_custom_modules=False).to_dict()
        assert cfg['model']['decode_head'][key] == value
        assert cfg['model']['decode_head']['type'] == 'OffSegCCMIACSProtoRoute'
        assert cfg['work_dir'] == './work_dirs/' + Path(stem).stem
        assert cfg['work_dir'] not in dirs
        dirs.add(cfg['work_dir'])
        normalised = copy.deepcopy(cfg)
        normalised['model']['decode_head'][key] = control['model']['decode_head'][key]
        normalised['work_dir'] = control['work_dir']
        assert normalised == control, name
        assert cfg['model']['backbone']['type'] == 'efficientformerv2_s2_feat'
        assert cfg['randomness']['seed'] == 1370346084
        assert cfg['train_cfg']['max_iters'] == 160000
        assert cfg['train_cfg']['val_interval'] == 8000
        assert cfg['train_dataloader']['batch_size'] == 4
        assert cfg['load_from'] is None and cfg['resume'] is False
        assert cfg['default_hooks']['checkpoint']['save_best'] == 'mIoU'
        print('PASS complete resolved config:', name)
    print('ALL MEMORY TUNING CONFIG CHECKS PASSED')


def loss(model, x, labels):
    result = model.loss_by_feat(model(x), labels)
    assert {k for k in result if k.startswith('loss')} == {'loss_stage1', 'loss_ccm'}
    assert all(torch.isfinite(v).all() for v in result.values())
    return result['loss_stage1'] + result['loss_ccm']


def main():
    torch.set_num_threads(1)
    proto, variants, _ = load_components()
    cls = variants.OffSegCCMIACSProtoRoute
    for value in (1e-8, .05, 1., 50., 200., 700.):
        assert proto._inverse_softplus(value) == math.log(math.expm1(value))
    assert proto._inverse_softplus(800.) == 800.
    close(F.softplus(torch.tensor(proto._inverse_softplus(800.))),
          torch.tensor(800.))
    print('PASS large-scale initialisation without overflow; previous arithmetic unchanged')

    torch.manual_seed(71)
    x = torch.randn(2, 12, 4, 4)
    labels = torch.randint(0, 5, (2, 4, 4))
    for name, key, value in ARMS:
        control = make(cls)
        control_rng = torch.get_rng_state().clone()
        model = make(cls, **{key: value})
        assert torch.equal(control_rng, torch.get_rng_state())
        for k, v in control.state_dict().items():
            if not (key == 'proto_n0_init' and k == 'proto_n0_raw'):
                assert torch.equal(v, model.state_dict()[k]), k
        assert sum(p.numel() for p in model.parameters()) == sum(
            p.numel() for p in control.parameters())
        for xx in (x, x * .5):
            a, b = control(xx), model(xx)
            assert torch.equal(a['final_logits'], b['final_logits'])
            assert torch.equal(a['stage1_logits'], b['stage1_logits'])
            assert b['proto_lambda'] == 0
        assert model(x)['proto_lambda'] > 0
    print('PASS common initialisation, parameter count, exact warmup and activation')

    for rate in (.001, .01, .1):
        model = make(cls, proto_momentum=rate)
        support = torch.tensor([[2., 0., 2., 0., 0.],
                                [4., 0., 0., 0., 0.]])
        model._update_prototypes(torch.ones(2, 5, 12), support)
        close(model.prototypes[0], torch.ones(12))
        close(model.prototypes[2], torch.ones(12))
        for step in range(1, 11):
            model._update_prototypes(torch.zeros(2, 5, 12), support)
            close(model.prototypes[0], torch.full((12,), (1. - rate) ** step))
        assert torch.equal(model.prototypes[1], torch.zeros(12))
        assert model.proto_steps == 11
        assert torch.equal(model.proto_seen, torch.tensor([11., 0., 11., 0., 0.]))
        before = model.prototypes.clone()
        model._update_prototypes(torch.randn(2, 5, 12), torch.zeros_like(support))
        assert torch.equal(model.prototypes, before)
    print('PASS EMA rates, geometric decay, direct first observation, absent-class protection')

    control = make(cls).eval()
    stronger = make(cls, proto_n0_init=800).eval()
    for model in (control, stronger):
        model.proto_seen.fill_(1)
    masks = torch.zeros(2, 5, 1000)  # support=200 for each class
    centres = torch.randn(2, 5, 12)
    _, a = control._blend_prototypes(masks, centres)
    _, b = stronger._blend_prototypes(masks, centres)
    close(a['proto_lambda'], torch.tensor(.5))
    close(b['proto_lambda'], torch.tensor(.8))
    print('PASS read strength: lambda .50 -> .80 at support 200')

    for name, key, value in ARMS:
        kwargs = {key: value}
        model = make(cls, warmup=0, **kwargs)
        optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4)
        for xx in (x, x * .5):
            optimizer.zero_grad(set_to_none=True)
            loss(model, xx, labels).backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all()
                       for p in model.parameters())
            assert model.proto_n0_raw.grad.abs() > 0
            optimizer.step()
        model.eval()
        before = {k: v.clone() for k, v in model.named_buffers()}
        with torch.no_grad():
            expected = model(x)['final_logits']
            model(x * 2)
        assert all(torch.equal(v, before[k]) for k, v in model.named_buffers())
        # Same saved state has identical inference independent of EMA rate.
        rate_control = make(cls).eval()
        rate_control.load_state_dict(model.state_dict())
        with torch.no_grad():
            assert torch.equal(expected, rate_control(x)['final_logits'])
        stream = io.BytesIO()
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict()), stream)
        stream.seek(0)
        state = torch.load(stream, weights_only=True)
        restored = make(cls, warmup=0, **kwargs)
        restored.load_state_dict(state['model'])
        opt_restored = torch.optim.AdamW(restored.parameters(), lr=6e-4)
        opt_restored.load_state_dict(state['optimizer'])
        for m, opt in ((model, optimizer), (restored, opt_restored)):
            m.train()
            opt.zero_grad(set_to_none=True)
            loss(m, x * .8, labels).backward()
            opt.step()
        assert all(torch.equal(v, restored.state_dict()[k])
                   for k, v in model.state_dict().items())
    print('PASS finite two-CE training, eval freeze, same-state inference and optimizer resume')
    print('ALL PROTO MEMORY TUNING CHECKS PASSED')


if __name__ == '__main__':
    if '--configs-only' in sys.argv:
        check_configs()
    else:
        main()
