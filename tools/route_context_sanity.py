"""Check two context structures with real heads and lightweight framework stubs."""
import copy
import importlib.util
from pathlib import Path
import sys
import torch
from proto_variants_sanity import load_components, make

ROOT = Path(__file__).resolve().parents[1]


def configs():
    from mmengine.config import Config
    def read(name):
        return Config.fromfile(str(ROOT / 'local_configs/offseg2' / name),
                               import_custom_modules=False).to_dict()
    suffix = '_r4_responsibility_ade20k_160k-512x512.py'
    control = read('Base/offsegccmiacs_protoroute' + suffix)
    dirs = set()
    for kind in ('spatial', 'relation'):
        cfg = read('Base/offsegccmiacs_protoroute_' + kind + suffix)
        assert cfg['model']['decode_head']['type'] == 'OffSegCCMIACSProtoRoute' + kind.title()
        dirs.add(cfg['work_dir'])
        normal = copy.deepcopy(cfg)
        normal['model']['decode_head']['type'] = control['model']['decode_head']['type']
        normal['custom_imports'] = control['custom_imports']
        normal['work_dir'] = control['work_dir']
        assert normal == control
        assert cfg['randomness']['seed'] == 1370346084
        assert cfg['train_cfg']['max_iters'] == 160000
        assert cfg['load_from'] is None and cfg['resume'] is False
        assert cfg['model']['decode_head']['channels'] == 256
    suffix = '_r4_responsibility_stuff164k_80k-512x512.py'
    tiny = read('Tiny/offsegccmiacs_protoroute' + suffix)
    parent = read('Tiny/offsegccmiacs_proto' + suffix)
    dirs.add(tiny['work_dir'])
    assert len(dirs) == 3
    normal = copy.deepcopy(tiny)
    for key in ('custom_imports', 'work_dir', 'randomness', 'load_from', 'resume'):
        if key in parent:
            normal[key] = parent[key]
        else:
            normal.pop(key, None)
    normal['model']['decode_head']['type'] = parent['model']['decode_head']['type']
    assert normal == parent
    assert tiny['model']['decode_head']['num_classes'] == 171
    assert tiny['model']['decode_head']['type'] == 'OffSegCCMIACSProtoRoute'
    assert tiny['model']['backbone']['type'] == 'efficientformerv2_s1_feat'
    assert tiny['train_cfg']['max_iters'] == 80000
    assert tiny['train_dataloader']['batch_size'] == 4
    assert tiny['randomness']['seed'] == 2000199364
    assert tiny['load_from'] is None and tiny['resume'] is False
    print('PASS resolved recipes: independent ADE structures and original Tiny/Stuff Route')


def main():
    torch.set_num_threads(1)
    _, variants, _ = load_components()
    name = 'mmseg.models.decode_heads.OffSegRouteContext'
    spec = importlib.util.spec_from_file_location(name, ROOT / 'mmseg/models/decode_heads/OffSegRouteContext.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    x, labels = torch.randn(2, 12, 3, 5), torch.randint(0, 5, (2, 3, 5))
    for kind in ('spatial', 'relation'):
        cls = getattr(module, 'OffSegCCMIACSProtoRoute' + kind.title())
        control = make(variants.OffSegCCMIACSProtoRoute, warmup=2)
        model = make(cls, warmup=2)
        extra = sum(p.numel() for p in model.parameters()) - sum(p.numel() for p in control.parameters())
        assert extra == (108 if kind == 'spatial' else 1560)
        # Activate the existing generator only in this fixture: otherwise its
        # original zero last layer could hide incorrect context implementation.
        with torch.no_grad():
            control.ccm.ccm_g[-1].weight.normal_(std=.02)
        state = {('ccm.core.' + k[4:] if k.startswith('ccm.') else k): v
                 for k, v in control.state_dict().items()}
        missing = model.load_state_dict(state, strict=False)
        assert not missing.unexpected_keys
        assert all(k.startswith('ccm.') and not k.startswith('ccm.core.') for k in missing.missing_keys)
        captured = {}
        def hook(_, inputs):
            captured['inputs'] = inputs
        handle = model.ccm.register_forward_pre_hook(hook)
        for _ in range(3):
            a, b = control(x), model(x)
            for key in ('stage1_logits', 'final_logits'):
                torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
        assert b['proto_lambda'] > 0
        assert not captured['inputs'][1].requires_grad
        assert not captured['inputs'][2].requires_grad
        handle.remove()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        def step(m, optimizer):
            m.train()
            optimizer.zero_grad(set_to_none=True)
            loss = m.loss_by_feat(m(x), labels)
            assert {k for k in loss if k.startswith('loss')} == {'loss_stage1', 'loss_ccm'}
            (loss['loss_stage1'] + loss['loss_ccm']).backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
            optimizer.step()
        step(model, opt)
        learned = model.ccm.local if kind == 'spatial' else model.ccm.output
        assert learned.weight.grad.abs().sum() > 0
        step(model, opt)
        if kind == 'relation':
            for layer in (model.ccm.query, model.ccm.key, model.ccm.value):
                assert layer.weight.grad.abs().sum() > 0
        restored = make(cls, warmup=2)
        restored.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
        opt2 = torch.optim.AdamW(restored.parameters(), lr=1e-3)
        opt2.load_state_dict(copy.deepcopy(opt.state_dict()))
        step(model, opt)
        step(restored, opt2)
        for k, v in model.state_dict().items():
            torch.testing.assert_close(v, restored.state_dict()[k], rtol=0, atol=0)
        model.eval()
        before = {k: v.clone() for k, v in model.named_buffers()}
        with torch.no_grad():
            model(torch.randn(1, 12, 5, 2))
        assert all(torch.equal(v, before[k]) for k, v in model.named_buffers())
        assert model.ccm.spatial_shape == (5, 2)
        print('PASS', kind, 'zero residual parity, active memory, new gradients, two CE, resume and frozen eval')
    # Relation messages are equivariant to class ordering, not fixed class IDs.
    wrapper = model.ccm
    feat, centres, logits = torch.randn(2, 15, 12), torch.randn(2, 5, 12), torch.randn(2, 5, 15)
    perm = torch.tensor([3, 1, 4, 0, 2])
    torch.testing.assert_close(wrapper(feat, centres, logits)[0],
                               wrapper(feat, centres[:, perm], logits[:, perm])[0])
    # A local impulse must reach a neighbour in the actual rectangular grid.
    spatial = make(module.OffSegCCMIACSProtoRouteSpatial).ccm
    with torch.no_grad():
        spatial.local.weight.fill_(1.)
    grid = torch.zeros(1, 12, 3, 5)
    grid[:, :, 1, 2] = 1.
    output = spatial.local(grid)
    assert output[0, 0, 1, 1] == 1 and output[0, 0, 1, 0] == 0
    print('PASS class permutation and local receptive field')


if __name__ == '__main__':
    configs() if '--configs-only' in sys.argv else main()
