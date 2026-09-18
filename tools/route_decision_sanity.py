"""CPU tensor/config/DDP checks for the five second-batch Route experiments."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import torch
import torch.distributed as dist
from proto_variants_sanity import load_components, make

ROOT = Path(__file__).resolve().parents[1]
KINDS = [('modebank', 'ModeBank'), ('centretilt', 'CentreTilt'),
         ('blockmetric', 'BlockMetric'), ('contrastmetric', 'ContrastMetric'),
         ('softenergy', 'SoftEnergy')]


def load():
    _, variants, _ = load_components()
    name = 'mmseg.models.decode_heads.OffSegRouteDecision'
    spec = importlib.util.spec_from_file_location(name, ROOT / 'mmseg/models/decode_heads/OffSegRouteDecision.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module, variants


def configs():
    from mmengine.config import Config
    base = ROOT / 'local_configs/offseg2/Base'
    control = Config.fromfile(str(base / 'offsegccmiacs_protoroute_r4_responsibility_ade20k_160k-512x512.py'), import_custom_modules=False).to_dict()
    manifest = json.loads((ROOT / 'tools/slurm/experiments.json').read_text())
    for kind, suffix in KINDS:
        cfg = Config.fromfile(str(ROOT / manifest[kind + '_b_ade']['config']), import_custom_modules=False).to_dict()
        assert cfg['model']['decode_head']['type'] == 'OffSegCCMIACSProtoRoute' + suffix
        assert cfg['work_dir'].endswith(Path(manifest[kind + '_b_ade']['config']).stem)
        normal = copy.deepcopy(cfg)
        for key in ('work_dir', 'custom_imports'):
            normal[key] = control[key]
        normal['model']['decode_head']['type'] = control['model']['decode_head']['type']
        if kind == 'centretilt':
            keys = normal['optim_wrapper']['paramwise_cfg']['custom_keys']
            assert keys.pop('acs.core.mix_logit') == keys['acs.mix_logit']
        assert normal == control, kind
        checkpoint = cfg['default_hooks']['checkpoint']
        assert checkpoint['max_keep_ckpts'] == 2 and checkpoint['save_best'] == 'mIoU' and checkpoint['save_last']
    print('PASS five configs: original protocol/seed, optimizer mapping, two rolling checkpoints + best')


def tensors():
    torch.set_num_threads(1)
    module, variants = load()
    x, labels = torch.randn(2, 12, 3, 5), torch.randint(0, 5, (2, 3, 5))
    for kind, suffix in KINDS:
        cls = getattr(module, 'OffSegCCMIACSProtoRoute' + suffix)
        model = make(cls, warmup=2)
        control = make(variants.OffSegCCMIACSProtoRoute, warmup=2)
        with torch.no_grad():
            control.ccm.ccm_g[-1].weight.normal_(std=.02)
        def rename(key):
            if kind in ('blockmetric', 'contrastmetric') and key.startswith('ccm.'):
                return 'ccm.core.' + key[4:]
            if kind == 'centretilt' and key.startswith('acs.'):
                return 'acs.core.' + key[4:]
            return key
        missing = model.load_state_dict({rename(k): v for k, v in control.state_dict().items()}, strict=False)
        assert not missing.unexpected_keys
        if kind in ('blockmetric', 'centretilt'):
            for _ in range(3):
                a, b = control(x), model(x)
                torch.testing.assert_close(a['final_logits'], b['final_logits'], rtol=1e-5, atol=1e-6)
            assert b['proto_lambda'] > 0
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        def step(net, opt):
            net.train()
            opt.zero_grad(set_to_none=True)
            out = net(x)
            loss = net.loss_by_feat(out, labels)
            assert {k for k in loss if k.startswith('loss')} == {'loss_ccm', 'loss_stage1'}
            (loss['loss_ccm'] + loss['loss_stage1']).backward()
            bad = [n for n, p in net.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
            assert not bad, bad
            opt.step()
        for _ in range(3):
            step(model, optimizer)
        if kind == 'centretilt':
            assert model.acs.tilt.grad.abs().sum() > 0
            basis = model.acs.basis(torch.randn(2, 5, 12))
            torch.testing.assert_close(basis.transpose(-1, -2) @ basis,
                                       torch.eye(4).expand(2, 5, 4, 4), atol=1e-5, rtol=1e-5)
        if kind == 'blockmetric':
            assert model.ccm.offdiag.weight.grad.abs().sum() > 0
        restored = make(cls, warmup=2)
        restored.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
        opt2 = torch.optim.AdamW(restored.parameters(), lr=.001)
        opt2.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        step(model, optimizer)
        step(restored, opt2)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)
        model.eval()
        before = {k: v.clone() for k, v in model.named_buffers()}
        with torch.no_grad():
            out = model(torch.randn(1, 12, 5, 2))
        assert out['final_logits'].shape == (1, 5, 5, 2)
        assert all(torch.equal(v, before[k]) for k, v in model.named_buffers())
        added = sum(p.numel() for p in model.parameters()) - sum(p.numel() for p in control.parameters())
        print('PASS', kind, 'finite gradients, two losses, exact resume, frozen eval; extra tiny-test params:', added)
    # Concave redistribution preserves the class-weighted mean and compresses
    # the relative difference between a large and a small positive bonus.
    score = torch.tensor([[[1., 0.], [4., 0.], [100., 0.]]], requires_grad=True)
    weight = torch.full_like(score, 1 / 3)
    changed = module.redistribute_energy(score, weight)
    torch.testing.assert_close((weight * changed).sum(1), (weight * score).sum(1))
    assert changed[0, 2, 0] / changed[0, 0, 0] < 100
    assert torch.equal(changed[..., 1], score[..., 1])
    changed.sum().backward()
    assert torch.isfinite(score.grad).all()
    # Explicit affine-contrast CCM oracle with an active generator.
    wrap = make(module.OffSegCCMIACSProtoRouteContrastMetric).ccm
    with torch.no_grad():
        wrap.core.ccm_g[-1].weight.normal_(std=.1)
    feat, centres, logits = torch.randn(2, 10, 12), torch.randn(2, 5, 12), torch.randn(2, 5, 10)
    posterior = logits.transpose(1, 2).softmax(-1)
    posterior = posterior * wrap.core._nucleus(posterior)
    posterior = posterior / posterior.sum(-1, keepdim=True)
    z = posterior @ centres
    gain = wrap.core.gain_scale * wrap.core.ccm_g(torch.cat([z, feat], -1)).tanh()
    torch.testing.assert_close(wrap(feat, centres, logits)[0], feat + wrap.core.ccm_u(gain * wrap.core.ccm_v(feat - z)))
    # Two clearly separated image centres must remain two modes and read back.
    bank = make(module.OffSegCCMIACSProtoRouteModeBank, warmup=0)
    values = torch.ones(2, 5, 12)
    values[1] *= -1
    support = torch.full((2, 5), 3.)
    bank._update_prototypes(values, support)
    torch.testing.assert_close(bank.mode_bank[:, 0], values[0])
    torch.testing.assert_close(bank.mode_bank[:, 1], values[1])
    bank.eval()
    blended, state = bank._blend_prototypes(torch.zeros(2, 5, 15), values)
    torch.testing.assert_close(blended, values)
    assert state['mode_second'] == .5 and state['mode_ready'] == 1
    # Unseen classes and identical observations are valid, finite fallbacks.
    empty = make(module.OffSegCCMIACSProtoRouteModeBank, warmup=0).eval()
    output, state = empty._blend_prototypes(torch.zeros(2, 5, 15), values)
    torch.testing.assert_close(output, values)
    print('PASS scoring/contrast/mode oracles, zero-energy and unseen-class cases')


def distributed_worker(rank, url):
    torch.set_num_threads(1)
    module, _ = load()
    # Compute a concatenated-batch reference before starting the process group.
    generator = torch.Generator().manual_seed(123)
    batches = [torch.randn(3, 5, 12, generator=generator) for _ in range(3)]
    expected = make(module.OffSegCCMIACSProtoRouteModeBank)
    support = torch.full((3, 5), 3.)
    support[:, 4] = 0.  # This class must never be initialised.
    for values in batches:
        expected._update_prototypes(values, support)
    dist.init_process_group('gloo', init_method=url, rank=rank, world_size=2)
    try:
        model = make(module.OffSegCCMIACSProtoRouteModeBank)
        selection = slice(0, 2) if rank == 0 else slice(2, 3)
        for values in batches:
            model._update_prototypes(values[selection], support[selection])
        for key in ('mode_bank', 'mode_seen', 'prototypes', 'proto_seen', 'proto_steps'):
            torch.testing.assert_close(getattr(model, key), getattr(expected, key), rtol=1e-5, atol=1e-6)
        assert not model.mode_seen[4].any()
        print('PASS distributed rank', rank, 'uneven batches, global seeding and all-reduced mode updates')
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    if '--configs-only' in sys.argv:
        configs()
    elif '--distributed' in sys.argv:
        import torch.multiprocessing as mp
        (ROOT / 'tmp').mkdir(exist_ok=True)
        directory = Path(tempfile.mkdtemp(prefix='route_modes_ddp_', dir=ROOT / 'tmp'))
        mp.spawn(distributed_worker, args=((directory / 'rendezvous').resolve().as_uri(),), nprocs=2, join=True)
    else:
        tensors()
