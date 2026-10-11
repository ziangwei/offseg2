"""Numerical Route ablations; --configs-only uses real MMEngine inheritance."""
import copy
import importlib.util
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def configs():
    sys.path.insert(0, str(ROOT / 'tools/slurm'))
    import submit
    from mmengine.config import Config
    def read(path):
        return Config.fromfile(str(ROOT / path), import_custom_modules=False).to_dict()
    control = read('local_configs/offseg2/Base/offsegccmiacs_protoroute_r4_responsibility_ade20k_160k-512x512.py')
    dirs = set()
    modes = dict(nomem='no_memory', noccm='no_ccm', noresidual='no_residual',
                 staticacs='static_acs', fixedmix='fixed_lambda')
    assert submit.select_jobs('ablations') == submit.select_jobs('round7') == submit.ABLATIONS
    assert len(submit.ABLATIONS) == 6
    for name in submit.ABLATIONS:
        item = submit.CATALOG[name]
        cfg = read(item['config'])
        arm = name.split('_')[1]
        normal = copy.deepcopy(cfg)
        assert cfg['randomness'] == control['randomness'] == dict(seed=1370346084, deterministic=False)
        assert cfg['work_dir'] not in dirs
        dirs.add(cfg['work_dir'])
        normal['work_dir'] = control['work_dir']
        head = normal['model']['decode_head']
        if arm == 'spatialstats':
            assert head['iacs_assignment'] == 'spatial'
            head['iacs_assignment'] = 'posterior'
        else:
            assert head.pop('ablation') == modes[arm]
            assert head['type'] == 'OffSegProtoRouteAblation'
            head['type'] = control['model']['decode_head']['type']
            normal['custom_imports'] = control['custom_imports']
            if arm == 'fixedmix':
                assert head.pop('proto_fixed_lambda') == .5
        keys = normal['optim_wrapper']['paramwise_cfg']['custom_keys']
        for key in ('acs.mix_logit', 'proto_n0_raw'):
            if key not in keys:
                keys[key] = control['optim_wrapper']['paramwise_cfg']['custom_keys'][key]
        assert normal == control, name
        assert cfg['train_cfg']['max_iters'] == 160000
        assert cfg['default_hooks']['checkpoint']['max_keep_ckpts'] == 2
        assert cfg['default_hooks']['checkpoint']['save_best'] == 'mIoU'
        script = (ROOT / item['script']).read_text()
        assert '#SBATCH --time=36:00:00' in script
        assert item['config'] in script and '#SBATCH --gres=gpu:4' in script
    print('PASS six resolved recipes: exact protocol, only intended deletions, fixed seed, 36h/4GPU/2+best')


def tensors():
    import torch
    from proto_variants_sanity import load_components, make, close
    torch.set_num_threads(1)
    proto, variants, _ = load_components()
    spec = importlib.util.spec_from_file_location(
        'mmseg.models.decode_heads.OffSegRouteAblations',
        ROOT / 'mmseg/models/decode_heads/OffSegRouteAblations.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    Route = variants.OffSegCCMIACSProtoRoute
    Ablation = module.OffSegProtoRouteAblation
    acs = sys.modules['mmseg.models.decode_heads.OffSegACS']
    torch.manual_seed(719)
    x, labels = torch.randn(2, 12, 3, 7), torch.randint(0, 5, (2, 3, 7))
    modes = ('no_memory', 'no_ccm', 'no_residual', 'static_acs', 'fixed_lambda')
    for mode in modes:
        kwargs = dict(proto_fixed_lambda=.5) if mode == 'fixed_lambda' else {}
        model = make(Ablation, ablation=mode, **kwargs)
        rng = torch.get_rng_state().clone()
        control = make(Route)
        assert torch.equal(torch.get_rng_state(), rng)
        for name, value in model.state_dict().items():
            assert torch.equal(value, control.state_dict()[name]), name
        assert sum(p.numel() for p in model.parameters()) < sum(p.numel() for p in control.parameters())
        if mode == 'no_ccm':
            assert not list(model.ccm.parameters())
            with patch.object(control.ccm, 'forward', side_effect=module._IdentityCCM().forward):
                expected = control(x)
        elif mode == 'no_residual':
            assert not hasattr(model, 'acs')
            with patch.object(control, '_subspace_correction',
                              side_effect=lambda f, c, l, **kw: (torch.zeros_like(l), {})):
                expected = control(x)
        elif mode == 'static_acs':
            assert type(model.acs) is acs.AffineClassSubspace
            assert not hasattr(model.acs, 'mix_logit')
            with torch.no_grad():
                control.acs.mix_logit.fill_(-float('inf'))
            expected = control(x)
        elif mode == 'fixed_lambda':
            assert 'proto_n0_raw' not in dict(model.named_parameters())
            control.proto_fixed_lambda = .5
            expected = control(x)
        else:
            assert not hasattr(model, 'prototypes')
            with patch.object(control, '_blend_prototypes',
                              side_effect=lambda masks, centres: (centres, {})):
                expected = control(x)
        actual = model(x)
        close(actual['final_logits'], expected['final_logits'])
        assert torch.equal(actual['stage1_logits'], expected['stage1_logits'])
        # Traverse first-write/warmup/active memory states, then train both CEs.
        for _ in range(4):
            actual = model(x)
        losses = model.loss_by_feat(actual, labels)
        assert {k for k in losses if k.startswith('loss')} == {'loss_ccm', 'loss_stage1'}
        if mode in ('no_residual', 'static_acs'):
            assert not any(k.startswith('acc_iacs') for k in losses)
        sum(v for k, v in losses.items() if k.startswith('loss')).backward()
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        state = copy.deepcopy(model.state_dict())
        restored = make(Ablation, ablation=mode, **kwargs)
        restored.load_state_dict(state)
        model.eval(); restored.eval()
        before = copy.deepcopy(model.state_dict())
        with torch.no_grad():
            close(model(x)['final_logits'], restored(x)['final_logits'])
        for name, value in model.state_dict().items():
            assert torch.equal(value, before[name]), 'eval updated state: ' + name
        print('PASS', mode, ': deletion equivalence, common initial RNG/weights, active memory, CE gradients, checkpoint/eval')
    # Spatial assignment is the existing real IACS switch, not neighbourhood CCM.
    model = make(Route)
    model.acs.assignment = 'spatial'
    for _ in range(4):
        result = model(x)
    losses = model.loss_by_feat(result, labels)
    sum(v for k, v in losses.items() if k.startswith('loss')).backward()
    assert model.acs.assignment == 'spatial'
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for n, p in model.named_parameters())
    logits = torch.randn(2, 21, 5)
    torch.testing.assert_close(model.acs.assignment_weights(logits), logits.softmax(dim=1))
    # No-memory Route equals the old no-memory forward with identical weights.
    model = make(Ablation, ablation='no_memory').eval()
    old = make(acs.OffSegCCMIACS).eval()
    old.load_state_dict(model.state_dict())
    close(model(x)['final_logits'], old(x)['final_logits'])
    # Fixed mix is independent of class support after activation, unseen classes excluded.
    model = make(Ablation, ablation='fixed_lambda', proto_fixed_lambda=.5, warmup=0).eval()
    model.proto_seen.fill_(1); model.proto_seen[-1] = 0
    model.prototypes.normal_()
    centres = torch.randn(2, 5, 12)
    for masks in (torch.zeros(2, 5, 21), torch.randn(2, 5, 21) * 10):
        blend, _ = model._blend_prototypes(masks, centres)
        close(blend[:, :-1], .5 * centres[:, :-1] + .5 * model.prototypes[None, :-1])
        close(blend[:, -1], centres[:, -1])
    print('PASS existing spatial assignment, no-memory equivalence, fixed support-independent .5 blend/unseen guard')


if __name__ == '__main__':
    configs() if '--configs-only' in sys.argv else tensors()
