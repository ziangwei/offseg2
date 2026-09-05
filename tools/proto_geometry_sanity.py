"""CPU numerical checks for static/local prototype geometry experiments.

Uses real Offset Learning, CCM, ACS/IACS and memory code with the existing
feature-trunk/framework stubs. Run --configs-only separately with MMEngine.
This is not a GPU backbone/FreqFusion or multi-process DDP training test.
"""

import copy
import importlib.util
import io
from pathlib import Path
import sys
from unittest.mock import patch

import torch
import torch.nn.functional as F

from proto_variants_sanity import close, load_components, make


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = (
    ('offsegccmacs_proto_r4_ade20k_160k-512x512.py', 'OffSegCCMACSProto'),
    ('offsegccmiacs_protolocal_r4_responsibility_ade20k_160k-512x512.py',
     'OffSegCCMIACSProtoLocal'),
)


def check_configs():
    from mmengine.config import Config

    root = ROOT / 'local_configs/offseg2/Base'
    suffix = '_r4_responsibility_ade20k_160k-512x512.py'
    control = Config.fromfile(
        str(root / ('offsegccmiacs_proto' + suffix)),
        import_custom_modules=False).to_dict()
    directories = set()
    for old in ('protoroute', 'protooffset', 'protologn0'):
        cfg = Config.fromfile(str(root / ('offsegccmiacs_' + old + suffix)),
                              import_custom_modules=False)
        directories.add(cfg.work_dir)
    for filename, head in CONFIGS:
        cfg = Config.fromfile(str(root / filename),
                              import_custom_modules=False).to_dict()
        assert cfg['model']['decode_head']['type'] == head
        assert cfg['custom_imports'] == dict(
            imports=['mmseg.models.decode_heads.OffSegProtoGeometry'],
            allow_failed_imports=False)
        assert cfg['randomness'] == dict(seed=1370346084, deterministic=False)
        assert cfg['load_from'] is None and cfg['resume'] is False
        assert cfg['train_cfg']['max_iters'] == 160000
        assert cfg['train_cfg']['val_interval'] == 8000
        assert cfg['train_dataloader']['batch_size'] == 4
        assert cfg['model']['backbone']['type'] == 'efficientformerv2_s2_feat'
        assert cfg['default_hooks']['checkpoint']['save_best'] == 'mIoU'
        assert cfg['work_dir'] == './work_dirs/' + Path(filename).stem
        assert cfg['work_dir'] not in directories
        directories.add(cfg['work_dir'])
        # Check the entire inherited protocol, not just selected fields.
        normalised = copy.deepcopy(cfg)
        normalised['model']['decode_head']['type'] = control['model']['decode_head']['type']
        normalised['custom_imports'] = control['custom_imports']
        normalised['work_dir'] = control['work_dir']
        normalised.pop('randomness')
        assert normalised == control, filename
        print('PASS resolved config:', head)
    assert len(directories) == 5
    print('ALL GEOMETRY CONFIG CHECKS PASSED')


def components():
    proto, _, registry = load_components()
    name = 'mmseg.models.decode_heads.OffSegProtoGeometry'
    spec = importlib.util.spec_from_file_location(
        name, ROOT / 'mmseg/models/decode_heads/OffSegProtoGeometry.py')
    geometry = importlib.util.module_from_spec(spec)
    sys.modules[name] = geometry
    spec.loader.exec_module(geometry)
    acs = sys.modules['mmseg.models.decode_heads.OffSegACS']
    return proto.OffSegCCMIACSProto, geometry, acs, registry


def check_initialisation(control_cls, arms, acs, registry):
    control = make(control_cls)
    rng = torch.get_rng_state().clone()
    count = sum(p.numel() for p in control.parameters())
    for index, cls in enumerate(arms):
        arm = make(cls)
        assert registry.classes[cls.__name__] is cls
        assert torch.equal(rng, torch.get_rng_state())
        assert sum(p.numel() for p in arm.parameters()) == count - (index == 0)
        for name, value in arm.state_dict().items():
            assert torch.equal(value, control.state_dict()[name]), name
        if index == 0:
            assert type(arm.acs) is acs.AffineClassSubspace
            assert not hasattr(arm.acs, 'mix_logit')
            assert not hasattr(arm.acs, 'image_metric')
    print('PASS common initial weights/RNG, registry and parameter counts')


def check_static(control_cls, static_cls, acs, x, labels):
    arm, reference = make(static_cls), make(control_cls)
    with torch.no_grad():
        reference.acs.mix_logit.fill_(-float('inf'))
    # Static must be equivalent to the controlled identity metric at each
    # memory state, including first write and the warmup activation step.
    for _ in range(4):
        expected = reference(x)
        with patch.object(acs.ImageAdaptiveAffineClassSubspace,
                          'image_metric', side_effect=AssertionError('dynamic pooling')):
            actual = arm(x)
            losses = arm.loss_by_feat(actual, labels)
        close(actual['final_logits'], expected['final_logits'])
        close(actual['acs_correction'], expected['acs_correction'])
        assert torch.equal(actual['stage1_logits'], expected['stage1_logits'])
        assert torch.equal(arm.prototypes, reference.prototypes)
        assert torch.equal(actual['proto_lambda'], expected['proto_lambda'])
        assert not any(key.startswith('acc_iacs') for key in losses)
        assert {k for k in losses if k.startswith('loss')} == {'loss_ccm', 'loss_stage1'}
    for option in ({'iacs_candidate_topk': 2}, {'iacs_persistent_spectrum': True}):
        try:
            make(static_cls, **option)
        except ValueError:
            pass
        else:
            raise AssertionError('unsupported geometry accepted')
    print('PASS static identity-metric equivalence, removed pooling and CE diagnostics')


def check_local(control_cls, local_cls, x):
    local, reference = make(local_cls), make(control_cls)
    for _ in range(2):
        actual, expected = local(x), reference(x)
        assert torch.equal(actual['final_logits'], expected['final_logits'])
        assert actual['proto_anchor_shift'].item() == 0
    assert local.proto_steps.item() == 2
    assert local(x)['proto_lambda'].item() > 0

    local = make(local_cls, warmup=0).eval()
    with torch.no_grad():
        local.proto_seen.fill_(1)
        local.prototypes.normal_()
        local.ccm.ccm_g[-1].weight.normal_(std=0.1)
    reference = make(control_cls, warmup=0).eval()
    reference.load_state_dict(local.state_dict())
    captured = {}
    handles = []
    for tag, model in (('local', local), ('control', reference)):
        def capture_ccm(module, args, output, tag=tag):
            captured[tag + '_ccm'] = (args, output)
        def capture_acs(module, args, tag=tag):
            captured[tag + '_acs'] = args
        handles.append(model.ccm.register_forward_hook(capture_ccm))
        handles.append(model.acs.register_forward_pre_hook(capture_acs))
    actual, expected = local(x), reference(x)
    for handle in handles:
        handle.remove()
    for a, b in zip(captured['local_ccm'][0], captured['control_ccm'][0]):
        assert torch.equal(a, b)
    for a, b in zip(captured['local_ccm'][1], captured['control_ccm'][1]):
        assert torch.equal(a, b)
    assert not captured['local_ccm'][0][1].requires_grad
    assert not captured['local_ccm'][0][2].requires_grad
    feat, anchor, logits = captured['local_acs']
    control_feat, blend, control_logits = captured['control_acs']
    masks, image_centres, _, _ = local._offset_learning_parts(x)
    close(anchor, image_centres)
    assert anchor.requires_grad
    assert torch.equal(feat, control_feat) and torch.equal(logits, control_logits)
    raw = feat @ blend.transpose(1, 2)
    close(logits, local.offset_learning.mask_norm(raw))
    correction, *_ = local.acs(feat, image_centres, logits)
    manual = local.offset_learning.mask_norm(raw + correction)
    manual = manual.transpose(1, 2).reshape_as(actual['final_logits'])
    close(actual['final_logits'], manual)
    assert torch.equal(actual['stage1_logits'], expected['stage1_logits'])
    assert torch.equal(actual['proto_lambda'], expected['proto_lambda'])
    assert (actual['final_logits'] - expected['final_logits']).abs().max() > 1e-6
    assert actual['proto_anchor_shift'].item() > 0
    shift = (blend - image_centres).norm(dim=-1)
    close(actual['proto_anchor_relative'],
          (shift / image_centres.norm(dim=-1).clamp_min(1e-6)).mean())

    # Explicit derivative for the residual origin, with the estimated
    # metric detached as in the actual model. This would fail on a detached
    # anchor or an extra (1-lambda) factor on its direct residual path.
    live_anchor = anchor.detach().clone().requires_grad_()
    fixed_feat = feat.detach()
    projection = local.acs.project_residual(fixed_feat, live_anchor)
    metric, *_ = local.acs.image_metric(projection, logits.detach())
    q = projection.permute(0, 2, 1, 3)
    scale = F.softplus(local.acs.log_scale)
    energy = 0.5 * (q * (q @ metric)).sum(-1) * scale[None, :, None]
    gradient, = torch.autograd.grad(energy.sum(), live_anchor)
    basis = local.acs.orthonormal_basis().detach()
    projected_gradient = -(q.detach() @ metric.detach()).sum(dim=2) * scale.detach()[None, :, None]
    analytical = torch.einsum('bkr,kcr->bkc', projected_gradient, basis)
    close(gradient, analytical, rtol=2e-5, atol=2e-5)
    print('PASS local warmup, unchanged classification/context, residual formula and direct gradient')


def train_step(model, optimizer, x, labels):
    optimizer.zero_grad(set_to_none=True)
    output = model(x)
    losses = model.loss_by_feat(output, labels)
    assert all(torch.isfinite(value).all() for value in losses.values())
    sum(value for key, value in losses.items() if key.startswith('loss')).backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    optimizer.step()
    return output


def check_memory_and_resume(control_cls, arms, x, labels):
    support = torch.tensor([[2., 0., 3., 0., 0.], [4., 0., 0., 0., 0.]])
    centres = torch.randn(2, 5, 12)
    for cls in arms:
        model, reference = make(cls), make(control_cls)
        for observation in (centres, centres * 2):
            model._update_prototypes(observation, support)
            reference._update_prototypes(observation, support)
            assert torch.equal(model.prototypes, reference.prototypes)
            assert torch.equal(model.proto_seen, reference.proto_seen)
        assert torch.equal(model.prototypes[1], torch.zeros(12))
        model = make(cls, warmup=0).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4)
        for _ in range(3):
            train_step(model, optimizer, x, labels)
        model.eval()
        buffers = {key: value.clone() for key, value in model.named_buffers()}
        expected = model(x)['final_logits'].detach()
        assert all(torch.equal(value, buffers[key]) for key, value in model.named_buffers())
        stream = io.BytesIO()
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict()), stream)
        stream.seek(0)
        saved = torch.load(stream, map_location='cpu', weights_only=True)
        restored = make(cls, warmup=0).eval()
        restored.load_state_dict(saved['model'], strict=True)
        restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=6e-4)
        restored_optimizer.load_state_dict(saved['optimizer'])
        assert torch.equal(expected, restored(x)['final_logits'])
        model.train()
        restored.train()
        train_step(model, optimizer, x, labels)
        train_step(restored, restored_optimizer, x, labels)
        for key, value in model.state_dict().items():
            assert torch.equal(value, restored.state_dict()[key]), key
    print('PASS inherited memory/EMA/unseen protection, finite gradients, eval freeze and resumed optimizer step')


def main():
    torch.set_num_threads(1)
    control_cls, geometry, acs, registry = components()
    arms = (geometry.OffSegCCMACSProto, geometry.OffSegCCMIACSProtoLocal)
    torch.manual_seed(42)
    x = torch.randn(2, 12, 4, 4)
    labels = torch.randint(5, (2, 4, 4))
    check_initialisation(control_cls, arms, acs, registry)
    check_static(control_cls, arms[0], acs, x, labels)
    check_local(control_cls, arms[1], x)
    check_memory_and_resume(control_cls, arms, x, labels)
    print('ALL PROTO GEOMETRY CHECKS PASSED')


if __name__ == '__main__':
    if '--configs-only' in sys.argv:
        check_configs()
    else:
        main()
