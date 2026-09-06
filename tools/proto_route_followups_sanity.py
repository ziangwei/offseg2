"""Numerical checks for the two independent ProtoRoute follow-ups.

Uses the existing real-head CPU harness; only trunk/framework interfaces
are stubbed. Distributed aggregation is checked against an unequal-rank
partition with a simulated all_reduce, not a multi-process GPU run.
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
SUFFIX = '_r4_responsibility_ade20k_160k-512x512.py'
ARMS = (
    ('protoroutewrite', 'OffSegCCMIACSProtoRouteWrite'),
    ('protoroutece', 'OffSegCCMIACSProtoRouteCE'),
)


def check_configs():
    from mmengine.config import Config
    folder = ROOT / 'local_configs/offseg2/Base'
    control = Config.fromfile(
        str(folder / ('offsegccmiacs_protoroute' + SUFFIX)),
        import_custom_modules=False).to_dict()
    directories = {control['work_dir']}
    for slug, head in ARMS:
        filename = 'offsegccmiacs_' + slug + SUFFIX
        cfg = Config.fromfile(str(folder / filename),
                              import_custom_modules=False).to_dict()
        assert cfg['model']['decode_head']['type'] == head
        assert cfg['custom_imports'] == dict(
            imports=['mmseg.models.decode_heads.OffSegProtoRouteFollowups'],
            allow_failed_imports=False)
        assert cfg['work_dir'] == './work_dirs/' + Path(filename).stem
        assert cfg['work_dir'] not in directories
        directories.add(cfg['work_dir'])
        normalised = copy.deepcopy(cfg)
        for key in ('custom_imports', 'work_dir'):
            normalised[key] = control[key]
        normalised['model']['decode_head']['type'] = control['model']['decode_head']['type']
        assert normalised == control, filename
        assert cfg['randomness']['seed'] == 1370346084
        assert cfg['train_cfg']['max_iters'] == 160000
        assert cfg['train_cfg']['val_interval'] == 8000
        assert cfg['train_dataloader']['batch_size'] == 4
        assert cfg['model']['backbone']['type'] == 'efficientformerv2_s2_feat'
        assert cfg['load_from'] is None and cfg['resume'] is False
        assert cfg['default_hooks']['checkpoint']['save_best'] == 'mIoU'
        print('PASS full resolved config:', slug)
    print('ALL FOLLOW-UP CONFIG CHECKS PASSED')


def components():
    _, variants, registry = load_components()
    name = 'mmseg.models.decode_heads.OffSegProtoRouteFollowups'
    spec = importlib.util.spec_from_file_location(
        name, ROOT / 'mmseg/models/decode_heads/OffSegProtoRouteFollowups.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return variants.OffSegCCMIACSProtoRoute, module, registry


def objective(model, x, labels):
    losses = model.loss_by_feat(model(x), labels)
    assert {k for k in losses if k.startswith('loss')} == {
        'loss_stage1', 'loss_ccm'}
    assert all(torch.isfinite(v).all() for v in losses.values())
    return losses['loss_stage1'] + losses['loss_ccm']


def check_writes(control_cls, write_cls, module):
    centres = torch.randn(3, 5, 12, requires_grad=True)
    # Includes unequal evidence, a single eligible observation, and an
    # entirely unseen class. The single-observation mass is well below 1.
    support = torch.tensor([[2., 0., 1., 200., 0.],
                            [2000., 2., 0., 200., 1.],
                            [20., 0., 1., 200., 0.]])
    write = make(write_cls, warmup=0)
    write._update_prototypes(centres, support)
    for k in range(5):
        valid = support[:, k] > 1
        if valid.any():
            n = support[valid, k].double()
            w = n / (n + 200)
            reference = sum(w[j] * centres[valid, k][j].detach().double()
                            for j in range(len(w))) / w.sum()
            close(write.prototypes[k], reference.float())
        else:
            assert torch.equal(write.prototypes[k], torch.zeros(12))
    close(write.prototypes[1], centres[1, 1])
    assert torch.equal(write.proto_seen, torch.tensor([1., 1., 0., 1., 0.]))
    assert not write.prototypes.requires_grad and write.prototypes.grad is None
    assert write.proto_n0_raw.grad is None
    previous = write.prototypes.clone()
    write._update_prototypes(centres * 2, support)
    close(write.prototypes, previous * 1.01)
    assert 0 < write._proto_write_ess_ratio <= 1.00001
    assert write.proto_steps == 2
    before_empty_write = write.prototypes.clone()
    write._update_prototypes(centres, torch.zeros_like(support))
    assert torch.equal(write.prototypes, before_empty_write)
    assert write._proto_write_ess_ratio == 0

    # Equal support must recover equal-image writing, including initialisation.
    equal = torch.full_like(support, 10)
    old, new = make(control_cls), make(write_cls)
    for _ in range(3):
        old._update_prototypes(centres, equal)
        new._update_prototypes(centres, equal)
        close(old.prototypes, new.prototypes)

    # Simulate unequal rank-local evidence. Verify every rank produces the
    # global weighted mean, not an average of local normalised means.
    weights = support / (support + 200) * (support > 1)
    global_sum = (centres.detach() * weights[..., None]).sum(0)
    global_totals = torch.stack((weights.sum(0), weights.square().sum(0),
                                (support > 1).float().sum(0)))
    for indices in ([0], [1, 2]):
        rank = make(write_cls)
        calls = []
        def reduce(tensor):
            target = global_sum if len(calls) == 0 else global_totals
            calls.append(tensor.shape)
            tensor.copy_(target)
        with patch.object(module.dist, 'is_available', return_value=True), \
             patch.object(module.dist, 'is_initialized', return_value=True), \
             patch.object(module.dist, 'all_reduce', side_effect=reduce):
            rank._update_prototypes(centres[indices], support[indices])
        assert len(calls) == 2
        close(rank.prototypes, previous)
    print('PASS weighted write: analytic mean, first/absent class, EMA, equal support, rank partition')


def check_ce(control_cls, ce_cls, x, labels):
    old, new = make(control_cls, warmup=0), make(ce_cls, warmup=0)
    with torch.no_grad():
        old.proto_seen.fill_(1)
        old.prototypes.normal_()
        old.ccm.ccm_g[-1].weight.normal_(std=0.1)
    new.load_state_dict(old.state_dict())
    masks, _, _, _ = new._offset_learning_parts(x)
    expected_support = masks.detach().float().softmax(dim=1).sum(-1)
    original_update = new._update_prototypes
    support_seen = []
    def update(centres, support):
        support_seen.append(support.clone())
        return original_update(centres, support)
    captured = []
    handle = new.ccm.register_forward_pre_hook(
        lambda module, args: captured.append(args))
    with patch.object(new, '_update_prototypes', side_effect=update):
        out = new(x)
    reference = old(x)
    handle.remove()
    assert torch.equal(support_seen[0], expected_support)
    assert torch.equal(new.prototypes, old.prototypes)
    assert torch.equal(out['final_logits'], reference['final_logits'])
    assert torch.equal(out['proto_lambda'], reference['proto_lambda'])
    assert not torch.equal(out['stage1_logits'], reference['stage1_logits'])
    close(out['stage1_logits'].flatten(2), captured[0][2])
    assert not captured[0][1].requires_grad
    assert not captured[0][2].requires_grad
    assert out['stage1_logits'].requires_grad
    loss = new.loss_by_feat(out, labels)['loss_stage1']
    close(loss, F.cross_entropy(captured[0][2].view(2, 5, 4, 4), labels))
    loss.backward()
    assert new.proto_n0_raw.grad is not None
    assert new.proto_n0_raw.grad.abs() > 0
    assert new.offset_learning.cls_repr.grad.abs().sum() > 0
    assert new.prototypes.grad is None
    assert all(p.grad is None for p in new.ccm.parameters())
    print('PASS route CE: original support/bank, identical final forward, correct CE, live gradient and detached context')


def main():
    torch.set_num_threads(1)
    control_cls, module, registry = components()
    write_cls = module.OffSegCCMIACSProtoRouteWrite
    ce_cls = module.OffSegCCMIACSProtoRouteCE
    torch.manual_seed(71)
    x = torch.randn(2, 12, 4, 4)
    labels = torch.randint(0, 5, (2, 4, 4))
    for cls in (write_cls, ce_cls):
        old, new = make(control_cls), make(cls)
        assert registry.classes[cls.__name__] is cls
        assert old.state_dict().keys() == new.state_dict().keys()
        for key, value in old.state_dict().items():
            assert torch.equal(value, new.state_dict()[key]), key
        assert sum(p.numel() for p in old.parameters()) == sum(
            p.numel() for p in new.parameters())
        for _ in range(2):
            a, b = old(x), new(x)
            assert torch.equal(a['final_logits'], b['final_logits'])
            assert torch.equal(a['stage1_logits'], b['stage1_logits'])
            assert b['proto_lambda'] == 0
        assert new(x)['proto_lambda'] > 0
    print('PASS registration, common initial state, parameter count, warmup and activation')
    check_writes(control_cls, write_cls, module)
    check_ce(control_cls, ce_cls, x, labels)

    for cls in (write_cls, ce_cls):
        model = make(cls, warmup=0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            objective(model, x, labels).backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all()
                       for p in model.parameters())
            optimizer.step()
        model.eval()
        bank = {k: v.clone() for k, v in model.named_buffers()}
        with torch.no_grad():
            expected = model(x)['final_logits']
            model(x * 2)
        for k, v in model.named_buffers():
            assert torch.equal(v, bank[k]), k
        stream = io.BytesIO()
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict()), stream)
        stream.seek(0)
        state = torch.load(stream, weights_only=True)
        restored = make(cls, warmup=0).eval()
        restored.load_state_dict(state['model'], strict=True)
        restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=6e-4)
        restored_optimizer.load_state_dict(state['optimizer'])
        with torch.no_grad():
            assert torch.equal(expected, restored(x)['final_logits'])
        for m, opt in ((model, optimizer), (restored, restored_optimizer)):
            m.train()
            opt.zero_grad(set_to_none=True)
            objective(m, x * .9, labels).backward()
            opt.step()
        for k, v in model.state_dict().items():
            assert torch.equal(v, restored.state_dict()[k]), k
    print('PASS finite forward/backward, exactly two CEs, frozen eval bank and resumed optimizer update')

    for cls in (write_cls, ce_cls):
        torch.manual_seed(1370346084)
        model = cls(in_channels=[256], new_channels=[256], num_classes=150,
                    channels=256, ccm_rank=64, ccm_hidden=128,
                    iacs_assignment='posterior', proto_warmup=0)
        y = torch.randint(0, 150, (2, 4, 4))
        objective(model, torch.randn(2, 256, 4, 4), y).backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in model.parameters())
    print('PASS 150-class/256-channel real head forward/backward (small spatial grid)')
    print('ALL PROTO ROUTE FOLLOW-UP CHECKS PASSED')


if __name__ == '__main__':
    if '--configs-only' in sys.argv:
        check_configs()
    else:
        main()
