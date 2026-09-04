"""CPU numerical checks for the three ProtoMem experiments; requires torch.

Execute real Offset Learning, CCM, ACS/IACS, ProtoMem and variant source.
Only the feature trunk, registry, resize and initialisation plumbing are
stubbed, so no dataset or compiled mmcv is required. This does not exercise
the GPU backbone, FreqFusion kernels or a multi-process training launcher.
"""

import importlib.util
import copy
import io
import math
from pathlib import Path
import sys
import types

import torch
import torch.nn as nn
import torch.nn.functional as F


def load_components():
    root = Path(__file__).resolve().parents[1]
    for name in ('mmseg', 'mmseg.models', 'mmseg.models.decode_heads',
                 'mmcv', 'mmengine', 'mmengine.model'):
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module

    class Registry:
        def __init__(self):
            self.classes = {}

        def register_module(self):
            def register(cls):
                assert cls.__name__ not in self.classes
                self.classes[cls.__name__] = cls
                return cls
            return register

    registry = types.ModuleType('mmseg.registry')
    registry.MODELS = Registry()
    sys.modules[registry.__name__] = registry
    utils = types.ModuleType('mmseg.models.utils')
    utils.resize = F.interpolate
    sys.modules[utils.__name__] = utils

    def constant_init(module, val, bias=0):
        nn.init.constant_(module.weight, val)
        if module.bias is not None:
            nn.init.constant_(module.bias, bias)

    def trunc_normal_init(module, std, bias=0):
        nn.init.trunc_normal_(module.weight, std=std)
        if module.bias is not None:
            nn.init.constant_(module.bias, bias)

    init = types.ModuleType('mmengine.model.weight_init')
    init.constant_init = constant_init
    init.trunc_normal_ = nn.init.trunc_normal_
    init.trunc_normal_init = trunc_normal_init
    sys.modules[init.__name__] = init
    cnn = types.ModuleType('mmcv.cnn')
    cnn.build_norm_layer = lambda cfg, n, postfix=1: ('ln', nn.LayerNorm(n))
    sys.modules[cnn.__name__] = cnn

    def load(short_name):
        name = 'mmseg.models.decode_heads.' + short_name
        path = root / 'mmseg/models/decode_heads' / (short_name + '.py')
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    offset = load('offset_learning')

    class HeadStub(nn.Module):
        def __init__(self, in_channels, new_channels, num_classes,
                     channels=12, **kwargs):
            super().__init__()
            self.channels = channels
            self.num_classes = num_classes
            self.align_corners = False
            self.ignore_index = 255
            self.offset_learning = offset.Offset_Learning(num_classes, channels)

        def _transform_inputs(self, x):
            return x

        def _build_feature(self, x):
            return x

        def _stack_batch_gt(self, labels):
            return labels

        def loss_decode(self, scores, labels, ignore_index):
            return F.cross_entropy(scores, labels, ignore_index=ignore_index)

    head = types.ModuleType('mmseg.models.decode_heads.offseg_head')
    head.OffSegHead = HeadStub
    sys.modules[head.__name__] = head
    load('OffSegCCM')
    load('OffSegACS')
    proto = load('OffSegProtoMem')
    variants = load('OffSegProtoVariants')
    return proto, variants, registry.MODELS


def make(cls, warmup=3, **kwargs):
    # Identical common initial weights and RNG consumption for every arm.
    torch.manual_seed(1370346084)
    return cls(in_channels=[12], new_channels=[12], num_classes=5,
               channels=12, ccm_rank=4, ccm_hidden=8,
               iacs_assignment='posterior', proto_warmup=warmup, **kwargs)


def close(a, b, rtol=1e-5, atol=1e-6):
    torch.testing.assert_close(a, b, rtol=rtol, atol=atol)


def check_configs():
    """Use real MMEngine inheritance; run separately from the tensor stubs."""
    from mmengine.config import Config

    root = Path(__file__).resolve().parents[1] / 'local_configs/offseg2/Base'
    suffix = '_r4_responsibility_ade20k_160k-512x512.py'
    control = Config.fromfile(
        str(root / ('offsegccmiacs_proto' + suffix)),
        import_custom_modules=False).to_dict()
    work_dirs = set()
    for name, head in (
            ('protoroute', 'OffSegCCMIACSProtoRoute'),
            ('protooffset', 'OffSegCCMIACSProtoOffset'),
            ('protologn0', 'OffSegCCMIACSProtoLogN0')):
        path = root / ('offsegccmiacs_' + name + suffix)
        cfg = Config.fromfile(str(path), import_custom_modules=False).to_dict()
        assert cfg['model']['decode_head']['type'] == head
        assert cfg['custom_imports'] == dict(
            imports=['mmseg.models.decode_heads.OffSegProtoVariants'],
            allow_failed_imports=False)
        assert cfg['randomness'] == dict(seed=1370346084, deterministic=False)
        assert cfg['train_cfg']['max_iters'] == 160000
        assert cfg['train_cfg']['val_interval'] == 8000
        assert cfg['train_dataloader']['batch_size'] == 4
        assert cfg['load_from'] is None and cfg['resume'] is False
        assert cfg['default_hooks']['checkpoint']['save_best'] == 'mIoU'
        assert cfg['work_dir'] == './work_dirs/' + path.stem
        assert cfg['work_dir'] != control['work_dir']
        work_dirs.add(cfg['work_dir'])
        original_keys = control['optim_wrapper']['paramwise_cfg']['custom_keys']
        keys = cfg['optim_wrapper']['paramwise_cfg']['custom_keys']
        if name == 'protologn0':
            assert keys['proto_log_n0'] == dict(lr_mult=10., decay_mult=0.)
            # MMEngine applies the longest matching custom key first.
            ordered = sorted(sorted(keys), key=len, reverse=True)
            match = next(k for k in ordered if k in 'decode_head.proto_log_n0')
            assert match == 'proto_log_n0'
            keys.pop('proto_log_n0')
        assert keys == original_keys
        # Reject any unintended resolved-config differences, not just the
        # handful of important fields listed above.
        normalised = copy.deepcopy(cfg)
        normalised['model']['decode_head']['type'] = control['model']['decode_head']['type']
        normalised['custom_imports'] = control['custom_imports']
        normalised['work_dir'] = control['work_dir']
        normalised.pop('randomness')
        assert normalised == control, name
        print('PASS config:', name, '| seed 1370346084 | 160k | batch 4/GPU')
    assert len(work_dirs) == 3
    print('ALL RESOLVED CONFIG CHECKS PASSED')


def main():
    torch.set_num_threads(1)
    proto, variants, registry = load_components()
    control_cls = proto.OffSegCCMIACSProto
    arms = [variants.OffSegCCMIACSProtoRoute,
            variants.OffSegCCMIACSProtoOffset,
            variants.OffSegCCMIACSProtoLogN0]
    torch.manual_seed(81)
    x = torch.randn(2, 12, 4, 4)
    labels = torch.randint(0, 5, (2, 4, 4))
    param_count = sum(p.numel() for p in make(control_cls).parameters())

    for cls in arms:
        assert registry.classes[cls.__name__] is cls
        arm, control = make(cls), make(control_cls)
        assert sum(p.numel() for p in arm.parameters()) == param_count
        for _ in range(2):
            out, reference = arm(x), control(x)
            assert torch.equal(out['final_logits'], reference['final_logits'])
            assert torch.equal(out['stage1_logits'], reference['stage1_logits'])
            assert out['proto_lambda'].item() == 0
        assert arm.proto_steps.item() == 2
        out = arm(x)  # Control's warmup activates at update number 3.
        assert out['proto_lambda'].item() > 0
    print('PASS: registration, parameter count, exact warmup and activation')

    # Shared plumbing must be identical to the original centre-memory blend.
    helper = make(variants._ProtoBlendVariant, warmup=0)
    control = make(control_cls, warmup=0)
    for _ in range(3):
        out, reference = helper(x), control(x)
        for key in ('final_logits', 'stage1_logits', 'proto_lambda',
                    'proto_n0', 'proto_norm'):
            assert torch.equal(out[key], reference[key]), key
        assert torch.equal(helper.prototypes, control.prototypes)
    print('PASS: shared blend preserves original centre-memory numerics')

    route = make(arms[0], warmup=0).eval()
    with torch.no_grad():
        route.proto_seen.fill_(1)
        route.prototypes.normal_()
        # Nonzero CCM gains expose routing changes even before training.
        route.ccm.ccm_g[-1].weight.normal_(std=0.1)
    control = make(control_cls, warmup=0).eval()
    control.load_state_dict(route.state_dict())
    masks, centres, feat, _ = route._offset_learning_parts(x)
    blended, state = route._blend_prototypes(masks, centres)
    expected_route = route.offset_learning.mask_norm(
        feat @ blended.transpose(1, 2)).transpose(1, 2).contiguous()
    captured = []
    handle = route.ccm.register_forward_pre_hook(
        lambda module, args: captured.append(args))
    out, reference = route(x), control(x)
    handle.remove()
    close(captured[0][2], expected_route)
    assert not captured[0][1].requires_grad
    assert not captured[0][2].requires_grad
    assert torch.equal(out['stage1_logits'], reference['stage1_logits'])
    assert torch.equal(out['proto_lambda'], reference['proto_lambda'])
    assert out['proto_route_move'].item() > 0
    assert (out['final_logits'] - reference['final_logits']).abs().max() > 1e-6
    stage_loss = route.loss_by_feat(out, labels)['loss_stage1']
    close(stage_loss, F.cross_entropy(reference['stage1_logits'], labels))
    print('PASS: route changes CCM input, preserves CE/support, detaches context')

    offset = make(arms[1], warmup=0).eval()
    with torch.no_grad():
        offset.proto_seen.fill_(1)
        offset.prototypes.normal_()
    delta = torch.randn(2, 5, 12, requires_grad=True)
    base = offset.offset_learning.cls_repr
    masks = torch.randn(2, 5, 16)
    support = masks.softmax(dim=1).sum(dim=-1)
    n0 = F.softplus(offset.proto_n0_raw)
    lam = n0 / (support + n0)
    blended, state = offset._blend_prototypes(masks, base + delta)
    expected = (base + (1 - lam[..., None]) * delta +
                lam[..., None] * offset.prototypes.unsqueeze(0))
    close(blended, expected)
    blended.sum().backward()
    close(base.grad, torch.full_like(base, 2.0))
    close(delta.grad, (1 - lam[..., None]).expand_as(delta))
    assert not offset.prototypes.requires_grad
    close(state['proto_norm'], (base.detach() + offset.prototypes)
          .norm(dim=-1).mean())

    offset = make(arms[1], warmup=5).train()
    # Explicit support test: unseen classes neither initialise nor blend.
    known_delta = torch.randn(2, 5, 12)
    eligibility = torch.tensor([[2., 0., 3., 0., 0.], [4., 0., 0., 0., 0.]])
    offset._update_prototypes(offset._memory_observation(
        offset.offset_learning.cls_repr + known_delta), eligibility)
    close(offset.prototypes[0], known_delta[:, 0].mean(dim=0))
    close(offset.prototypes[2], known_delta[0, 2])
    assert torch.equal(offset.proto_seen, torch.tensor([1., 0., 1., 0., 0.]))
    before = offset.prototypes.clone()
    offset._update_prototypes(known_delta * 2, eligibility)
    close(offset.prototypes[0], before[0] * 0.99 +
          (known_delta[:, 0] * 2).mean(dim=0) * 0.01)
    assert torch.equal(offset.prototypes[1], torch.zeros(12))
    offset.eval()
    blended, _ = offset._blend_prototypes(masks, offset.offset_learning.cls_repr + delta.detach())
    close(blended[:, 1], (offset.offset_learning.cls_repr + delta.detach())[:, 1])
    print('PASS: offset formula, live W gradient, detached bank, eligibility and EMA')

    log_model = make(arms[2])
    control = make(control_cls)
    assert not hasattr(log_model, 'proto_n0_raw')
    supports = torch.tensor([0., 0.01, 1., 20., 200., 10000.])
    n0, log_lam = log_model._mixing_weight(supports, torch.float32)
    raw_n0 = F.softplus(control.proto_n0_raw)
    raw_lam = raw_n0 / (supports + raw_n0)
    close(log_lam, raw_lam)
    close(n0, raw_n0)
    log_grad, = torch.autograd.grad(log_lam.sum(), log_model.proto_log_n0)
    raw_grad, = torch.autograd.grad(raw_lam.sum(), control.proto_n0_raw)
    close(log_grad, raw_grad * raw_n0.detach())
    with torch.no_grad():
        log_model.proto_log_n0.add_(math.log(2))
    n0, doubled = log_model._mixing_weight(supports, torch.float32)
    close(n0, torch.tensor(400.0))
    assert (doubled[1:] > log_lam[1:]).all()
    print('PASS: logn0 initial equivalence, zero support, relative derivative and update')

    for cls in arms:
        arm = make(cls, warmup=0).train()
        optimizer = torch.optim.AdamW(arm.parameters(), lr=6e-4)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            out = arm(x)
            losses = arm.loss_by_feat(out, labels)
            loss = sum(value for key, value in losses.items()
                       if key.startswith('loss'))
            assert torch.isfinite(loss)
            loss.backward()
            n0_param = (arm.proto_log_n0 if cls is arms[2] else arm.proto_n0_raw)
            assert n0_param.grad is not None and n0_param.grad.abs().item() > 0
            for parameter in arm.parameters():
                if parameter.grad is not None:
                    assert torch.isfinite(parameter.grad).all()
            assert all(torch.isfinite(value).all() for value in losses.values())
            optimizer.step()
        arm.eval()
        memory_before = {key: value.clone() for key, value in arm.named_buffers()}
        expected = arm(x)['final_logits'].detach()
        for key, value in arm.named_buffers():
            assert torch.equal(value, memory_before[key]), key
        stream = io.BytesIO()
        torch.save(dict(model=arm.state_dict(), optimizer=optimizer.state_dict()), stream)
        stream.seek(0)
        checkpoint = torch.load(stream, map_location='cpu', weights_only=True)
        restored = make(cls, warmup=0).eval()
        restored.load_state_dict(checkpoint['model'], strict=True)
        restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=6e-4)
        restored_optimizer.load_state_dict(checkpoint['optimizer'])
        assert torch.equal(expected, restored(x)['final_logits'])
        arm.train()
        restored.train()
        assert torch.equal(arm(x)['final_logits'], restored(x)['final_logits'])
        assert torch.equal(arm.prototypes, restored.prototypes)
    print('PASS: finite forward/backward, learned scalar gradients, eval freeze, checkpoint resume')
    print('ALL PROTO VARIANT CHECKS PASSED')


if __name__ == '__main__':
    if '--configs-only' in sys.argv:
        check_configs()
    else:
        main()
