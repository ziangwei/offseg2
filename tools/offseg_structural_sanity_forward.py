"""CPU sanity checks for ACS and SRG; no dataset or mmcv required."""

import importlib.util
from pathlib import Path
import sys
import types

import torch
import torch.nn as nn


def _load_components():
    """Load the real component classes while stubbing only mmseg plumbing.

    This machine need not have mmcv/mmengine.  The tensor implementations are
    executed from their actual source files; only the registry decorator and
    the unused parent head are replaced for this standalone check.
    """
    class _Registry:
        @staticmethod
        def register_module(*args, **kwargs):
            def decorator(cls):
                return cls
            return decorator

    root = Path(__file__).resolve().parents[1]
    package_names = ['mmseg', 'mmseg.models', 'mmseg.models.decode_heads']
    for name in package_names:
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module

    registry = types.ModuleType('mmseg.registry')
    registry.MODELS = _Registry()
    sys.modules['mmseg.registry'] = registry

    parent = types.ModuleType('mmseg.models.decode_heads.OffSegCCM')
    parent.OffSegCCM = nn.Module
    sys.modules['mmseg.models.decode_heads.OffSegCCM'] = parent

    loaded = {}
    for short_name in ('OffSegACS', 'OffSegSRG'):
        qualified = f'mmseg.models.decode_heads.{short_name}'
        path = root / 'mmseg' / 'models' / 'decode_heads' / f'{short_name}.py'
        spec = importlib.util.spec_from_file_location(qualified, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[qualified] = module
        spec.loader.exec_module(module)
        loaded[short_name] = module
    return (loaded['OffSegACS'].restrict_correction_to_topk,
            loaded['OffSegACS'].AffineClassSubspace,
            loaded['OffSegACS'].ImageAdaptiveAffineClassSubspace,
            loaded['OffSegSRG'].SemanticResidualRegionGraph)


restrict_correction_to_topk, AffineClassSubspace, \
    ImageAdaptiveAffineClassSubspace, \
    SemanticResidualRegionGraph = _load_components()


def check(name, condition, detail=''):
    print(f"  [{'OK ' if condition else 'FAIL'}] {name} {detail}")
    return bool(condition)


def main():
    torch.manual_seed(0)
    batch, channels, classes, height, width = 2, 256, 150, 16, 16
    length = height * width
    all_ok = True

    print('1) affine class subspace')
    feat = torch.randn(batch, length, channels, requires_grad=True)
    centres = torch.randn(batch, classes, channels, requires_grad=True)
    acs = AffineClassSubspace(classes, channels, rank=4, scale_init=0.05)
    correction, scale = acs(feat, centres)
    basis = acs.orthonormal_basis()
    direct_pixel = torch.einsum('bnc,kcr->bnkr', feat, basis)
    direct_centre = torch.einsum('bkc,kcr->bkr', centres, basis)
    direct_energy = (
        direct_pixel - direct_centre[:, None]).square().sum(dim=-1)
    direct_correction = 0.5 * direct_energy * scale.view(1, 1, -1)
    gram = torch.einsum('kcr,kcs->krs', basis, basis)
    identity = torch.eye(4).expand_as(gram)
    all_ok &= check('correction shape', correction.shape == (
        batch, length, classes), str(tuple(correction.shape)))
    gram_error = float((gram - identity).abs().max().detach())
    all_ok &= check('basis is orthonormal', gram_error < 1e-4,
                    f"max|U^TU-I|={gram_error:.2e}")
    scale_mean = float(scale.mean().detach())
    all_ok &= check('scale starts positive',
                    bool((scale > 0).all()), f'mean={scale_mean:.4f}')
    all_ok &= check('refactor is identical to measured ACS formula',
                    float((correction - direct_correction).abs().max()
                          .detach()) == 0.0)
    correction.mean().backward()
    all_ok &= check('basis receives gradient at step 0',
                    acs.raw_basis.grad is not None and
                    float(acs.raw_basis.grad.abs().sum()) > 0)

    print('2) image-adaptive affine class subspace')
    iacs = ImageAdaptiveAffineClassSubspace(
        classes, channels, rank=4, scale_init=0.05, mix_init=0.10)
    logits = torch.randn(batch, length, classes)
    adaptive, adaptive_scale, mix, anisotropy = iacs(
        feat.detach(), centres.detach(), logits)
    projection = iacs.project_residual(feat.detach(), centres.detach())
    metric, _, _ = iacs.image_metric(projection, logits)
    metric_trace = metric.diagonal(dim1=-2, dim2=-1).sum(-1)
    min_eigenvalue = torch.linalg.eigvalsh(metric.float()).min()
    all_ok &= check('adaptive correction shape',
                    adaptive.shape == (batch, length, classes),
                    str(tuple(adaptive.shape)))
    all_ok &= check('metric keeps trace r',
                    float((metric_trace - 4).abs().max().detach()) < 1e-4)
    all_ok &= check('metric is positive definite',
                    float(min_eigenvalue.detach()) > 0)
    all_ok &= check('mix initialises as configured',
                    abs(float(mix.detach()) - 0.10) < 1e-6,
                    f'mix={float(mix.detach()):.4f}')
    all_ok &= check('anisotropy is finite and active',
                    bool(torch.isfinite(anisotropy)) and
                    float(anisotropy.detach()) > 0)
    adaptive.mean().backward()
    all_ok &= check('IACS basis receives gradient',
                    iacs.raw_basis.grad is not None and
                    float(iacs.raw_basis.grad.abs().sum()) > 0)
    all_ok &= check('IACS mix receives gradient',
                    iacs.mix_logit.grad is not None and
                    float(iacs.mix_logit.grad.abs()) > 0)

    print('3) competition restriction and class-wise mix')
    candidate_logits = torch.randn(batch, length, classes)
    raw_correction = torch.randn(
        batch, length, classes, requires_grad=True)
    restricted = restrict_correction_to_topk(
        raw_correction, candidate_logits, topk=3)
    top3 = candidate_logits.topk(3, dim=-1).indices
    selected = raw_correction.gather(-1, top3)
    selected_after = restricted.gather(-1, top3)
    nonzero = (restricted.detach() != 0).sum(dim=-1)
    all_ok &= check('exactly three corrections survive per pixel',
                    bool((nonzero == 3).all()))
    all_ok &= check('selected correction values are unchanged',
                    float((selected - selected_after).abs().max().detach())
                    == 0.0)
    restricted.sum().backward()
    grad_selected = raw_correction.grad.gather(-1, top3)
    all_ok &= check('gradient survives only on selected classes',
                    bool((grad_selected == 1).all()) and
                    int((raw_correction.grad != 0).sum()) == top3.numel())

    classmix = ImageAdaptiveAffineClassSubspace(
        classes, channels, rank=4, scale_init=0.05, mix_init=0.90,
        classwise_mix=True)
    class_adaptive, _, class_mix, _ = classmix(
        feat.detach(), centres.detach(), logits)
    all_ok &= check('class-wise mix shape and initial value',
                    class_mix.shape == (classes,) and
                    float((class_mix - 0.9).abs().max().detach()) < 1e-6)
    class_adaptive.mean().backward()
    all_ok &= check('all class-wise mix values receive gradient',
                    classmix.mix_logit.grad is not None and
                    bool((classmix.mix_logit.grad != 0).all()))

    # Both launch candidates also exercise rank 8.  Keep this tensor tiny so
    # the standalone CPU check remains cheap.
    iacs8 = ImageAdaptiveAffineClassSubspace(
        num_classes=5, embed_dims=32, rank=8,
        scale_init=0.05, mix_init=0.10)
    small_feat = torch.randn(1, 9, 32)
    small_centres = torch.randn(1, 5, 32)
    small_logits = torch.randn(1, 9, 5)
    small_correction, _, _, _ = iacs8(
        small_feat, small_centres, small_logits)
    basis8 = iacs8.orthonormal_basis()
    gram8 = torch.einsum('kcr,kcs->krs', basis8, basis8)
    eye8 = torch.eye(8).expand_as(gram8)
    all_ok &= check('rank-8 path shape and orthogonality',
                    small_correction.shape == (1, 9, 5) and
                    float((gram8 - eye8).abs().max().detach()) < 1e-4)

    print('4) semantic residual region graph')
    residual = torch.randn(batch, channels, height, width,
                           requires_grad=True)
    srg = SemanticResidualRegionGraph(
        channels, num_nodes=32, node_channels=64)
    delta, relation = srg(residual)
    all_ok &= check('delta shape', delta.shape == residual.shape,
                    str(tuple(delta.shape)))
    all_ok &= check('relation shape', relation.shape == (batch, 32, 32),
                    str(tuple(relation.shape)))
    all_ok &= check('zero-init is exact identity residual',
                    float(delta.abs().max().detach()) == 0.0)
    delta.square().mean().backward()
    # Squared zero output has zero derivative.  Use a linear readout to verify
    # that the zero output projection itself can leave zero immediately.
    srg.zero_grad(set_to_none=True)
    delta, _ = srg(residual)
    delta.mean().backward()
    all_ok &= check('zero output projection receives gradient',
                    srg.out_proj.weight.grad is not None and
                    float(srg.out_proj.weight.grad.abs().sum()) > 0)
    all_ok &= check('all outputs finite',
                    bool(torch.isfinite(correction).all() and
                         torch.isfinite(relation).all()))

    n_acs = sum(parameter.numel() for parameter in acs.parameters())
    n_iacs = sum(parameter.numel() for parameter in iacs.parameters())
    n_srg = sum(parameter.numel() for parameter in srg.parameters())
    print(f'    ACS: {n_acs / 1e6:.3f} M parameters')
    print(f'   IACS: {n_iacs / 1e6:.3f} M parameters')
    print(f'    SRG: {n_srg / 1e6:.3f} M parameters')
    print('=' * 62)
    print('ALL PASS' if all_ok else 'SOMETHING FAILED -- do not launch')
    return 0 if all_ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
