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
    adaptive, adaptive_scale, mix, anisotropy, statistics = iacs(
        feat.detach(), centres.detach(), logits)
    projection = iacs.project_residual(feat.detach(), centres.detach())
    metric, _, _, _ = iacs.image_metric(projection, logits)
    metric_trace = metric.diagonal(dim1=-2, dim2=-1).sum(-1)
    min_eigenvalue = torch.linalg.eigvalsh(metric.float()).min()
    default_weight = torch.softmax(logits, dim=1)
    default_q = projection.detach().permute(0, 2, 1, 3)
    default_sqrt_weight = default_weight.permute(
        0, 2, 1).unsqueeze(-1).sqrt()
    default_scatter = (
        default_q * default_sqrt_weight).transpose(-1, -2) @ (
            default_q * default_sqrt_weight)
    default_identity = torch.eye(4).view(1, 1, 4, 4)
    default_trace = default_scatter.diagonal(
        dim1=-2, dim2=-1).sum(-1)
    default_normalised = (
        4 * default_scatter + iacs.scatter_eps * default_identity
    ) / (default_trace[..., None, None] + iacs.scatter_eps)
    default_metric = ((1 - mix) * default_identity +
                      mix * default_normalised)
    all_ok &= check('adaptive correction shape',
                    adaptive.shape == (batch, length, classes),
                    str(tuple(adaptive.shape)))
    all_ok &= check('metric keeps trace r',
                    float((metric_trace - 4).abs().max().detach()) < 1e-4)
    all_ok &= check('metric is positive definite',
                    float(min_eigenvalue.detach()) > 0)
    all_ok &= check('default estimator remains bitwise-equivalent',
                    float((metric - default_metric).abs().max().detach())
                    == 0.0)
    all_ok &= check('mix initialises as configured',
                    abs(float(mix.detach()) - 0.10) < 1e-6,
                    f'mix={float(mix.detach()):.4f}')
    all_ok &= check('anisotropy is finite and active',
                    bool(torch.isfinite(anisotropy)) and
                    float(anisotropy.detach()) > 0)
    all_ok &= check('moment diagnostics are finite',
                    all(torch.isfinite(value) for value in
                        statistics.values()))
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
    class_adaptive, _, class_mix, _, _ = classmix(
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
    small_correction, _, _, _, _ = iacs8(
        small_feat, small_centres, small_logits)
    basis8 = iacs8.orthonormal_basis()
    gram8 = torch.einsum('kcr,kcs->krs', basis8, basis8)
    eye8 = torch.eye(8).expand_as(gram8)
    all_ok &= check('rank-8 path shape and orthogonality',
                    small_correction.shape == (1, 9, 5) and
                    float((gram8 - eye8).abs().max().detach()) < 1e-4)

    print('4) centered and responsibility moment estimators')
    moment_projection = torch.randn(2, 17, 5, 4)
    moment_logits = torch.randn(2, 17, 5)
    residual_shift = torch.randn(2, 1, 5, 4)
    centered = ImageAdaptiveAffineClassSubspace(
        num_classes=5, embed_dims=32, rank=4,
        center_statistics=True)
    centered_metric, _, _, _ = centered.image_metric(
        moment_projection, moment_logits)
    shifted_metric, _, _, _ = centered.image_metric(
        moment_projection + residual_shift, moment_logits)
    all_ok &= check('centered covariance is translation invariant',
                    float((centered_metric - shifted_metric).abs().max()
                          .detach())
                    < 1e-5)

    responsibility = ImageAdaptiveAffineClassSubspace(
        num_classes=5, embed_dims=32, rank=4,
        assignment='posterior')
    responsibility_weight = responsibility.assignment_weights(moment_logits)
    all_ok &= check('responsibilities normalise over space per class',
                    float((responsibility_weight.sum(dim=1) - 1).abs().max())
                    < 1e-6)
    class_bias = torch.linspace(-2, 2, 5).view(1, 1, 5)
    posterior_shift = responsibility.assignment_weights(
        moment_logits + class_bias)
    spatial_shift = centered.assignment_weights(moment_logits + class_bias)
    spatial_base = centered.assignment_weights(moment_logits)
    all_ok &= check('posterior assignment retains class competition',
                    float((posterior_shift - responsibility_weight).abs()
                          .max()) > 1e-4 and
                    float((spatial_shift - spatial_base).abs().max())
                    < 1e-7)

    combined = ImageAdaptiveAffineClassSubspace(
        num_classes=5, embed_dims=32, rank=4,
        center_statistics=True, assignment='posterior')
    combined_metric, _, _, combined_statistics = combined.image_metric(
        moment_projection, moment_logits)
    combined_trace = combined_metric.diagonal(
        dim1=-2, dim2=-1).sum(-1)
    combined_min_eigenvalue = torch.linalg.eigvalsh(
        combined_metric.float()).min()
    all_ok &= check('combined metric stays trace-normalised and PD',
                    float((combined_trace - 4).abs().max().detach())
                    < 1e-4 and
                    float(combined_min_eigenvalue.detach()) > 0 and
                    all(torch.isfinite(value) for value in
                        combined_statistics.values()))

    reliable = ImageAdaptiveAffineClassSubspace(
        num_classes=5, embed_dims=32, rank=4,
        center_statistics=True, assignment='posterior',
        reliability_shrink=True)
    constant_logits = torch.zeros_like(moment_logits)
    constant_metric, _, _, constant_statistics = reliable.image_metric(
        moment_projection, constant_logits)
    constant_unshrunk, _, _, _ = combined.image_metric(
        moment_projection, constant_logits)
    moment_identity = torch.eye(4).view(1, 1, 4, 4)
    constant_expected = moment_identity + 0.2 * (
        constant_unshrunk - moment_identity)
    all_ok &= check('uniform posterior uses its self-confidence',
                    float((constant_metric - constant_expected).abs().max()
                          .detach()) < 1e-6 and
                    abs(float(constant_statistics[
                        'iacs_reliability'].detach()) - 0.2) < 1e-6)

    random_posterior = torch.softmax(moment_logits, dim=2)
    manual_reliability = (
        random_posterior.square().sum(dim=1) /
        random_posterior.sum(dim=1))
    _, measured_reliability, _ = reliable.assignment_statistics(
        moment_logits)
    all_ok &= check('reliability equals posterior self-confidence',
                    float((manual_reliability - measured_reliability).abs()
                          .max().detach()) < 1e-6)

    labels = torch.arange(17).remainder(5).view(1, 17, 1).expand(2, -1, -1)
    certain_logits = torch.full_like(moment_logits, -8.0)
    certain_logits.scatter_(2, labels, 8.0)
    _, certain_reliability, _ = reliable.assignment_statistics(
        certain_logits)
    all_ok &= check('coherent posterior enables adaptive covariance',
                    float(certain_reliability.min().detach()) > 0.99)
    reliable_feat = torch.randn(2, 17, 32)
    reliable_centres = torch.randn(2, 5, 32)
    reliable_correction, _, _, _, _ = reliable(
        reliable_feat, reliable_centres, certain_logits)
    reliable_correction.mean().backward()
    all_ok &= check('reliable full path trains basis and mixture',
                    reliable.raw_basis.grad is not None and
                    float(reliable.raw_basis.grad.abs().sum()) > 0 and
                    reliable.mix_logit.grad is not None and
                    float(reliable.mix_logit.grad.abs()) > 0)
    all_ok &= check('all moment variants add zero parameters',
                    sum(parameter.numel() for parameter in
                        reliable.parameters()) ==
                    sum(parameter.numel() for parameter in
                        centered.parameters()))

    print('5) semantic residual region graph')
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
