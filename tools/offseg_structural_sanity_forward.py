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
    return (loaded['OffSegACS'].AffineClassSubspace,
            loaded['OffSegSRG'].SemanticResidualRegionGraph)


AffineClassSubspace, SemanticResidualRegionGraph = _load_components()


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
    correction.mean().backward()
    all_ok &= check('basis receives gradient at step 0',
                    acs.raw_basis.grad is not None and
                    float(acs.raw_basis.grad.abs().sum()) > 0)

    print('2) semantic residual region graph')
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
    n_srg = sum(parameter.numel() for parameter in srg.parameters())
    print(f'    ACS: {n_acs / 1e6:.3f} M parameters')
    print(f'    SRG: {n_srg / 1e6:.3f} M parameters')
    print('=' * 62)
    print('ALL PASS' if all_ok else 'SOMETHING FAILED -- do not launch')
    return 0 if all_ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
