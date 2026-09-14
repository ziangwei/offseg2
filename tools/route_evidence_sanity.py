"""CPU checks of real evidence heads; backbone/framework plumbing is stubbed."""
import copy
import importlib.util
from pathlib import Path
import sys
import torch
from proto_variants_sanity import load_components, make

ROOT = Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(1)
    _, variants, _ = load_components()
    name = 'mmseg.models.decode_heads.OffSegRouteEvidence'
    spec = importlib.util.spec_from_file_location(name, ROOT / 'mmseg/models/decode_heads/OffSegRouteEvidence.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    x, labels = torch.randn(2, 12, 3, 5), torch.randint(0, 5, (2, 3, 5))
    for kind in ('Dispersion', 'Recollect'):
        cls = getattr(module, 'OffSegCCMIACSProtoRoute' + kind)
        control, model = make(variants.OffSegCCMIACSProtoRoute, warmup=2), make(cls, warmup=2)
        assert sum(p.numel() for p in model.parameters()) - sum(p.numel() for p in control.parameters()) == 792
        with torch.no_grad():
            control.ccm.ccm_g[-1].weight.normal_(std=.03)
        state = {('ccm.core.' + k[4:] if k.startswith('ccm.') else k): v
                 for k, v in control.state_dict().items()}
        missing = model.load_state_dict(state, strict=False)
        assert not missing.unexpected_keys
        assert len(missing.missing_keys) == 4
        for _ in range(3):
            a, b = control(x), model(x)
            for key in ('stage1_logits', 'final_logits'):
                torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
        assert b['proto_lambda'] > 0
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        def step(net, opt):
            opt.zero_grad(set_to_none=True)
            losses = net.loss_by_feat(net(x), labels)
            assert {k for k in losses if k.startswith('loss')} == {'loss_ccm', 'loss_stage1'}
            (losses['loss_ccm'] + losses['loss_stage1']).backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
            opt.step()
        step(model, optimizer)
        assert model.ccm.output.weight.grad.abs().sum() > 0
        step(model, optimizer)
        assert model.ccm.project.weight.grad.abs().sum() > 0
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
        # Both modules are class-order equivariant and accept rectangular maps.
        feat, centres, logits = torch.randn(2, 15, 12), torch.randn(2, 5, 12), torch.randn(2, 5, 15)
        perm = torch.tensor([3, 1, 4, 0, 2])
        torch.testing.assert_close(model.ccm(feat, centres, logits)[0],
                                   model.ccm(feat, centres[:, perm], logits[:, perm])[0])
        posterior = logits.transpose(1, 2).softmax(-1)
        evidence = model.ccm.evidence(feat, centres, posterior, posterior)
        if kind == 'Dispersion':
            # Verify against explicit central variance, including one-class zero.
            values = model.ccm.project(model.ccm.norm(centres))
            mean = posterior @ values
            explicit = ((values[:, None] - mean[:, :, None]).square() * posterior[..., None]).sum(2)
            torch.testing.assert_close(evidence, explicit.log1p(), rtol=1e-4, atol=1e-6)
            one = torch.zeros_like(posterior)
            one[..., 0] = 1
            assert model.ccm.evidence(feat, centres, one, one).abs().max() < 1e-6
        else:
            # Direct weighted gather oracle and stopped pixel evidence gradient.
            weights = posterior.transpose(1, 2)
            weights = weights / weights.sum(-1, keepdim=True)
            values = model.ccm.project(model.ccm.norm(weights @ feat))
            torch.testing.assert_close(evidence, posterior @ values)
            probe = feat.clone().requires_grad_()
            read = model.ccm.evidence(probe, centres, posterior, posterior)
            assert torch.autograd.grad(read.sum(), probe, allow_unused=True)[0] is None
        print('PASS', kind, 'zero-init parity, memory, gradients, resume, frozen eval and evidence oracle')


if __name__ == '__main__':
    main()
