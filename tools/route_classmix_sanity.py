"""Check the existing classwise IACS implementation in the ProtoRoute arm."""
import copy
from pathlib import Path
import sys
import torch
from proto_variants_sanity import load_components, make, close


def configs():
    from mmengine.config import Config
    folder = Path(__file__).resolve().parents[1] / 'local_configs/offseg2/Base'
    suffix = '_r4_responsibility_ade20k_160k-512x512.py'
    a = Config.fromfile(str(folder / ('offsegccmiacs_protoroute' + suffix)),
                        import_custom_modules=False).to_dict()
    b = Config.fromfile(str(folder / ('offsegccmiacs_protoroute_classmix' + suffix)),
                        import_custom_modules=False).to_dict()
    assert b['model']['decode_head']['iacs_classwise_mix'] is True
    assert b['work_dir'].endswith('offsegccmiacs_protoroute_classmix' + suffix[:-3])
    normal = copy.deepcopy(b)
    normal['model']['decode_head']['iacs_classwise_mix'] = False
    normal['work_dir'] = a['work_dir']
    assert normal == a
    head = b['model']['decode_head']
    assert head['iacs_candidate_topk'] == 0
    assert head['iacs_mix_init'] == .10
    assert head['proto_n0_init'] == 200 and head['proto_momentum'] == .01
    assert head['proto_warmup'] == 4000
    assert b['randomness']['seed'] == 1370346084
    assert b['train_cfg']['max_iters'] == 160000
    assert b['load_from'] is None and b['resume'] is False
    assert b['optim_wrapper']['paramwise_cfg']['custom_keys']['acs.mix_logit'] == dict(lr_mult=10., decay_mult=0.)
    print('PASS resolved config: only classwise flag/work_dir change; original protocol intact')


def main():
    torch.set_num_threads(1)
    _, variants, _ = load_components()
    cls = variants.OffSegCCMIACSProtoRoute
    shared = make(cls, warmup=0)
    shared_rng = torch.get_rng_state().clone()
    model = make(cls, warmup=0, iacs_classwise_mix=True)
    assert torch.equal(shared_rng, torch.get_rng_state())
    assert model.acs.mix_logit.shape == (5,)
    assert sum(p.numel() for p in model.parameters()) - sum(p.numel() for p in shared.parameters()) == 4
    for k, v in shared.state_dict().items():
        expected = v.expand(5) if k == 'acs.mix_logit' else v
        torch.testing.assert_close(expected, model.state_dict()[k], rtol=0, atol=0)
    x = torch.randn(2, 12, 4, 4)
    labels = torch.randint(0, 5, (2, 4, 4))
    a, b = shared(x), model(x)
    close(a['final_logits'], b['final_logits'])
    close(a['stage1_logits'], b['stage1_logits'])
    la, lb = shared.loss_by_feat(a, labels), model.loss_by_feat(b, labels)
    for losses in (la, lb):
        assert {k for k in losses if k.startswith('loss')} == {'loss_ccm', 'loss_stage1'}
        (losses['loss_ccm'] + losses['loss_stage1']).backward()
    grad = model.acs.mix_logit.grad
    assert torch.isfinite(grad).all() and grad.abs().sum() > 0
    assert grad.std() > 0
    close(shared.acs.mix_logit.grad, grad.sum(), rtol=1e-4)
    print('PASS common initialisation, equivalent initial scores and per-class gradients summing to shared gradient')
    opt = torch.optim.AdamW(model.parameters(), lr=6e-4)
    opt.step()
    state = copy.deepcopy(model.state_dict())
    restored = make(cls, warmup=0, iacs_classwise_mix=True)
    restored.load_state_dict(state, strict=True)
    opt2 = torch.optim.AdamW(restored.parameters(), lr=6e-4)
    opt2.load_state_dict(copy.deepcopy(opt.state_dict()))
    for m, optimizer in ((model, opt), (restored, opt2)):
        optimizer.zero_grad(set_to_none=True)
        losses = m.loss_by_feat(m(x * .8), labels)
        (losses['loss_ccm'] + losses['loss_stage1']).backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
        optimizer.step()
    for k, v in model.state_dict().items():
        torch.testing.assert_close(v, restored.state_dict()[k], rtol=0, atol=0)
    model.eval()
    before = {k: v.clone() for k, v in model.named_buffers()}
    with torch.no_grad():
        assert torch.isfinite(model(x)['final_logits']).all()
    assert all(torch.equal(v, before[k]) for k, v in model.named_buffers())
    print('PASS finite training, classwise checkpoint/optimizer resume and frozen inference memory')


if __name__ == '__main__':
    configs() if '--configs-only' in sys.argv else main()
