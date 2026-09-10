"""Numerical checks of final-loss gradients through ProtoRoute's CCM context."""
import copy
from pathlib import Path
import sys
import torch
from proto_variants_sanity import load_components, make, close


def configs():
    from mmengine.config import Config
    folder = Path(__file__).resolve().parents[1] / 'local_configs/offseg2/Base'
    suffix = '_r4_responsibility_ade20k_160k-512x512.py'
    a = Config.fromfile(str(folder / ('offsegccmiacs_protoroute' + suffix)), import_custom_modules=False).to_dict()
    b = Config.fromfile(str(folder / ('offsegccmiacs_protoroute_grad' + suffix)), import_custom_modules=False).to_dict()
    assert b['model']['decode_head']['ccm_detach_context'] is False
    assert b['work_dir'] == './work_dirs/offsegccmiacs_protoroute_grad' + suffix[:-3]
    normal = copy.deepcopy(b)
    normal['model']['decode_head']['ccm_detach_context'] = True
    normal['work_dir'] = a['work_dir']
    assert normal == a
    h = b['model']['decode_head']
    assert h['iacs_detach_statistics'] is True and h['ccm_top_p'] == .9
    assert h['acs_rank'] == 4 and h['iacs_classwise_mix'] is False
    assert h['proto_n0_init'] == 200 and h['proto_momentum'] == .01
    assert b['randomness']['seed'] == 1370346084
    assert b['train_cfg']['max_iters'] == 160000
    assert b['load_from'] is None and b['resume'] is False
    print('PASS resolved config: only CCM detach/work_dir change, original losses and all other settings retained')


def main():
    torch.set_num_threads(1)
    _, variants, _ = load_components()
    cls = variants.OffSegCCMIACSProtoRoute
    control = make(cls, warmup=0)
    rng = torch.get_rng_state().clone()
    model = make(cls, warmup=0, ccm_detach_context=False)
    assert torch.equal(rng, torch.get_rng_state())
    assert sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in control.parameters())
    for k, v in control.state_dict().items():
        assert torch.equal(v, model.state_dict()[k]), k
    x, labels = torch.randn(2, 12, 4, 4), torch.randint(0, 5, (2, 4, 4))
    # Context is initially behind the zero final layer of the CCM generator.
    # Activate that layer only in this fixture to test the later-training path.
    with torch.no_grad():
        control.ccm.ccm_g[-1].weight.normal_(std=.02)
    model.load_state_dict(control.state_dict())
    captured = {}
    def capture(name):
        def hook(module, inputs):
            captured[name] = inputs
            if inputs[2].requires_grad:
                inputs[2].retain_grad()
        return hook
    hooks = [m.ccm.register_forward_pre_hook(capture(name))
             for name, m in (('control', control), ('grad', model))]
    a, b = control(x), model(x)
    torch.testing.assert_close(a['final_logits'], b['final_logits'], rtol=0, atol=0)
    torch.testing.assert_close(a['stage1_logits'], b['stage1_logits'], rtol=0, atol=0)
    assert not captured['control'][1].requires_grad
    assert not captured['control'][2].requires_grad
    assert captured['grad'][1].requires_grad and captured['grad'][2].requires_grad
    losses = model.loss_by_feat(b, labels)
    assert {k for k in losses if k.startswith('loss')} == {'loss_stage1', 'loss_ccm'}
    losses['loss_ccm'].backward()
    assert captured['grad'][2].grad is not None
    assert torch.isfinite(captured['grad'][2].grad).all()
    assert captured['grad'][2].grad.abs().sum() > 0
    assert all(not buffer.requires_grad for buffer in model.buffers())
    assert model.acs.detach_statistics
    for hook in hooks:
        hook.remove()
    print('PASS identical forward scores, final-CE gradient through routing, context gradients and detached memory/statistics')
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4)
    def step(m, opt):
        m.train()
        opt.zero_grad(set_to_none=True)
        loss = m.loss_by_feat(m(x), labels)
        (loss['loss_ccm'] + loss['loss_stage1']).backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
        opt.step()
    step(model, optimizer)
    restored = make(cls, warmup=0, ccm_detach_context=False)
    restored.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
    opt2 = torch.optim.AdamW(restored.parameters(), lr=6e-4)
    opt2.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    step(model, optimizer)
    step(restored, opt2)
    for k, v in model.state_dict().items():
        torch.testing.assert_close(v, restored.state_dict()[k], rtol=0, atol=0)
    model.eval()
    control.eval().load_state_dict(model.state_dict())
    before = {k: v.clone() for k, v in model.named_buffers()}
    with torch.no_grad():
        torch.testing.assert_close(model(x)['final_logits'], control(x)['final_logits'], rtol=0, atol=0)
    assert all(torch.equal(v, before[k]) for k, v in model.named_buffers())
    print('PASS original two-CE training, optimizer resume, same-state inference and frozen evaluation memory')


if __name__ == '__main__':
    configs() if '--configs-only' in sys.argv else main()
