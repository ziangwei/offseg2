"""Check the rank-eight IACS scorer inside the actual ProtoRoute head."""
import copy
from pathlib import Path
import sys
import torch
from proto_variants_sanity import load_components, make, close


def configs():
    from mmengine.config import Config
    folder = Path(__file__).resolve().parents[1] / 'local_configs/offseg2/Base'
    a = Config.fromfile(str(folder / 'offsegccmiacs_protoroute_r4_responsibility_ade20k_160k-512x512.py'), import_custom_modules=False).to_dict()
    b = Config.fromfile(str(folder / 'offsegccmiacs_protoroute_r8_responsibility_ade20k_160k-512x512.py'), import_custom_modules=False).to_dict()
    assert b['model']['decode_head']['acs_rank'] == 8
    assert b['work_dir'] == './work_dirs/offsegccmiacs_protoroute_r8_responsibility_ade20k_160k-512x512'
    normal = copy.deepcopy(b)
    normal['model']['decode_head']['acs_rank'] = 4
    normal['work_dir'] = a['work_dir']
    assert normal == a
    h = b['model']['decode_head']
    assert h['iacs_classwise_mix'] is False
    assert h['iacs_candidate_topk'] == 0 and h['iacs_assignment'] == 'posterior'
    assert h['proto_n0_init'] == 200 and h['proto_momentum'] == .01
    assert b['randomness']['seed'] == 1370346084
    assert b['load_from'] is None and b['resume'] is False
    assert b['train_cfg']['max_iters'] == 160000
    print('PASS complete config comparison: rank/work_dir only; no classmix combination')


def main():
    torch.set_num_threads(1)
    _, variants, _ = load_components()
    cls = variants.OffSegCCMIACSProtoRoute
    base = make(cls, warmup=0)
    model = make(cls, warmup=0, acs_rank=8)
    assert sum(p.numel() for p in model.parameters()) - sum(p.numel() for p in base.parameters()) == 5 * 12 * 4
    basis = model.acs.orthonormal_basis()
    close(basis.transpose(-1, -2) @ basis, torch.eye(8).expand(5, 8, 8), atol=2e-6)
    feat, centres, logits = torch.randn(2, 16, 12), torch.randn(2, 5, 12), torch.randn(2, 16, 5)
    q = model.acs.project_residual(feat, centres)
    explicit = torch.einsum('bnkc,kcr->bnkr', feat[:, :, None, :] - centres[:, None, :, :], basis)
    close(q, explicit)
    metric, mix, _, _ = model.acs.image_metric(q, logits)
    assert metric.shape == (2, 5, 8, 8)
    close(metric.diagonal(dim1=-2, dim2=-1).sum(-1), torch.full((2, 5), 8.))
    assert torch.linalg.eigvalsh(metric).min() >= -1e-5
    correction = model.acs(feat, centres, logits)[0]
    expected = .5 * torch.einsum('bnkr,bkrs,bnks->bnk', q, metric, q)
    expected *= torch.nn.functional.softplus(model.acs.log_scale)[None, None, :]
    close(correction, expected)
    print('PASS r8 orthogonality, residual projection, PSD/trace and explicit quadratic scorer')
    x, labels = torch.randn(2, 12, 4, 4), torch.randint(0, 5, (2, 4, 4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4)
    def step(m, opt):
        m.train()
        opt.zero_grad(set_to_none=True)
        loss = m.loss_by_feat(m(x), labels)
        assert {k for k in loss if k.startswith('loss')} == {'loss_ccm', 'loss_stage1'}
        (loss['loss_ccm'] + loss['loss_stage1']).backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
        assert (m.acs.raw_basis.grad.abs().sum(dim=(0, 1)) > 0).all()
        opt.step()
    step(model, optimizer)
    restored = make(cls, warmup=0, acs_rank=8)
    restored.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
    opt2 = torch.optim.AdamW(restored.parameters(), lr=6e-4)
    opt2.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    step(model, optimizer)
    step(restored, opt2)
    for k, v in model.state_dict().items():
        torch.testing.assert_close(v, restored.state_dict()[k], rtol=0, atol=0)
    model.eval()
    buffers = {k: v.clone() for k, v in model.named_buffers()}
    with torch.no_grad():
        assert torch.isfinite(model(x)['final_logits']).all()
    assert all(torch.equal(v, buffers[k]) for k, v in model.named_buffers())
    print('PASS all eight directions train, original losses, checkpoint/optimizer resume and eval freeze')


if __name__ == '__main__':
    configs() if '--configs-only' in sys.argv else main()
