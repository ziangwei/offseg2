"""CPU checks with synthetic descriptors; never substitutes them for real assets."""
import copy
import importlib.util
from pathlib import Path
import sys
import tempfile
import torch
from proto_variants_sanity import make
from route_decision_sanity import load, ROOT


def configs():
    from mmengine.config import Config
    base = ROOT / 'local_configs/offseg2/Base'
    suffix = '_r4_responsibility_ade20k_160k-512x512.py'
    control = Config.fromfile(str(base / ('offsegccmiacs_protoroute' + suffix)), import_custom_modules=False).to_dict()
    for kind in ('textmetric', 'textsubspace'):
        cfg = Config.fromfile(str(base / ('offsegccmiacs_protoroute_' + kind + '_b' + suffix)), import_custom_modules=False).to_dict()
        normal = copy.deepcopy(cfg)
        normal['model']['decode_head'].pop('text_asset')
        normal['model']['decode_head']['type'] = control['model']['decode_head']['type']
        for key in ('custom_imports', 'work_dir'):
            normal[key] = control[key]
        if kind == 'textsubspace':
            keys = normal['optim_wrapper']['paramwise_cfg']['custom_keys']
            assert keys.pop('acs.core.mix_logit') == keys['acs.mix_logit']
        assert normal == control
    print('PASS text configs: fixed original protocol, only intended head/asset and optimizer-path changes')


def main():
    torch.set_num_threads(1)
    _, variants = load()
    name = 'mmseg.models.decode_heads.OffSegRouteText'
    spec = importlib.util.spec_from_file_location(name, ROOT / 'mmseg/models/decode_heads/OffSegRouteText.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    folder = Path(tempfile.mkdtemp(prefix='route_text_fixture_', dir=ROOT / 'tmp'))
    asset = folder / 'descriptions.pt'
    torch.save(dict(embeddings=torch.randn(5, 6, 512), class_names=[str(i) for i in range(5)]), asset)
    x, labels = torch.randn(2, 12, 3, 5), torch.randint(0, 5, (2, 3, 5))
    for kind in ('TextMetric', 'TextSubspace'):
        cls = getattr(module, 'OffSegCCMIACSProtoRoute' + kind)
        model = make(cls, warmup=2, text_asset=str(asset))
        control = make(variants.OffSegCCMIACSProtoRoute, warmup=2)
        with torch.no_grad():
            control.ccm.ccm_g[-1].weight.normal_(std=.02)
        state = {('acs.core.' + k[4:] if kind == 'TextSubspace' and k.startswith('acs.') else k): v
                 for k, v in control.state_dict().items()}
        missing = model.load_state_dict(state, strict=False)
        assert not missing.unexpected_keys
        for _ in range(3):
            a, b = control(x), model(x)
            torch.testing.assert_close(a['final_logits'], b['final_logits'], rtol=1e-5, atol=1e-6)
        opt = torch.optim.AdamW(model.parameters(), lr=.001)
        def step(net, optimizer):
            net.train()
            optimizer.zero_grad(set_to_none=True)
            loss = net.loss_by_feat(net(x), labels)
            assert {k for k in loss if k.startswith('loss')} == {'loss_stage1', 'loss_ccm'}
            (loss['loss_stage1'] + loss['loss_ccm']).backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
            optimizer.step()
        step(model, opt)
        step(model, opt)
        learned = model.text_metric.projection if kind == 'TextMetric' else model.acs.project[0]
        assert learned.weight.grad.abs().sum() > 0
        restored = make(cls, warmup=2, text_asset=str(asset))
        restored.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
        opt2 = torch.optim.AdamW(restored.parameters(), lr=.001)
        opt2.load_state_dict(copy.deepcopy(opt.state_dict()))
        step(model, opt)
        step(restored, opt2)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)
        model.eval()
        buffers = {k: v.clone() for k, v in model.named_buffers()}
        with torch.no_grad():
            model(torch.randn(1, 12, 5, 2))
        assert all(torch.equal(v, buffers[k]) for k, v in model.named_buffers())
        if kind == 'TextMetric':
            w = model.text_metric.weights()
            assert (w > .5).all() and (w < 1.5).all()
        print('PASS', kind, 'zero-init, activated memory, gradients, two CE, exact resume, frozen text/eval')
    sys.path.insert(0, str(ROOT))
    from tools.gen_text_anchors import ADE_CLASSES
    for malformed in (dict(embeddings=torch.randn(150, 512), class_names=ADE_CLASSES),
                      dict(embeddings=torch.randn(150, 6, 512), class_names=ADE_CLASSES[::-1])):
        torch.save(malformed, folder / 'bad.pt')
        try:
            module.load_descriptions(folder / 'bad.pt', 150)
            raise AssertionError('Bad asset accepted')
        except ValueError:
            pass
    print('PASS class-name-only and misordered assets rejected')


if __name__ == '__main__':
    configs() if '--configs-only' in sys.argv else main()
