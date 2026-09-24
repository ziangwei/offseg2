"""Real new-head numerics with the established trunk/framework stubs."""
import copy
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
KINDS = [('featurecontext','FeatureContext'), ('centrepool','CentrePool')]


def configs():
    from mmengine.config import Config
    folder=ROOT/'local_configs/offseg2/Base'
    control=Config.fromfile(str(folder/'offsegccmiacs_protoroute_r4_responsibility_ade20k_160k-512x512.py'),import_custom_modules=False).to_dict()
    for kind,suffix in KINDS:
        path=folder/f'offsegccmiacs_protoroute_{kind}_b_r4_responsibility_ade20k_160k-512x512.py'
        cfg=Config.fromfile(str(path),import_custom_modules=False).to_dict()
        assert cfg['model']['decode_head']['type']=='OffSegCCMIACSProtoRoute'+suffix
        assert cfg['work_dir'].endswith(path.stem)
        assert cfg['default_hooks']['checkpoint']['max_keep_ckpts']==2
        assert cfg['default_hooks']['checkpoint']['save_best']=='mIoU'
        if kind=='featurecontext':
            assert cfg['model']['decode_head'].pop('feature_width')==64
        for key in ('work_dir','custom_imports'):
            cfg[key]=control[key]
        cfg['model']['decode_head']['type']=control['model']['decode_head']['type']
        assert cfg==control
    print('PASS two resolved configs: only intended head change, identical seed/protocol/loss setup, 2 recent+best')


def tensors():
    import torch
    from proto_variants_sanity import load_components, make
    torch.set_num_threads(1)
    _, variants, _=load_components()
    name='mmseg.models.decode_heads.OffSegRouteInput'
    spec=importlib.util.spec_from_file_location(name,ROOT/'mmseg/models/decode_heads/OffSegRouteInput.py')
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    # Explicit class-then-space normalisation, shift invariance and extreme logits.
    scores=torch.randn(2,5,21,requires_grad=True)
    p=scores.softmax(1)
    weights=module.competitive_pool_weights(scores)
    torch.testing.assert_close(weights,p/p.sum(-1,keepdim=True))
    torch.testing.assert_close(weights.sum(-1),torch.ones(2,5))
    shift=torch.randn(2,1,21)*3
    torch.testing.assert_close(weights,module.competitive_pool_weights(scores+shift),rtol=1e-5,atol=1e-6)
    permutation=torch.tensor([4,0,2,1,3])
    torch.testing.assert_close(weights[:,permutation],module.competitive_pool_weights(scores[:,permutation]))
    extreme=torch.tensor([[[10000.,-10000.],[0.,0.],[-10000.,10000.]]],requires_grad=True)
    ew=module.competitive_pool_weights(extreme)
    assert torch.isfinite(ew).all()
    ew.square().sum().backward()
    assert torch.isfinite(extreme.grad).all()
    count=sum(p.numel() for p in module.FeatureContext(256).parameters())
    assert count==34048
    print('PASS competitive pooling oracle/invariance/extremes; FeatureContext adds exactly 34048 parameters')
    for kind,suffix in KINDS:
        cls=getattr(module,'OffSegCCMIACSProtoRoute'+suffix)
        extra={'feature_width':4} if kind=='featurecontext' else {}
        model=make(cls,warmup=2,**extra)
        control=make(variants.OffSegCCMIACSProtoRoute,warmup=2)
        missing=model.load_state_dict(control.state_dict(),strict=False)
        assert not missing.unexpected_keys
        x=torch.randn(2,12,3,7)
        labels=torch.randint(0,5,(2,1,3,7))
        if kind=='featurecontext':
            for _ in range(3):
                a,b=control(x),model(x)
                torch.testing.assert_close(a['final_logits'],b['final_logits'],rtol=0,atol=0)
                torch.testing.assert_close(control.prototypes,model.prototypes,rtol=0,atol=0)
            assert b['proto_lambda']>0
        else:
            masks,centres,aligned,shape=model._offset_learning_parts(x)
            _,old_centres,old_aligned,_=control._offset_learning_parts(x)
            torch.testing.assert_close(aligned,old_aligned,rtol=0,atol=0)
            feat=x.permute(0,2,3,1).reshape(2,21,12)
            reference=model.offset_learning.cls_repr.expand(2,-1,-1)
            coupled=(feat@reference.transpose(1,2)).transpose(1,2)
            p=coupled.softmax(1)
            oracle=reference+model.offset_learning.cls_offset_proj((p/p.sum(-1,keepdim=True))@feat)
            torch.testing.assert_close(centres,oracle,rtol=1e-5,atol=1e-6)
            assert not torch.equal(centres,old_centres)
            assert sum(p.numel() for p in model.parameters())==sum(p.numel() for p in control.parameters())
        opt=torch.optim.AdamW(model.parameters(),lr=.001)
        def step(net,optimizer):
            net.train()
            optimizer.zero_grad(set_to_none=True)
            out=net(x)
            losses=net.loss_by_feat(out,labels)
            assert {k for k in losses if k.startswith('loss')}=={'loss_ccm','loss_stage1'}
            (losses['loss_ccm']+losses['loss_stage1']).backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
            optimizer.step()
        for _ in range(3):
            step(model,opt)
        if kind=='featurecontext':
            assert model.feature_context.down.weight.grad.abs().sum()>0
        restored=make(cls,warmup=2,**extra)
        restored.load_state_dict(copy.deepcopy(model.state_dict()),strict=True)
        opt2=torch.optim.AdamW(restored.parameters(),lr=.001)
        opt2.load_state_dict(copy.deepcopy(opt.state_dict()))
        step(model,opt)
        step(restored,opt2)
        for k,v in model.state_dict().items():
            torch.testing.assert_close(v,restored.state_dict()[k],rtol=0,atol=0)
        model.eval()
        before={k:v.clone() for k,v in model.named_buffers()}
        with torch.no_grad():
            out=model(torch.randn(1,12,5,2))
        assert out['final_logits'].shape==(1,5,5,2) and torch.isfinite(out['final_logits']).all()
        assert all(torch.equal(v,before[k]) for k,v in model.named_buffers())
        print('PASS',kind,'real head gradients/two CE/exact optimiser recovery/rectangular inputs/frozen eval')


if __name__=='__main__':
    configs() if '--configs-only' in sys.argv else tensors()
