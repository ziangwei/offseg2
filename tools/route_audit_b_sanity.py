"""Synthetic metric oracles, frozen real-head transparency and mocked sbatch."""
import contextlib
import ast
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
import zipfile

import numpy as np
from route_error_audit_b import ErrorAudit, boundary_band, region_sizes, merge, write_report

ROOT = Path(__file__).resolve().parents[1]


def statistics_checks():
    gt = np.zeros((100,100), dtype=np.int64)
    gt[:2,:2] = 1       # four pixels, small at <=0.1%.
    gt[20:25,20:25] = 2 # 25 pixels, medium at <=1%.
    gt[-1] = 255
    valid = gt != 255
    sizes = region_sizes(gt, valid)
    assert sizes[0,0] == 4 and sizes[20,20] == 25 and sizes[50,50] == 9871
    edge = boundary_band(gt, valid, 0)
    assert edge[1,1] and edge[2,1] and not edge[50,50] and not edge[-2,50]
    pred = gt.copy()
    pred[~valid] = 0
    pred[0,0] = 0
    pred[20,20] = 4  # predicted class absent from GT.
    pred[50,50] = 1
    top = np.stack([(pred+i)%6 for i in range(5)])
    audit = ErrorAudit(6)
    audit.consume(gt, top, 'one')
    s = merge([audit.shard()], [str(i) for i in range(6)])
    assert s['groups']['all']['valid_pixels'] == 9900
    assert s['groups']['all']['wrong_pixels'] == 3
    assert s['groups']['absent_prediction']['wrong_pixels'] == 1
    assert s['groups']['small_region']['valid_pixels'] == 4
    assert s['groups']['medium_region']['valid_pixels'] == 25
    assert s['groups']['large_region']['valid_pixels'] == 9871
    assert s['groups']['all']['wrong_gt_in_topk'] == {'2':1, '3':1, '5':2}
    expected = np.zeros((6,6), dtype=np.int64)
    for g,p in zip(gt[valid],pred[valid]):
        expected[g,p] += 1
    assert np.array_equal(s['confusion'],expected)
    for row in s['classes']:
        k = row['class_id']
        assert row['fn'] == expected[k].sum()-expected[k,k]
        assert row['fp'] == expected[:,k].sum()-expected[k,k]
    assert sum(s['groups'][n+'_region']['valid_pixels'] for n in ('small','medium','large')) == 9900
    assert s['groups']['boundary_r3']['valid_pixels'] + s['groups']['interior_r3']['valid_pixels'] == 9900
    empty = ErrorAudit(6)
    assert merge([audit.shard(), empty.shard()], [str(i) for i in range(6)]) == s
    second = ErrorAudit(6)
    second.consume(np.zeros((1,1),dtype=np.int64),np.arange(5).reshape(5,1,1),'two')
    combined = merge([audit.shard(),second.shard()],[str(i) for i in range(6)])
    assert combined['images'] == 2 and combined['groups']['all']['valid_pixels'] == 9901
    try:
        merge([audit.shard(),audit.shard()],[str(i) for i in range(6)])
        raise AssertionError('Duplicate input was accepted')
    except ValueError:
        pass
    # Even with a different tied top-k ordering, use the actual evaluated argmax.
    tied = ErrorAudit(6)
    tied.consume(np.zeros((1,1),dtype=np.int64),np.arange(1,6).reshape(5,1,1), 'tie', np.zeros((1,1),dtype=np.int64))
    assert tied.groups['all'][1] == 0
    with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as output:
        s.update(expected_score_check='PASS', definition='test')
        write_report(s, output)
        assert json.loads((Path(output)/'error_audit.json').read_text(encoding='utf-8'))['miou'] == s['miou']
    print('PASS confusion/FP/FN, ignored labels, boundary exclusion, connected areas, top-k, unequal shards, ties, CSV/JSON')


def frozen_head():
    import torch
    from proto_variants_sanity import load_components, make
    torch.set_num_threads(1)
    _, variants, _ = load_components()
    model = make(variants.OffSegCCMIACSProtoRoute, warmup=1)
    x = torch.randn(2,12,4,4)
    with torch.no_grad():
        model.train()(x)
        model.eval()
        saved = copy.deepcopy(model.state_dict())
        before = model(x)['final_logits']
        audit = ErrorAudit(5)
        for i in range(2):
            top = before[i].topk(5,dim=0).indices.cpu().numpy()
            audit.consume(np.zeros((4,4),dtype=np.int64),top,str(i))
        after = model(x)['final_logits']
        torch.testing.assert_close(before,after,rtol=0,atol=0)
        for key,value in saved.items():
            torch.testing.assert_close(value,model.state_dict()[key],rtol=0,atol=0)
    print('PASS real Route head output/state unchanged by output-only audit')


def submission():
    sys.path.insert(0,str(ROOT/'tools/slurm'))
    import submit_audit_b as task
    from submit import snapshot
    with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as folder:
        root = Path(folder).resolve()
        checkpoint = root/'best_mIoU_iter_160000.pth'
        checkpoint.write_bytes(b'fixture, never loaded')
        archive = io.BytesIO()
        with zipfile.ZipFile(archive,'w') as z:
            for name in ['tools/route_error_audit_b.py','tools/route_cost_audit_b.py',
                         'tools/slurm/route_audit_b.slurm','tools/slurm/run_audit_b.sh',task.CONFIG]:
                z.writestr(name,(ROOT/name).read_bytes())
        calls=[]
        def command(args):
            calls.append(args)
            return '12345' if args[0]=='sbatch' else 'fixture-sha'
        args=['submit_audit_b.py','--checkpoint',str(checkpoint),'--runs-root',str(root/'runs')]
        with patch.object(sys,'argv',args), patch.object(task,'command',command), \
             patch.object(task,'active_names',return_value=set()), \
             patch.object(task,'snapshot',side_effect=lambda data,dest:snapshot(data,dest,repo=root)), \
             patch('subprocess.check_output',return_value=archive.getvalue()):
            task.main()
        records=list((root/'runs').glob('*/run.json'))
        assert len(records)==1
        meta=json.loads(records[0].read_text())
        assert meta['checkpoint']==str(checkpoint) and meta['job_id']=='12345'
        assert len([c for c in calls if c[0]=='sbatch'])==1
        assert all('train.py' not in ' '.join(c) for c in calls)
        with patch.object(sys,'argv',args+['--dry-run']), patch.object(task,'command',command):
            with contextlib.redirect_stdout(io.StringIO()):
                task.main()
        assert len(list((root/'runs').glob('*/run.json')))==1
        with patch.object(sys,'argv',args), patch.object(task,'active_names',return_value={'os2_route_audit_b'}):
            with contextlib.redirect_stderr(io.StringIO()):
                try:
                    task.main()
                    raise AssertionError('Duplicate active audit accepted')
                except SystemExit as exc:
                    assert exc.code==2
    print('PASS one independent mocked sbatch, frozen source, exact checkpoint, dry-run, duplicate rejection')


def configs():
    from mmengine.config import Config
    from route_cost_audit_b import normalise_state
    base=Config.fromfile(str(ROOT/'local_configs/offseg/Base/offseg-b_ade20k_160k-512x512.py'),import_custom_modules=False)
    route=Config.fromfile(str(ROOT/'local_configs/offseg2/Base/offsegccmiacs_protoroute_r4_responsibility_ade20k_160k-512x512.py'),import_custom_modules=False)
    assert base.model.backbone == route.model.backbone
    assert base.model.test_cfg == route.model.test_cfg
    assert route.test_dataloader.batch_size==1
    assert route.model.test_cfg.mode=='slide'
    assert normalise_state({'state_dict':{'module.a':1,'b':2}})=={'a':1,'b':2}
    print('PASS unchanged ADE slide protocol, identical S2 cost backbone, checkpoint prefixes')


def cost_cache_regression():
    import itertools
    import math
    import torch
    from mmengine.analysis import FlopAnalyzer
    from route_cost_audit_b import prepare_eval, short_error
    # Execute the actual backbone class without importing unrelated compiled ops.
    path=ROOT/'mmseg/models/backbones/efficientformer_v2.py'
    node=next(n for n in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(n,ast.ClassDef) and n.name=='Attention4D')
    scope=dict(torch=torch,nn=torch.nn,itertools=itertools,math=math)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),scope)
    attention=scope['Attention4D'](dim=8,key_dim=2,num_heads=2,attn_ratio=2,resolution=3)
    attention.eval()
    assert attention.ab.requires_grad
    x=torch.randn(1,8,3,3)
    with torch.no_grad():
        reference=attention(x)
        try:
            FlopAnalyzer(attention,(x,)).total()
            raise AssertionError('Expected original cached-gradient tracing failure')
        except RuntimeError as exc:
            assert 'requires grad as a constant' in str(exc)
    state=copy.deepcopy(attention.state_dict())
    prepare_eval(attention)
    assert not attention.ab.requires_grad and attention.ab.grad_fn is None
    with torch.no_grad():
        torch.testing.assert_close(reference,attention(x),rtol=0,atol=0)
        analyzer=FlopAnalyzer(attention,(x,)).unsupported_ops_warnings(False)
        assert analyzer.total()>0
    for k,v in state.items():
        torch.testing.assert_close(v,attention.state_dict()[k],rtol=0,atol=0)
    assert len(short_error(RuntimeError('failure\nTensor:'+'x'*100000)))<100
    print('PASS actual EfficientFormer cache failure reproduced; no_grad eval fixes tracing with exact outputs/state')


def isolated_cost_dispatch():
    import torch
    import route_cost_audit_b as cost
    with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as folder:
        calls=[]
        def worker(cmd,check):
            assert check
            which=cmd[cmd.index('--worker-model')+1]
            calls.append(which)
            label='OffSeg-B' if which=='offseg' else 'Proto-route-B'
            result=dict(protocol='fixture',gpu='fixture',torch='fixture',cuda='fixture',cudnn='fixture',models={label:dict(
                weights='fixture',parameters=1,repeats=[dict(wall_mean_ms=1,peak_allocated_bytes=10)],complexity=dict(status='TRACED_ESTIMATE'))})
            (Path(folder)/('cost_'+which+'.json')).write_text(json.dumps(result))
        args=['cost','route.py','route.pth','--work-dir',folder]
        with patch.object(sys,'argv',args),patch.object(torch.cuda,'is_available',return_value=True),patch.object(cost.subprocess,'run',side_effect=worker):
            with contextlib.redirect_stdout(io.StringIO()):
                cost.main()
        assert calls==['offseg','route']
        assert len(json.loads((Path(folder)/'cost_audit.json').read_text())['models'])==2
    import submit_audit_b as task
    with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as folder:
        ckpt=Path(folder)/'test.pth'
        ckpt.write_bytes(b'fixture')
        capture=io.StringIO()
        with patch.object(sys,'argv',['submit','--checkpoint',str(ckpt),'--cost-only','--dry-run']),patch.object(task,'command',return_value='test-sha'):
            with contextlib.redirect_stdout(capture):
                task.main()
        data=json.loads(capture.getvalue())
        assert data['metadata']['cost_only'] and '--time=01:00:00' in data['command']
        assert '--job-name=os2_route_cost_b' in data['command']
    print('PASS separate cost-worker dispatch/merge and cost-only one-hour sbatch preview')


if __name__ == '__main__':
    (ROOT/'tmp').mkdir(exist_ok=True)
    statistics_checks()
    configs()
    cost_cache_regression()
    frozen_head()
    submission()
    isolated_cost_dispatch()
