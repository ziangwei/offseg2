"""CPU checks: real Route transparency, paired count oracles, safe packaging."""
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
import zipfile
import numpy as np
from PIL import Image
from .common import ROOT, crop_boxes, write_json, sha256
from .probe import RouteCapture, align_gt


def frozen_route():
    import torch
    sys.path.insert(0, str(ROOT/'tools'))
    from proto_variants_sanity import load_components, make
    _, variants, _ = load_components()
    head = make(variants.OffSegCCMIACSProtoRoute, warmup=1)
    torch.set_num_threads(1)
    with torch.no_grad():
        head.train()(torch.randn(1,12,4,4))
        head.eval()

    class Segmentor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.decode_head = head
            self.test_cfg = dict(mode='slide',crop_size=(4,4),stride=(3,3))
            from types import SimpleNamespace
            self.data_preprocessor = SimpleNamespace(_enable_normalize=True,
                mean=torch.zeros(12,1,1),std=torch.ones(12,1,1),channel_conversion=False)

        def encode_decode(self, inputs, metas):
            return self.decode_head(inputs)['final_logits']

        def inference(self, inputs, metas):
            result = torch.zeros(1,5,*inputs.shape[-2:])
            count = torch.zeros_like(result)
            for y1,x1,y2,x2 in crop_boxes(*inputs.shape[-2:],(4,4),(3,3)):
                result[:,:,y1:y2,x1:x2] += self.encode_decode(inputs[:,:,y1:y2,x1:x2],metas)
                count[:,:,y1:y2,x1:x2] += 1
            return result/count

    model = Segmentor().eval()
    x = torch.randn(1,12,5,7)
    gt = np.zeros((10,14), dtype='uint8')
    gt[3:7,4:9] = 2
    state = copy.deepcopy(model.state_dict())
    with torch.no_grad():
        reference = model.inference(x,[dict(img_shape=(5,7))])
        for dense in (False,True):
            probe = RouteCapture(model,dense=dense,class_ids=[0,2])
            probe.install()
            observed = model.inference(x,[dict(img_shape=(5,7))])
            torch.testing.assert_close(reference,observed,rtol=0,atol=0)
            windows = probe.consume(gt)
            assert len(windows) == 4
            for window in windows:
                assert window['gt_area'].sum() == 16
                np.testing.assert_allclose(window['support'].sum(),16,rtol=1e-6)
                assert window['lam'].shape == (5,)
                if dense:
                    assert window['responsibility'].shape == (2,16)
                    assert window['delta'].shape == (2,16)
                    np.testing.assert_allclose(window['responsibility'].sum(1),1,rtol=1e-6)
                    np.testing.assert_allclose(window['spatial_weight'].sum(1),1,rtol=1e-6)
                    assert window['metric4x4'].shape == (2,4,4)
                    assert window['rgb_input'].shape == (4,4,3)
                    assert window['candidate_count'].shape == (16,)
                    assert (window['candidate_count'] >= 1).all()
                    assert (window['delta'] >= 0).all()
            probe.close()
        for key,value in state.items():
            torch.testing.assert_close(value,model.state_dict()[key],rtol=0,atol=0)
    padded = align_gt(gt,(5,7),(1,2,3,4),(12,10))
    assert np.all(padded[:3] == 255) and np.all(padded[:,:1] == 255)
    try:
        align_gt(gt,(5,7),(0,0,0,0),(12,10))
        raise AssertionError('Ambiguous crop geometry accepted')
    except ValueError:
        pass
    print('PASS real Route logits/state unchanged; actual crop geometry, weights, metrics, candidate counts')


def paired_counts_and_bundle():
    from .analyse import region_counts, choose_cases, analyse
    from .pipeline import bundle
    from .import_bundle import import_bundle
    from .collect import scan_log
    with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as directory:
        run = Path(directory)/'run'
        (run/'logs').mkdir(parents=True)
        write_json(run/'run.json',dict(baseline_source='synthetic fixture',
                   offseg_checkpoint='fixture-base.pth',route_checkpoint='fixture-route.pth'))
        names = [f'class{i}' for i in range(150)]
        palette = [[i%256,(i*7)%256,(i*19)%256] for i in range(150)]
        images = []
        matrices = [np.zeros((150,150),dtype='int64') for _ in range(2)]
        from .common import confusion,class_metrics
        for i in range(2):
            key = 'sample'+str(i)
            image = Path(directory)/(key+'.jpg')
            Image.new('RGB',(20,20),'white').save(image)
            images.append(dict(id=key,index=i,image=str(image),shape=[20,20]))
            gt = np.zeros((20,20),dtype='uint8')
            gt[:,10:] = 1
            gt[0] = 255
            base,route = gt.copy(),gt.copy()
            base[0],route[0] = 0,0
            base[5,5] = 1
            if i == 1:
                route[5,5] = 1
                route[6,6] = 1
            counts = region_counts(gt,base,route,150)['all']
            assert counts['corrected'] == (1 if i == 0 else 0)
            assert counts['damaged'] == (0 if i == 0 else 1)
            assert counts['pixels'] == 380
            for k,(label,pred) in enumerate(zip(('offseg','route'),(base,route))):
                for folder,array in (('predictions',pred),('gt',gt)):
                    dest = run/label/folder
                    dest.mkdir(parents=True,exist_ok=True)
                    Image.fromarray(array).save(dest/(key+'.png'))
                matrices[k] += confusion(gt,pred,150)
            dest = run/'route/memory'
            dest.mkdir(exist_ok=True)
            np.savez_compressed(dest/(key+'.npz'),gt_area=np.array([[200,180]+[0]*148]),
                                valid_pixels=np.array([380]),lam=np.full((1,150),.6),
                                support=np.full((1,150),16.0))
        for label,matrix in zip(('offseg','route'),matrices):
            _,score = class_metrics(matrix,names)
            write_json(run/label/'evaluation.json',dict(images=images,classes=names,palette=palette,
                 test_cfg=dict(mode='slide'),scope='subset',miou=score,expected_check='NOT_REQUESTED'))
        analyse(run)
        comparison = json.loads((run/'comparison/comparison.json').read_text())
        assert comparison['regions']['all']['corrected'] == 1
        assert comparison['regions']['all']['damaged'] == 1
        assert comparison['delta'] == 0
        if '--plot' in sys.argv:
            for row in images:
                folder=run/'mechanism/mechanisms'/row['id']
                folder.mkdir(parents=True)
                write_json(folder/'geometry.json',dict(img_shape=[20,20],padding=[0,0,0,0],input_shape=[20,20]))
                np.savez_compressed(folder/'crop_000.npz',feature_shape=np.array([4,4]),
                    box=np.array([0,0,20,20]),class_ids=np.array([0,1]),gt_grid=np.zeros((4,4),dtype='uint8'),
                    responsibility=np.full((2,16),1/16),spatial_weight=np.full((2,16),1/16),
                    candidate_count=np.ones(16,dtype='uint8'),delta=np.ones((2,16)),
                    delta_margin=np.zeros((2,16)),metric4x4=np.stack([np.eye(4)]*2))
            from .plot import render
            render(run)
            assert (run/'figures/per_class_extremes.pdf').is_file()
        log = run/'logs/fixture.log'
        log.write_text('Iter(train) [8000/160000] acc_proto_n0: 192.1\n'
                       'Iter(val) [2000/2000] mIoU: 40.10\n'
                       'The best checkpoint with 40.10 mIoU at 8000 iter\n'
                       'Iter(train) [16000/160000]\nIter(val) [2000/2000] mIoU: 41.20\n')
        parsed = scan_log(log)
        assert [(r['iteration'],r['miou']) for r in parsed['curves']] == [(8000,40.1),(16000,41.2)]
        assert parsed['last_iter'] == 16000 and parsed['best']['iteration'] == 8000
        assert parsed['final_logged_needles']['acc_proto_n0'] == 192.1
        bundle(run,[dict(stage='fixture',returncode=0)])
        imported = import_bundle(run/'thesis_evidence.zip',Path(directory)/'imported')
        assert (imported/'comparison/regions.csv').is_file()
        with zipfile.ZipFile(run/'thesis_evidence.zip','a') as z:
            z.writestr('../escape.txt','invalid')
        try:
            import_bundle(run/'thesis_evidence.zip',Path(directory)/'bad')
            raise AssertionError('Unsafe archive accepted')
        except ValueError:
            assert not (Path(directory)/'bad').exists()
    print('PASS paired metric oracles, gains and regressions, logged steps, manifest/hash import, ZIP traversal rejection')


def submission():
    sys.path.insert(0,str(ROOT/'tools/slurm'))
    import submit_thesis_evidence as task
    from submit import snapshot
    from contextlib import redirect_stdout,redirect_stderr
    with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as folder:
        root=Path(folder).resolve()
        route,base=root/'route.pth',root/'base.pth'
        route.write_bytes(b'route-fixture')
        base.write_bytes(b'baseline-fixture')
        write_json(base.with_suffix('.source.json'),dict(sha256=sha256(base),source='official_download'))
        archive=io.BytesIO()
        names=['tools/thesis_evidence/pipeline.py','tools/slurm/thesis_evidence.slurm',
               'tools/slurm/run_thesis_evidence.sh','docs/thesis_handoff/README_FOR_CLAUDE.md',
               task.ROUTE_CONFIG,task.OFFSEG_CONFIG]
        with zipfile.ZipFile(archive,'w') as z:
            for name in names:
                z.writestr(name,(ROOT/name).read_bytes())
        calls=[]
        def command(argv):
            calls.append(argv)
            return '12345' if argv[0] == 'sbatch' else 'fixture-sha'
        argv=['submit_thesis_evidence.py','--route-checkpoint',str(route),'--offseg-checkpoint',str(base),
              '--runs-root',str(root/'runs')]
        with patch.object(sys,'argv',argv),patch.object(task,'command',command), \
                patch.object(task,'active_names',return_value=set()), \
                patch.object(task,'snapshot',side_effect=lambda data,dest:snapshot(data,dest,repo=root)), \
                patch('subprocess.check_output',return_value=archive.getvalue()):
            task.main()
        records=list((root/'runs').glob('*/run.json'))
        assert len(records) == 1
        meta=json.loads(records[0].read_text())
        assert Path(meta['route_checkpoint']).read_bytes() == route.read_bytes()
        assert Path(meta['offseg_checkpoint']).parent.name == 'protected_weights'
        assert meta['route_original_checkpoint'] == str(route)
        assert len([c for c in calls if c[0]=='sbatch']) == 1
        assert all('train.py' not in ' '.join(c) for c in calls)
        with patch.object(sys,'argv',argv+['--dry-run']),patch.object(task,'command',command),redirect_stdout(io.StringIO()):
            task.main()
        assert len(list((root/'runs').glob('*/run.json'))) == 1
        with patch.object(sys,'argv',argv),patch.object(task,'active_names',return_value={'os2_thesis_evidence_b_ade'}),redirect_stderr(io.StringIO()):
            try:
                task.main()
                raise AssertionError('Duplicate active evidence job accepted')
            except SystemExit as exc:
                assert exc.code == 2
    print('PASS one mocked sbatch, immutable source, protected checkpoint copies, dry-run, duplicate rejection')


if __name__ == '__main__':
    paired_counts_and_bundle()
    submission()
    # Real head fixture stubs MMEngine; keep it last.
    frozen_route()
