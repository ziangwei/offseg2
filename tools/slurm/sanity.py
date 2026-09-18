"""Local config and mocked-scheduler checks; does not allocate GPUs."""
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('submit', ROOT / 'tools/slurm/submit.py')
submit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(submit)


def configs():
    from mmengine.config import Config
    def read(path):
        return Config.fromfile(str(ROOT / path), import_custom_modules=False).to_dict()
    route = read('local_configs/offseg2/Base/offsegccmiacs_protoroute_r4_responsibility_ade20k_160k-512x512.py')
    for kind in ('relation', 'dispersion', 'recollect'):
        cfg = read(submit.CATALOG[kind + '_b_ade']['config'])
        normal = copy.deepcopy(cfg)
        for key in ('custom_imports', 'work_dir'):
            normal[key] = route[key]
        normal['model']['decode_head']['type'] = route['model']['decode_head']['type']
        assert normal == route
    tiny = read(submit.CATALOG['proto_t_stuff']['config'])
    control = read('local_configs/offseg2/Tiny/offsegccmiacs_protoroute_r4_responsibility_stuff164k_80k-512x512.py')
    normal = copy.deepcopy(tiny)
    normal['work_dir'] = control['work_dir']
    normal['model']['decode_head']['type'] = control['model']['decode_head']['type']
    assert normal == control
    assert tiny['model']['backbone']['type'] == 'efficientformerv2_s1_feat'
    city = read(submit.CATALOG['offseg_b_city']['config'])
    city_route = read('local_configs/offseg2/Base/offsegccmiacs_protoroute_r4_responsibility_cityscapes_160k-1024x1024.py')
    for key in ('train_dataloader', 'val_dataloader', 'test_dataloader', 'train_cfg',
                'val_cfg', 'test_cfg', 'param_scheduler', 'randomness', 'env_cfg'):
        assert city[key] == city_route[key], key
    assert city['model']['backbone'] == city_route['model']['backbone']
    assert city['model']['test_cfg'] == city_route['model']['test_cfg']
    assert city['optim_wrapper']['optimizer'] == city_route['optim_wrapper']['optimizer']
    assert city['model']['decode_head']['type'] == 'OffSegHead'
    assert city['train_dataloader']['batch_size'] == 2
    dirs = set()
    for name in submit.ROUND1:
        item = submit.CATALOG[name]
        cfg = read(item['config'])
        assert cfg['randomness']['seed'] == item['seed']
        assert cfg['load_from'] is None and cfg['resume'] is False
        assert '_b_' in item['config'] or '_t_' in item['config']
        assert cfg['work_dir'] not in dirs
        dirs.add(cfg['work_dir'])
    assert len(dirs) == 5
    print('PASS five resolved configs: protocol parity, B/T names, seeds and isolated work dirs')


def scheduler():
    (ROOT / 'tmp').mkdir(exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix='slurm_sanity_', dir=ROOT / 'tmp')).resolve()
    assert ROOT.resolve() in temp.parents
    archive = io.BytesIO()
    files = ['tools/slurm/run_job.sh', 'tools/slurm/preflight.py']
    files += [v[k] for v in submit.CATALOG.values() for k in ('config', 'script')]
    with zipfile.ZipFile(archive, 'w') as z:
        for name in files:
            z.writestr(name, (ROOT / name).read_bytes())
    calls, queue = [], []
    def command(args, **kwargs):
        if args[:2] == ['git', 'rev-parse']:
            return 'test-commit'
        if args[0] == 'squeue':
            return '\n'.join(queue)
        assert args[0] == 'sbatch'
        calls.append(args)
        return str(100 + len(calls))
    original_snapshot = submit.snapshot
    original_freeze = submit.freeze_assets
    def snapshot(data, dest):
        # No dataset assets are necessary for this scheduler-only fixture.
        return original_snapshot(data, dest, repo=temp)
    def run(*args):
        with patch.object(sys, 'argv', ['submit.py', *args]), contextlib.redirect_stdout(io.StringIO()):
            submit.main()
    with patch.object(submit, 'command', command), patch.object(submit, 'snapshot', snapshot), \
         patch.object(submit, 'ROOT', temp), \
         patch.object(submit, 'freeze_assets', lambda meta: original_freeze(meta, repo=temp)), \
         patch.object(submit.subprocess, 'check_output', return_value=archive.getvalue()):
        run('all', '--runs-root', str(temp), '--dry-run')
        assert not calls and not list(temp.glob('*/run.json'))
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                run('text2', '--runs-root', str(temp))
            raise AssertionError('Missing text asset accepted')
        except SystemExit as exc:
            assert exc.code == 2
        assert not calls
        # Fixture only inside the temporary repository, never a real text asset.
        for name in {a for v in submit.CATALOG.values() for a in v.get('assets', [])}:
            asset = temp / name
            asset.parent.mkdir(parents=True, exist_ok=True)
            asset.write_bytes(b'frozen asset fixture')
        run('all', '--runs-root', str(temp))
        assert len(calls) == 7
        assert len({next(a for a in c if a.startswith('--chdir=')) for c in calls}) == 7
        assert len({c[c.index('--work-dir') + 1] for c in calls}) == 7
        records = list(temp.glob('*/*/run.json'))
        assert len(records) == 7
        for record in records:
            item = json.loads(record.read_text())
            for name, digest in item.get('asset_sha256', {}).items():
                frozen_asset = record.parent / 'source' / name
                assert frozen_asset.read_bytes() == b'frozen asset fixture'
                assert len(digest) == 64 and not frozen_asset.is_symlink()
        run_dir = records[0].parent
        meta = json.loads(records[0].read_text())
        frozen = run_dir / 'source' / meta['config']
        before = frozen.read_bytes()
        run('--resume-run', str(run_dir))
        assert calls[-1][-1] == '--resume'
        run('--evaluate-run', str(run_dir))
        assert calls[-1][-1] == '--eval-last'
        assert frozen.read_bytes() == before
        assert len(json.loads(records[0].read_text())['attempts']) == 3
        queue.append('os2_relation_b_ade')
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                run('relation', '--runs-root', str(temp))
            raise AssertionError('Duplicate active job was accepted')
        except SystemExit as exc:
            assert exc.code == 2
        assert len(calls) == 9
    bash = Path('C:/Program Files/Git/bin/bash.exe') if sys.platform == 'win32' else Path('/bin/bash')
    for name in ['tools/slurm/run_job.sh'] + [v['script'] for v in submit.CATALOG.values()]:
        text = (ROOT / name).read_text()
        assert '\\\\\n' not in text, 'Double backslash breaks shell continuation'
        subprocess.run([str(bash), '-n'], input=text, text=True, check=True)
        if name.endswith('.slurm'):
            for directive in ('--ntasks=1', '--gres=gpu:4', '--time=48:00:00', '--partition=mcml-hgx-a100-80x4'):
                assert directive in text
    print('PASS seven isolated jobs, missing-asset guard, frozen asset hashes, duplicate blocking, resume/eval and Bash syntax')


if __name__ == '__main__':
    configs()
    scheduler()
