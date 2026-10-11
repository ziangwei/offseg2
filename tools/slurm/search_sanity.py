"""Check Route search recipes and seed propagation without allocating GPUs."""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch
import zipfile

import submit
from preflight import configure_seed


def configs():
    from mmengine.config import Config
    for name in submit.ROUTE_SEARCH:
        item = submit.CATALOG[name]
        cfg = Config.fromfile(str(submit.ROOT / item['config']),
                              import_custom_modules=False)
        size = 'l' if '_l_' in name else 't'
        folder = 'Large' if size == 'l' else 'Tiny'
        data = 'ade20k' if name.endswith('_ade') else 'cityscapes'
        crop, batch = (512, 4) if data == 'ade20k' else (1024, 2)
        parent = Config.fromfile(str(submit.ROOT /
            f'local_configs/offseg2/{folder}/offsegccmiacs_protoroute_{size}_r4_responsibility_{data}_160k-{crop}x{crop}.py'),
            import_custom_modules=False).to_dict()
        normal = copy.deepcopy(cfg.to_dict())
        normal['randomness'] = parent['randomness']
        normal['work_dir'] = parent['work_dir']
        if 'lr4' in name:
            assert cfg.optim_wrapper.optimizer.lr == 4e-5
            normal['optim_wrapper']['optimizer']['lr'] = 6e-5
        else:
            assert cfg.optim_wrapper.optimizer.lr == 6e-5
        assert normal == parent, name + ': unintended config difference'
        assert cfg.randomness.seed is None
        assert item['seed_policy'] == 'random_unique'
        assert cfg.train_cfg.max_iters == cfg.param_scheduler[-1]['end'] == 160000
        assert cfg.train_dataloader.batch_size == batch
        assert tuple(cfg.model.data_preprocessor.size) == (crop, crop)
        hook = cfg.default_hooks.checkpoint
        assert hook.interval == cfg.train_cfg.val_interval == 8000
        assert hook.save_best == 'mIoU' and hook.max_keep_ckpts == 2
        assert cfg.model.decode_head.type == 'OffSegCCMIACSProtoRoute'
        try:
            configure_seed(cfg)
            raise AssertionError('Unrecorded seed accepted')
        except ValueError:
            pass
        configure_seed(cfg, 123456)
        assert cfg.randomness.seed == 123456
        for bad in (0, -1, 2**31, True):
            try:
                configure_seed(cfg, bad)
                raise AssertionError('Invalid seed accepted')
            except ValueError:
                pass
        print('PASS recipe:', name)


def scheduler():
    root = submit.ROOT
    temp = Path(tempfile.mkdtemp(prefix='route_search_', dir=root / 'tmp')).resolve()
    history = temp / 'old' / 'historical'
    history.mkdir(parents=True)
    (history / 'run.json').write_text(json.dumps({'seed': 111}))
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, 'w') as z:
        files = ['tools/slurm/run_job.sh', 'tools/slurm/preflight.py']
        files += [submit.CATALOG[name][key] for name in submit.ROUTE_SEARCH
                  for key in ('config', 'script')]
        for name in files:
            z.writestr(name, (root / name).read_bytes())
    for name in {submit.CATALOG[n]['backbone_checkpoint'] for n in submit.ROUTE_SEARCH}:
        path = temp / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'asset-existence fixture only')
    calls = []
    def command(args, **kwargs):
        if args[0] == 'git':return 'test-commit'
        if args[0] == 'squeue':return ''
        assert args[0] == 'sbatch'
        calls.append(args)
        return str(200 + len(calls))
    old_snapshot = submit.snapshot
    def snapshot(data, dest):
        return old_snapshot(data, dest, repo=temp / 'no_assets')
    def run(*args):
        with patch.object(sys, 'argv', ['submit.py', *args]), contextlib.redirect_stdout(io.StringIO()):
            submit.main()
    # Deterministically force collisions with known/history/in-batch seeds.
    # randbelow returns candidate-1: 111 is history, 2026 is reserved.
    draws = [110, 2025, 120, 120, 121, 122, 123, 124, 125, 126]
    with patch.object(submit, 'ROOT', temp), patch.object(submit, 'command', command), \
         patch.object(submit, 'snapshot', snapshot), \
         patch.object(submit.subprocess, 'check_output', return_value=archive.getvalue()), \
         patch.object(submit.secrets, 'randbelow', side_effect=draws):
        run('routesearch', '--runs-root', str(temp))
        assert len(calls) == 7
        records = [p for p in temp.glob('*/*/run.json') if p.parent.name in submit.ROUTE_SEARCH]
        seeds = {json.loads(p.read_text())['seed'] for p in records}
        assert seeds == set(range(121, 128))
        assert len({json.loads(p.read_text())['work_dir'] for p in records}) == 7
        assert len({next(a for a in c if a.startswith('--chdir=')) for c in calls}) == 7
        for record in records:
            meta = json.loads(record.read_text())
            cmd = meta['attempts'][0]['command']
            assert cmd[cmd.index('--seed') + 1] == str(meta['seed'])
            assert '--time=36:00:00' in cmd and meta['gpus'] == 4
        record = records[0]
        meta = json.loads(record.read_text())
        frozen = record.parent / 'source' / meta['config']
        before = frozen.read_bytes()
        run('--resume-run', str(record.parent))
        assert calls[-1][-1] == '--resume'
        assert calls[-1][calls[-1].index('--seed') + 1] == str(meta['seed'])
        run('--evaluate-run', str(record.parent))
        assert calls[-1][-1] == '--eval-last'
        assert calls[-1][calls[-1].index('--seed') + 1] == str(meta['seed'])
        assert frozen.read_bytes() == before
        assert json.loads(record.read_text())['seed'] == meta['seed']
        # No further random draw is available: resume/eval would fail if redrawn.
    assert submit.select_jobs('round6') == submit.select_jobs('new') == submit.ROUTE_SEARCH
    assert submit.select_jobs('citycontrols') == submit.CITY_CONTROLS
    print('PASS seven independent jobs, collision rejection, persisted seeds, resume/eval preservation')


def seed_cli():
    """Execute the real shell's argument assembly with a fake Python launcher."""
    root = submit.ROOT
    bash = 'C:/Program Files/Git/bin/bash.exe' if sys.platform == 'win32' else '/bin/bash'
    for name in ['tools/slurm/run_job.sh'] + [submit.CATALOG[n]['script'] for n in submit.ROUTE_SEARCH]:
        subprocess.run([bash, '-n'], input=(root / name).read_text(), text=True, check=True)
    temp = Path(tempfile.mkdtemp(prefix='route_search_cli_', dir=root / 'tmp')).resolve()
    (temp / 'tools').mkdir()
    (temp / 'tools/train.py').touch()
    work = temp / 'checkpoints'
    work.mkdir()
    # The fake Python records the real shell argv, without importing Torch or
    # launching workers. -c metadata reads are executed by the real interpreter.
    fake = temp / 'fake_python.py'
    fake.write_text('''import json,os,sys
from pathlib import Path
args=sys.argv[1:]
if args[0]=='-c':
    code=args[1];sys.argv=['-c']+args[2:];exec(code)
elif args[0]=='tools/slurm/preflight.py':
    seed=int(args[args.index('--seed')+1])
    record=dict(work_dir=os.environ['FIXTURE_WORK'],seed=seed,
        final_checkpoint=os.environ['FIXTURE_WORK']+'/iter_160000.pth',
        resume_checkpoint=(os.environ['FIXTURE_WORK']+'/iter_8000.pth' if '--resume' in args else None))
    Path(args[args.index('--output')+1]).write_text(json.dumps(record))
    with open(os.environ['FIXTURE_TRACE'],'a') as f:f.write(json.dumps(args)+'\\n')
else:
    with open(os.environ['FIXTURE_TRACE'],'a') as f:f.write(json.dumps(args)+'\\n')
''')
    shim = temp / 'fake_python.sh'
    shim.write_text(f'#!/bin/bash\n"{Path(sys.executable).as_posix()}" "{fake.as_posix()}" "$@"\n')
    shim.chmod(0o755)
    script = (root / 'tools/slurm/run_job.sh').read_text()
    # Git Bash on Windows need not bundle flock. Lock behaviour is unchanged;
    # here the mock only allows the argument-propagation fixture to proceed.
    if sys.platform == 'win32':script = 'function flock() { return 0; }\n' + script
    trace = temp / 'trace.jsonl'
    env = dict(os.environ, OFFSEG_PYTHON=shim.as_posix(), SLURM_JOB_ID='123',
               FIXTURE_WORK=work.as_posix(), FIXTURE_TRACE=trace.as_posix(),
               TMPDIR=temp.as_posix())
    for mode in ([], ['--resume'], ['--eval-last']):
        trace.write_text('')
        result = subprocess.run([bash, '-s', '--', 'config.py', '--work-dir', work.as_posix(),
                                 '--seed', '123456', *mode], input=script, text=True,
                                cwd=temp, env=env, capture_output=True)
        assert result.returncode == 0, result.stdout + result.stderr
        calls = [json.loads(line) for line in trace.read_text().splitlines()]
        assert calls[0][calls[0].index('--seed') + 1] == '123456'
        launched = calls[-1]
        assert launched.count('--cfg-options') == 1
        assert 'randomness.seed=123456' in launched
        assert '--rdzv-endpoint=localhost:0' in launched
        assert '--nproc_per_node=4' in launched
        overrides = launched[launched.index('--cfg-options') + 1:]
        if '--resume' in mode:
            assert 'load_from='+work.as_posix()+'/iter_8000.pth' in overrides
            assert '--resume' in launched
        else:
            assert not any(arg.startswith('load_from=') for arg in overrides)
        assert ('tools/test.py' in launched) == ('--eval-last' in mode)
        assert json.loads((work / 'launch-123.json').read_text())['seed'] == 123456
    print('PASS actual shell train/resume/eval argv: one seed for all workers and one combined override')


if __name__ == '__main__':
    configs()
    scheduler()
    seed_cli()
