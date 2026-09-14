"""Submit ONE allocation per experiment, with immutable tracked-code snapshots.

Stdlib only: run this on the LRZ login node, not on an allocated GPU node.
"""
import argparse
from datetime import datetime
import getpass
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import uuid
import zipfile

ROOT = Path(__file__).resolve().parents[2]
CATALOG = json.loads((Path(__file__).with_name('experiments.json')).read_text())
DEFAULT_CONDA = '/dss/dssmcmlfs01/pn39qo/pn39qo-dss-0000/di97fer/miniconda3'


def command(args, cwd=ROOT):
    return subprocess.check_output(args, cwd=str(cwd), text=True).strip()


def select_jobs(selection):
    if selection == 'all':
        return list(CATALOG)
    if selection == 'evidence':
        return ['offseg_b_city', 'proto_t_stuff']
    if selection == 'structures':
        return ['relation_b_ade', 'dispersion_b_ade', 'recollect_b_ade']
    if selection == 'new':
        return ['dispersion_b_ade', 'recollect_b_ade']
    if selection == 'relation':
        return ['relation_b_ade']
    if selection in CATALOG:
        return [selection]
    raise ValueError('Unknown experiment/group: ' + selection)


def sbatch_command(meta, mode='train'):
    run = Path(meta['run_dir'])
    args = ['sbatch', '--parsable', '--job-name=os2_' + meta['experiment'],
            '--chdir=' + str(run / 'source'),
            '--output=' + str(run / 'logs' / 'slurm-%j.log'),
            '--error=' + str(run / 'logs' / 'slurm-%j.log'),
            '--open-mode=append', '--export=ALL',
            str(run / 'source' / meta['script']),
            '--work-dir', meta['work_dir']]
    if mode == 'resume':
        args.append('--resume')
    elif mode == 'eval':
        args.append('--eval-last')
    return args


def snapshot(archive, dest, repo=ROOT):
    dest.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        for name in z.namelist():
            target = (dest / name).resolve()
            if dest.resolve() not in target.parents and target != dest.resolve():
                raise ValueError('Archive path escapes snapshot: ' + name)
        z.extractall(dest)
    # Shared datasets and backbone weights are not copied into each job.
    for name in ('data', 'pretrained'):
        asset = repo / name
        if asset.exists() and not (dest / name).exists():
            (dest / name).symlink_to(asset.resolve(), target_is_directory=True)


def save(meta):
    path = Path(meta['run_dir']) / 'run.json'
    temp = path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(meta, indent=2) + '\n')
    temp.replace(path)


def active_names():
    return set(command(['squeue', '--noheader', '--user', getpass.getuser(),
                        '--format=%j']).splitlines())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('selection', nargs='?', default='relation',
                        help='relation, new (two), structures (three), evidence (two), all (five), or exact experiment ID')
    parser.add_argument('--dry-run', action='store_true', help='Print commands without submitting or writing snapshots')
    parser.add_argument('--runs-root', type=Path, default=ROOT / 'work_dirs/slurm_runs')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--resume-run', type=Path, help='Resume the snapshot/work directory in RUN/run.json')
    modes.add_argument('--evaluate-run', type=Path, help='Only evaluate the final checkpoint in RUN/run.json')
    parser.add_argument('--resume-work-dir', type=Path,
                        help='Single experiment only: continue an existing legacy work directory')
    args = parser.parse_args()
    mode = 'train'
    existing = args.resume_run or args.evaluate_run
    if existing:
        if args.resume_work_dir:
            parser.error('--resume-work-dir cannot be combined with an existing run')
        meta = json.loads((existing.resolve() / 'run.json').read_text())
        if Path(meta['run_dir']).resolve() != existing.resolve():
            parser.error('Run directory moved; inspect and correct run.json paths first')
        mode = 'resume' if args.resume_run else 'eval'
        jobs = [meta['experiment']]
        prepared = [meta]
    else:
        jobs = select_jobs(args.selection)
        if args.resume_work_dir and len(jobs) != 1:
            parser.error('Legacy resume requires exactly one experiment')
        if args.resume_work_dir:
            if not (args.resume_work_dir / 'last_checkpoint').is_file():
                parser.error('Legacy work directory has no last_checkpoint')
            mode = 'resume'
        sha = command(['git', 'rev-parse', 'HEAD'])
        batch = args.runs_root.resolve() / (datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6])
        prepared = []
        for name in jobs:
            run = batch / name
            prepared.append(dict(CATALOG[name], experiment=name, git_sha=sha,
                run_dir=str(run), work_dir=str(args.resume_work_dir.resolve() if args.resume_work_dir else run / 'checkpoints'),
                conda_base=os.environ.get('OFFSEG_CONDA_BASE', DEFAULT_CONDA),
                conda_env=os.environ.get('OFFSEG_CONDA_ENV', 'offseg_new2'),
                python_override=os.environ.get('OFFSEG_PYTHON', ''),
                partition='mcml-hgx-a100-80x4', qos='mcml', gpus=4, time='48:00:00'))
    if args.dry_run:
        for meta in prepared:
            print(shlex.join(sbatch_command(meta, mode)))
        print('DRY RUN: {} separate jobs, each 4 GPUs / 48 hours.'.format(len(prepared)))
        return
    # Fail before creating snapshots if Slurm is unavailable. Do not silently
    # submit duplicate active experiment names, including across different SHAs.
    active = active_names()
    duplicate = [name for name in jobs if 'os2_' + name in active]
    if duplicate:
        parser.error('Already pending/running: ' + ', '.join(duplicate))
    if not existing:
        archive = subprocess.check_output(['git', 'archive', '--format=zip', sha], cwd=str(ROOT))
        with zipfile.ZipFile(io.BytesIO(archive)) as z:
            for meta in prepared:
                for name in (meta['config'], meta['script'], 'tools/slurm/run_job.sh', 'tools/slurm/preflight.py'):
                    if name not in z.namelist():
                        parser.error('Not in committed snapshot: ' + name + '. Pull the delivered commit first.')
        for meta in prepared:
            run = Path(meta['run_dir'])
            snapshot(archive, run / 'source')
            (run / 'logs').mkdir()
            if not args.resume_work_dir:
                Path(meta['work_dir']).mkdir()
            # All subsequent starts, including resume, read the same environment
            # choice rather than whatever conda environment the login shell uses.
            env = '\n'.join('export {}={}'.format(key, shlex.quote(value)) for key, value in (
                ('OFFSEG_CONDA_BASE', meta['conda_base']), ('OFFSEG_CONDA_ENV', meta['conda_env']),
                ('OFFSEG_PYTHON', meta['python_override']))) + '\n'
            (run / 'source' / '.offseg_job_env.sh').write_text(env)
            meta['attempts'] = []
            save(meta)
    for meta in prepared:
        cmd = sbatch_command(meta, mode)
        meta['attempts'].append(dict(mode=mode, command=cmd, submitted_at=datetime.now().isoformat(), status='submitting'))
        save(meta)
        try:
            result = command(cmd)
            job_id = result.split(';', 1)[0]
            if not job_id.isdigit():
                raise RuntimeError('Ambiguous sbatch response, inspect squeue before retrying: ' + result)
        except Exception as exc:
            meta['attempts'][-1].update(status='submission_error', error=str(exc))
            save(meta)
            raise  # Earlier successful jobs remain queued and recorded.
        meta['attempts'][-1].update(status='submitted', job_id=job_id)
        save(meta)
        print('{}: job {}\n  run: {}\n  checkpoints: {}'.format(meta['experiment'], job_id, meta['run_dir'], meta['work_dir']), flush=True)


if __name__ == '__main__':
    main()
