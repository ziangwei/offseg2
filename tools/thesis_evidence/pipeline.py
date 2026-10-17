"""Orchestrate read-only evidence stages and always package a status manifest."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import platform
import subprocess
import sys
import zipfile
from .common import read_json, write_json, sha256


def bundle(run, statuses):
    status = 'COMPLETE' if statuses and all(s['returncode'] == 0 for s in statuses) else 'PARTIAL'
    write_json(run/'status.json', dict(status=status, stages=statuses,
                                     finished_utc=datetime.now(timezone.utc).isoformat()))
    manifest = []
    folders = ('handoff','offseg','route','mechanism','comparison','selected_images','figures','cost','logs')
    files = [run/'run.json', run/'status.json', run/'environment.json',run/'stage_status.json']
    for folder in folders:
        files.extend(p for p in (run/folder).rglob('*') if p.is_file() and not p.is_symlink())
    files = sorted(set(p for p in files if p.is_file()))
    for path in files:
        # Never package checkpoints, optimizer state, event blobs or input datasets.
        if (path.suffix.lower() in ('.pth','.pt','.pkl','.bin') or 'vis_data' in path.parts
                or path.name.startswith('slurm-')):
            continue
        manifest.append(dict(path=path.relative_to(run).as_posix(), bytes=path.stat().st_size,
                             sha256=sha256(path)))
    write_json(run/'manifest.json', dict(status=status, files=manifest,
        note='No model weights are included. Numerical summaries, selected cases, raw class predictions, '
             'figure data, source labels and Claude handoff are included. A partial package is not a completed result.'))
    archive = run/'thesis_evidence.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        z.write(run/'manifest.json', 'manifest.json')
        for item in manifest:
            z.write(run/item['path'], item['path'])
    print(f'{status}: download {archive}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('record', type=Path)
    args = parser.parse_args()
    meta = read_json(args.record)
    run = Path(meta['run_dir'])
    import torch
    import cv2
    import matplotlib
    write_json(run/'environment.json', dict(python=platform.python_version(), torch=torch.__version__,
        cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(), cv2=cv2.__version__,
        matplotlib=matplotlib.__version__, gpu=torch.cuda.get_device_name(0),
        job_id=os.environ.get('SLURM_JOB_ID'), git_sha=meta['git_sha']))
    statuses = []

    def stage(name, argv):
        print('STAGE:', name, flush=True)
        with (run/'logs'/f'{name}.log').open('a', encoding='utf-8') as log:
            result = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT)
        statuses.append(dict(stage=name, returncode=result.returncode, command=argv))
        write_json(run/'stage_status.json', statuses)
        if result.returncode:
            print(f'FAILED {name}; see {run/"logs"/(name+".log")}', flush=True)
        return result.returncode == 0

    def export(label, output, distributed=True, extra=()):
        cmd = [sys.executable]
        if distributed:
            cmd += ['-m','torch.distributed.run','--rdzv-backend=c10d','--rdzv-endpoint=localhost:0',
                    '--rdzv-id=thesis-'+str(os.environ.get('SLURM_JOB_ID','local'))+'-'+output,
                    '--nnodes=1','--nproc_per_node=4','--max_restarts=0', '--module']
        else:
            cmd += ['-m']
        cmd += ['tools.thesis_evidence.export', meta[label+'_config'], meta[label+'_checkpoint'],
                '--output',str(run/output), '--label',label]
        if distributed:
            cmd += ['--launcher','pytorch']
        if label == 'route' or meta['baseline_source'] == 'local_training':
            cmd += ['--owner-training-checkpoint']
        return cmd+list(extra)

    try:
        for label in ('route','offseg'):
            if sha256(meta[label+'_checkpoint']) != meta[label+'_sha256']:
                raise RuntimeError('Protected checkpoint hash changed: '+label)
        if not stage('collect', [sys.executable,'-m','tools.thesis_evidence.collect',
                                '--repo',meta['repo_root'],'--output',str(run/'handoff')]):
            return
        # A small real-data pilot catches key, shape, crop and annotation problems.
        for label in ('offseg','route'):
            if not stage('pilot_'+label, export(label, 'pilot_'+label, extra=('--limit','20'))):
                return
        for label in ('offseg','route'):
            extra = ('--expected-miou','48.49') if label == 'route' else ()
            if not stage('full_'+label, export(label,label,extra=extra)):
                return
        if not stage('compare', [sys.executable,'-m','tools.thesis_evidence.analyse',str(run)]):
            return
        maps_ok = stage('mechanisms', export('route','mechanism',distributed=False,
                        extra=('--cases',str(run/'comparison/cases.json'))))
        # Independent cost measurement has no observation hooks or serialization.
        cost_cmd = [sys.executable,'tools/route_cost_audit_b.py', meta['route_config'],
                       meta['route_checkpoint'],'--offseg-checkpoint',meta['offseg_checkpoint'],
                       '--offseg-config',meta['offseg_config'],'--work-dir',str(run/'cost')]
        if meta['baseline_source'] == 'local_training':
            cost_cmd += ['--offseg-owner-training-checkpoint']
        stage('cost',cost_cmd)
        if maps_ok:
            stage('figures', [sys.executable,'-m','tools.thesis_evidence.plot',str(run)])
    except Exception as exc:
        statuses.append(dict(stage='orchestrator',returncode=1,error=f'{type(exc).__name__}: {exc}'))
        raise
    finally:
        bundle(run,statuses)
        if not statuses or any(s['returncode'] for s in statuses):
            raise SystemExit(1)


if __name__ == '__main__':
    main()
