"""Freeze and sbatch one read-only Route error/cost audit; no training jobs."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import uuid
import zipfile
import io

from submit import ROOT, DEFAULT_CONDA, active_names, command, snapshot

CONFIG = 'local_configs/offseg2/Base/offsegccmiacs_protoroute_r4_responsibility_ade20k_160k-512x512.py'
DEFAULT_CHECKPOINT = ROOT / 'work_dirs/offsegccmiacs_protoroute_r4_responsibility_ade20k_160k-512x512/best_mIoU_iter_160000.pth'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--offseg-checkpoint', type=Path)
    parser.add_argument('--runs-root', type=Path, default=ROOT/'work_dirs/slurm_audits')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        parser.error(f'Missing Route best: {checkpoint}; pass --checkpoint /actual/path.pth')
    if args.offseg_checkpoint and not args.offseg_checkpoint.is_file():
        parser.error('Missing --offseg-checkpoint')
    job_name = 'os2_route_audit_b'
    if not args.dry_run and job_name in active_names():
        parser.error('A Route audit is already queued/running; use squeue --me')
    sha = command(['git', 'rev-parse', 'HEAD'])
    run = args.runs_root.resolve() / (datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
    meta = dict(git_sha=sha, run_dir=str(run), config=CONFIG, checkpoint=str(checkpoint),
                offseg_checkpoint=str(args.offseg_checkpoint.resolve()) if args.offseg_checkpoint else None,
                conda_base=DEFAULT_CONDA, conda_env='offseg_new2', status='prepared')
    cmd = ['sbatch', '--parsable', '--job-name='+job_name, '--chdir='+str(run/'source'),
           '--output='+str(run/'logs/slurm-%j.log'), '--error='+str(run/'logs/slurm-%j.log'),
           str(run/'source/tools/slurm/route_audit_b.slurm'), str(run/'run.json')]
    if args.dry_run:
        print(json.dumps(dict(metadata=meta, command=cmd), indent=2))
        return
    import subprocess
    archive = subprocess.check_output(['git', 'archive', '--format=zip', 'HEAD'], cwd=ROOT)
    required = ['tools/route_error_audit_b.py', 'tools/route_cost_audit_b.py',
                'tools/slurm/route_audit_b.slurm', 'tools/slurm/run_audit_b.sh', CONFIG]
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        for name in required:
            if name not in z.namelist():
                parser.error('Not in committed snapshot: '+name)
    snapshot(archive, run/'source')
    for folder in ('logs', 'errors', 'cost'):
        (run/folder).mkdir()
    record = run/'run.json'
    record.write_text(json.dumps(meta, indent=2)+'\n')
    try:
        job = command(cmd).split(';')[0]
        if not job.isdigit():
            raise RuntimeError('Ambiguous sbatch response; inspect queue before retrying: '+job)
        meta.update(job_id=job, status='submitted')
    except Exception as exc:
        meta.update(status='submission_error', error=str(exc))
        raise
    finally:
        record.write_text(json.dumps(meta, indent=2)+'\n')
    print(f'Job: {job}\nResults: {run}\nSend back: {run / "route_audit_b.zip"}')


if __name__ == '__main__':
    main()
