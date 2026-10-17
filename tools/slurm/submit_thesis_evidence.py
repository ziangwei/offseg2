"""Submit one 4-hour read-only thesis evidence job; no training allocation."""
import argparse
from datetime import datetime
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import uuid
import zipfile

from submit import ROOT, DEFAULT_CONDA, active_names, command, snapshot
sys.path.insert(0, str(ROOT))
from tools.thesis_evidence.common import ROUTE_CONFIG, OFFSEG_CONFIG, sha256, write_json, read_json

DEFAULT_ROUTE = ROOT/'work_dirs/offsegccmiacs_protoroute_r4_responsibility_ade20k_160k-512x512/best_mIoU_iter_160000.pth'
DEFAULT_BASE = ROOT/'pretrained/thesis/offseg_b_ade_official.pth'


def protect(source, folder):
    digest = sha256(source)
    destination = folder/(digest[:16]+'_'+source.name)
    folder.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        temporary = destination.with_suffix('.copying')
        shutil.copy2(source, temporary)
        if sha256(temporary) != digest:
            raise RuntimeError('Checkpoint changed while being copied')
        temporary.replace(destination)
    elif sha256(destination) != digest:
        raise RuntimeError('Protected checkpoint collision')
    return destination, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--route-checkpoint', type=Path, default=DEFAULT_ROUTE)
    parser.add_argument('--offseg-checkpoint', type=Path, default=DEFAULT_BASE)
    parser.add_argument('--baseline-source', choices=('official_download','local_training'), default='official_download')
    parser.add_argument('--runs-root', type=Path, default=ROOT/'work_dirs/thesis_evidence')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    for path in (args.route_checkpoint,args.offseg_checkpoint):
        if not path.is_file():
            parser.error('Missing checkpoint: '+str(path)+'; locate Route best or run python -m tools.thesis_evidence.fetch_baseline')
    if not args.dry_run and 'os2_thesis_evidence_b_ade' in active_names():
        parser.error('Evidence job already queued/running; use squeue --me')
    sha = command(['git','rev-parse','HEAD'])
    run = args.runs_root.resolve()/(datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
    meta = dict(repo_root=str(ROOT), git_sha=sha, run_dir=str(run),
                route_config=ROUTE_CONFIG, offseg_config=OFFSEG_CONFIG,
                route_checkpoint=str(args.route_checkpoint.resolve()),
                offseg_checkpoint=str(args.offseg_checkpoint.resolve()),
                baseline_source=args.baseline_source, status='prepared',
                conda_base=DEFAULT_CONDA, conda_env='offseg_new2')
    source_record = args.offseg_checkpoint.with_suffix('.source.json')
    if source_record.is_file():
        meta['official_download'] = read_json(source_record)
        if args.baseline_source != 'official_download' or meta['official_download']['sha256'] != sha256(args.offseg_checkpoint):
            parser.error('Baseline source metadata/hash disagrees with checkpoint')
    elif args.baseline_source == 'official_download':
        parser.error('Official checkpoint needs adjacent .source.json; use fetch_baseline or select --baseline-source local_training for a genuine local run')
    cmd = ['sbatch','--parsable','--job-name=os2_thesis_evidence_b_ade',
           '--chdir='+str(run/'source'),'--output='+str(run/'logs/slurm-%j.log'),
           '--error='+str(run/'logs/slurm-%j.log'),
           str(run/'source/tools/slurm/thesis_evidence.slurm'),str(run/'run.json')]
    if args.dry_run:
        print(json.dumps(dict(metadata=meta,command=cmd),indent=2))
        return
    archive = subprocess.check_output(['git','archive','--format=zip','HEAD'],cwd=ROOT)
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        for name in ('tools/thesis_evidence/pipeline.py','tools/slurm/thesis_evidence.slurm',
                     'tools/slurm/run_thesis_evidence.sh','docs/thesis_handoff/README_FOR_CLAUDE.md',ROUTE_CONFIG,OFFSEG_CONFIG):
            if name not in z.namelist():
                parser.error('Required file not committed: '+name)
    # Independent, full copies live outside slurm_runs and its cleanup script.
    for label in ('route','offseg'):
        original = Path(meta[label+'_checkpoint'])
        protected,digest = protect(original,args.runs_root.resolve()/'protected_weights')
        meta[label+'_original_checkpoint'] = str(original)
        meta[label+'_checkpoint'],meta[label+'_sha256'] = str(protected),digest
    snapshot(archive,run/'source')
    (run/'logs').mkdir()
    write_json(run/'run.json',meta)
    try:
        job = command(cmd).split(';')[0]
        if not job.isdigit():
            raise RuntimeError('Ambiguous sbatch response: '+job+'; inspect queue before retry')
        meta.update(job_id=job,status='submitted')
    except Exception as exc:
        meta.update(status='submission_error',error=str(exc))
        raise
    finally:
        write_json(run/'run.json',meta)
    print(f'Job: {job}\nResults: {run}\nDownload when complete: {run/"thesis_evidence.zip"}')


if __name__ == '__main__':
    main()
