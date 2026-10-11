"""Preview/delete inactive Slurm training weights, preserving logs and source.

Default: only COMPLETED allocations; no recursive deletion, no symlink
following. Run on the login node. --delete is the explicit destructive mode.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import getpass
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
WEIGHT = re.compile(r'(?:iter_\d+|best_.+_iter_\d+)\.(?:pth|pt|ckpt)(?:\.tmp)?$')


def command(args):
    return subprocess.check_output(args, text=True, timeout=60)


def queue():
    rows = command(['squeue', '--noheader', '--user', getpass.getuser(),
                    '--format=%i|%.200j'])
    ids, names = set(), set()
    for row in rows.splitlines():
        job, name = row.split('|', 1)
        ids.add(job.strip())
        names.add(name.strip())
    return ids, names


def states(ids):
    if not ids:
        return {}
    rows = command(['sacct', '--noheader', '--parsable2', '--allocations',
                    '--jobs=' + ','.join(sorted(ids)), '--starttime=2026-01-01',
                    '--format=JobIDRaw,State%40'])
    result = {}
    for row in rows.splitlines():
        job, state = row.split('|', 1)
        result[job.strip()] = state.strip().split()[0].rstrip('+')
    return result


def checked_work(root, record, meta):
    run = record.parent
    work = Path(meta['work_dir'])
    # Only the submitter's run/checkpoints layout is eligible. Legacy external
    # work dirs and copied/moved metadata are retained for explicit handling.
    if (run.is_symlink() or record.is_symlink() or work.is_symlink() or
            run.resolve() != Path(meta['run_dir']).resolve() or
            work.resolve() != (run / 'checkpoints').resolve() or
            root not in work.resolve().parents):
        raise ValueError('external, moved or symlinked work directory')
    for parent in (run.parent, work):
        if parent.is_symlink():
            raise ValueError('symlinked directory')
    return work.resolve()


def blocked(meta, active, accounting, keep, include_incomplete):
    name = meta['experiment']
    if name in keep:
        return 'explicitly kept'
    attempts = meta.get('attempts', [])
    ids = {str(a.get('job_id', '')) for a in attempts}
    if ids & active[0] or 'os2_' + name in active[1]:
        return 'pending/running'
    if not attempts or attempts[-1].get('status') in ('submitting', 'submission_error'):
        return 'ambiguous/unsubmitted attempt'
    training = [a for a in attempts if a.get('mode', 'train') in ('train', 'resume')]
    if not training:
        return 'no recorded training allocation'
    # A successful evaluation alone does not complete an interrupted training.
    last = str(training[-1].get('job_id', ''))
    state = accounting.get(last, 'UNKNOWN')
    if state in ('PENDING', 'RUNNING', 'COMPLETING', 'CONFIGURING',
                 'SUSPENDED', 'REQUEUED', 'RESIZING', 'REQUEUE_HOLD'):
        return 'active accounting state: ' + state
    if state != 'COMPLETED' and not include_incomplete:
        return 'unfinished/unknown: ' + state
    return None


@contextmanager
def work_lock(work):
    import fcntl  # Linux login node only; shared with run_job.sh's flock lock.
    with (work / '.slurm-training.lock').open('a+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def weights(work):
    return sorted(p for p in work.iterdir()
                  if WEIGHT.fullmatch(p.name) and not p.is_symlink() and p.is_file())


def delete_run(root, record, meta, files, accounting, keep, include_incomplete):
    work = checked_work(root, record, meta)
    with work_lock(work):
        # Re-read metadata and queue while holding the same training lock.
        current = json.loads(record.read_text())
        if current != meta:
            raise ValueError('run metadata changed since preview')
        reason = blocked(current, queue(), accounting, keep, include_incomplete)
        if reason:
            raise ValueError(reason)
        if weights(work) != files:
            raise ValueError('checkpoint list changed since preview')
        evidence = dict(at=datetime.now(timezone.utc).isoformat(),
                        experiment=meta['experiment'], seed=meta.get('seed'),
                        git_sha=meta.get('git_sha'),
                        files=[dict(name=p.name, bytes=p.stat().st_size) for p in files])
        # Persist the deletion list alongside run.json; logs/config/metrics stay.
        audit = record.parent / ('checkpoint_cleanup_' +
                datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f') + '.json')
        audit.write_text(json.dumps(evidence, indent=2) + '\n')
        total = 0
        for path in files:
            if path.is_symlink() or path.resolve().parent != work:
                raise ValueError('checkpoint path changed: ' + str(path))
            total += path.stat().st_size
            path.unlink()
        # Only remove the pointer when it actually points to a deleted file.
        pointer = work / 'last_checkpoint'
        if pointer.is_file() and not pointer.is_symlink():
            target = Path(pointer.read_text().strip())
            if not target.is_absolute():
                target = work / target
            if target.resolve() in {p.resolve() for p in files}:
                pointer.unlink()
        return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs-root', type=Path, default=ROOT / 'work_dirs/slurm_runs')
    parser.add_argument('--delete', action='store_true', help='Delete the printed eligible weights, including best')
    parser.add_argument('--include-incomplete', action='store_true',
                        help='Also delete inactive failed/timed-out/unknown jobs; destroys resume capability')
    parser.add_argument('--keep-experiment', action='append', default=['centrepool_b_ade'],
                        help='Keep every run of this experiment (repeatable); CentrePool is always protected')
    args = parser.parse_args()
    root = args.runs_root.resolve(strict=True)
    if root.name != 'slurm_runs' or args.runs_root.is_symlink():
        parser.error('Expected a real slurm_runs directory')
    records = sorted(root.glob('*/*/run.json'))
    metas = [(record, json.loads(record.read_text())) for record in records]
    try:
        active = queue()  # Fail closed if Slurm is unreachable.
        ids = {str(a['job_id']) for _, meta in metas for a in meta.get('attempts', []) if a.get('job_id')}
        accounting = states(ids)
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        parser.error('Cannot verify Slurm state; nothing deleted: ' + str(exc))
    total, count = 0, 0
    for record, meta in metas:
        try:
            work = checked_work(root, record, meta)
            reason = blocked(meta, active, accounting, set(args.keep_experiment), args.include_incomplete)
            if reason:
                print('KEEP', record.parent.relative_to(root), '|', reason)
                continue
            files = weights(work)
            if not files:
                continue
            size = sum(p.stat().st_size for p in files)
            print('DELETE' if args.delete else 'PREVIEW', record.parent.relative_to(root),
                  '|', len(files), 'weights |', f'{size / 2**30:.2f} GiB', '|', ', '.join(p.name for p in files))
            if args.delete:
                size = delete_run(root, record, meta, files, accounting,
                                  set(args.keep_experiment), args.include_incomplete)
            total += size
            count += len(files)
        except (ValueError, OSError, subprocess.SubprocessError) as exc:
            print('KEEP', record.parent.relative_to(root), '|', exc)
    print(('Deleted' if args.delete else 'Would delete'), count, 'weights,', f'{total / 2**30:.2f} GiB')
    if not args.delete:
        print('Preview only. Repeat with --delete to apply. Logs, JSON, source and pretrained assets remain.')


if __name__ == '__main__':
    main()
