"""Check deletion boundaries with disposable fixtures and mocked Slurm/lock."""
from contextlib import contextmanager, redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import cleanup_checkpoints as cleanup


@contextmanager
def unlocked(work):
    yield


def main():
    parent = cleanup.ROOT / 'tmp'
    parent.mkdir(exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix='cleanup_sanity_', dir=parent)).resolve()
    assert parent.resolve() in temp.parents
    root = temp / 'slurm_runs'
    root.mkdir()
    def make(name, job, status='submitted', external=False):
        run = root / 'batch' / name
        work = temp / 'external' if external else run / 'checkpoints'
        work.mkdir(parents=True, exist_ok=True)
        (run / 'source/pretrained').mkdir(parents=True)
        (run / 'source/pretrained/iter_9.pth').write_bytes(b'pretrained')
        (run / 'logs').mkdir()
        (run / 'logs/slurm.log').write_text('The best checkpoint with 47.50 mIoU at 8000 iter')
        (work / 'iter_8000.pth').write_bytes(b'rolling')
        (work / 'best_mIoU_iter_8000.pth').write_bytes(b'best')
        (work / 'last_checkpoint').write_text(str(work / 'iter_8000.pth'))
        (work / 'metrics.json').write_text('{}')
        meta = dict(run_dir=str(run), work_dir=str(work), experiment=name,
                    attempts=[dict(status=status, job_id=job)])
        record = run / 'run.json'
        record.write_text(json.dumps(meta))
        return record, meta
    records = [make('done_b_ade', '101'), make('running_b_ade', '102'),
               make('failed_b_ade', '103'), make('unknown_b_ade', '104'),
               make('centrepool_b_ade', '105'), make('external_b_ade', '106', external=True),
               make('ambiguous_b_ade', '107', status='submission_error')]
    active = ({'102'}, {'os2_running_b_ade'})
    accounting = dict(zip(('101', '102', '103', '105', '106'),
                          ('COMPLETED', 'RUNNING', 'TIMEOUT', 'COMPLETED', 'COMPLETED')))
    keep = {'centrepool_b_ade'}
    expected = (False, True, True, True, True, False, True)
    for (_, meta), blocked in zip(records, expected):
        assert bool(cleanup.blocked(meta, active, accounting, keep, False)) == blocked
    assert cleanup.blocked(records[2][1], active, accounting, keep, True) is None
    assert cleanup.blocked(records[4][1], active, accounting, keep, True)
    interrupted_then_eval = dict(records[2][1], attempts=[
        dict(mode='train', status='submitted', job_id='103'),
        dict(mode='eval', status='submitted', job_id='101')])
    assert cleanup.blocked(interrupted_then_eval, active, accounting, keep, False)
    try:
        cleanup.checked_work(root, *records[5])
        raise AssertionError('External directory accepted')
    except ValueError:
        pass
    def slurm(args):
        if args[0] == 'squeue':
            return '102|os2_running_b_ade\n'
        if args[0] == 'sacct':
            return '\n'.join(f'{job}|{state}' for job, state in accounting.items())
        raise AssertionError(args)
    stream = io.StringIO()
    with patch.object(sys, 'argv', ['cleanup', '--runs-root', str(root)]), \
            patch.object(cleanup, 'command', side_effect=slurm), redirect_stdout(stream):
        cleanup.main()
    assert 'Would delete 2 weights' in stream.getvalue()
    assert all(Path(m['work_dir'], 'iter_8000.pth').is_file() for _, m in records)
    # Active race and changed metadata must block deletion, including best.
    record, meta = records[0]
    work = cleanup.checked_work(root, record, meta)
    files = cleanup.weights(work)
    with patch.object(cleanup, 'work_lock', unlocked), \
            patch.object(cleanup, 'queue', return_value=({'101'}, set())):
        try:
            cleanup.delete_run(root, record, meta, files, accounting, keep, False)
            raise AssertionError('Newly running job deleted')
        except ValueError:
            pass
    changed = dict(meta, seed=7)
    record.write_text(json.dumps(changed))
    with patch.object(cleanup, 'work_lock', unlocked):
        try:
            cleanup.delete_run(root, record, meta, files, accounting, keep, False)
            raise AssertionError('Changed metadata accepted')
        except ValueError:
            pass
    record.write_text(json.dumps(meta))
    stream = io.StringIO()
    with patch.object(sys, 'argv', ['cleanup', '--runs-root', str(root), '--delete']), \
            patch.object(cleanup, 'command', side_effect=slurm), \
            patch.object(cleanup, 'work_lock', unlocked), redirect_stdout(stream):
        cleanup.main()
    assert 'Deleted 2 weights' in stream.getvalue()
    assert not (work / 'iter_8000.pth').exists()
    assert not (work / 'best_mIoU_iter_8000.pth').exists()
    assert not (work / 'last_checkpoint').exists()
    assert list(record.parent.glob('checkpoint_cleanup_*.json'))
    assert (work / 'metrics.json').is_file()
    for rec, m in records:
        assert rec.is_file()
        assert (rec.parent / 'source/pretrained/iter_9.pth').read_bytes() == b'pretrained'
        assert (rec.parent / 'logs/slurm.log').is_file()
        if rec != record:
            assert (Path(m['work_dir']) / 'iter_8000.pth').is_file()
    with patch.object(cleanup, 'command', side_effect=OSError('offline')), \
            patch.object(sys, 'argv', ['cleanup', '--runs-root', str(root), '--delete']), \
            redirect_stdout(io.StringIO()), patch('sys.stderr', io.StringIO()):
        try:
            cleanup.main()
            raise AssertionError('Unavailable Slurm accepted')
        except SystemExit as exc:
            assert exc.code == 2
    print('PASS preview/delete, queue race, changed metadata, unfinished/CentrePool protection, external path rejection, logs/JSON/source/pretrained retained, Slurm fail-closed')
    print('Training-lock calls are mocked on Windows; Linux uses the existing flock file.')


if __name__ == '__main__':
    main()
