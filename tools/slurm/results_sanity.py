"""Check logged progress across resumes, validation and fresh restarts."""
from contextlib import redirect_stdout
from datetime import datetime
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import results


def main():
    parent = results.ROOT / 'tmp'
    parent.mkdir(exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix='progress_sanity_', dir=parent)).resolve()
    assert parent.resolve() in root.parents
    original = root / 'original.log'
    original.write_text(
        '2026/10/11 12:00:00 - INFO - The best checkpoint with 47.84 mIoU at 144000 iter\n'
        '2026/10/11 12:01:00 - INFO - Iter(train) [159950/160000] loss: 1\n')
    resumed = root / 'resumed.log'
    resumed.write_text(
        '2026/10/14 10:00:00 - INFO - Iter(train) [ 152050/160000] loss: 1\n'
        '2026/10/14 10:10:00 - INFO - Iter(train) [ 154000/160000] loss: 1\n'
        '2026/10/14 10:20:00 - INFO - Iter(val) [2000/2000] mIoU: 47.5\n')
    # Touching old files must not make a historical step look like the latest.
    os.utime(original, (datetime(2026, 10, 15).timestamp(),) * 2)
    assert results.log_summary([resumed, original, resumed]) == ((47.84, 144000), 154000)
    assert results.best_score([original, resumed]) == (47.84, 144000)
    with resumed.open('a') as stream:
        stream.write('2026-10-14 10:30:00 - INFO - Iter(train) [160000/160000]\n')
    assert results.log_summary([original, resumed])[1] == 160000
    empty = root / 'empty.log'
    empty.write_text('Iter(val) [160000/160000]\n')
    assert results.log_summary([empty]) == (None, None)
    undated = root / 'undated.log'
    undated.write_text('Iter(train) [50/160000]\nIter(train) [100/160000]\n')
    assert results.log_summary([undated]) == (None, 100)
    assert results.log_summary([]) == (None, None)

    def run(batch, name, content, seed):
        folder = root / batch / name
        work = folder / 'checkpoints'
        logs = folder / 'logs'
        logs.mkdir(parents=True)
        work.mkdir()
        (folder / 'run.json').write_text(json.dumps(dict(work_dir=str(work), seed=seed)))
        (logs / 'slurm-1.log').write_text(content)
        return folder, work
    name = 'route_draw1_t_ade'
    run('20261011', name, original.read_text(), 111)
    _, work = run('20261014', name,
                  '2026/10/14 12:00:00 - INFO - Iter(train) [6000/160000]\n', 222)
    second = 'route_noccm_b_ade'
    _, second_work = run('20261014', second, original.read_text(), 333)
    log_dir = second_work / '20261014'
    log_dir.mkdir()
    (log_dir / 'train.log').write_text(resumed.read_text())
    (second_work / 'best_mIoU_iter_144000.pth').touch()
    def report():
        out = io.StringIO()
        with patch.object(sys, 'argv', ['results.py', 'routesearch', '--runs-root', str(root)]), \
                patch.object(results, 'select_jobs', return_value=[name, second, 'absent']), redirect_stdout(out):
            results.main()
        return out.getvalue()
    output = report()
    assert 'Best iter' in output and 'Last iter' in output
    first_row = next(line for line in output.splitlines() if line.startswith(name)).split()
    assert first_row == [name, '--', '--', '6000', '--', '222']
    second_row = next(line for line in output.splitlines() if line.startswith(second)).split()
    assert second_row == [second, '47.84', '144000', '160000', 'yes', '333']
    assert 'absent                    NO RUN' in output
    (second_work / 'best_mIoU_iter_144000.pth').unlink()
    assert 'MISSING' in report() and '160000' in report()
    print('PASS timestamp-ordered resume progress, rollback, train/val separation, 160k, missing logs, early no-best rows, latest restarted run, deleted weights and matrix best_score compatibility')
    print(output)


if __name__ == '__main__':
    main()
