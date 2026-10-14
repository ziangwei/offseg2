"""Read best validation scores and last logged training step; no checkpoints."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import re
from submit import ROOT, select_jobs


def log_summary(logs):
    best_pattern = re.compile(r'The best checkpoint with ([\d.]+) mIoU at (\d+) iter')
    train_pattern = re.compile(r'Iter\(train\)\s*\[\s*(\d+)\s*/\s*\d+\s*\]')
    time_pattern = re.compile(r'\d{4}[-/]\d{2}[-/]\d{2} \d{2}:\d{2}:\d{2}')
    best = None
    last = None
    last_key = None
    for log in sorted(set(logs)):
        with log.open(errors='replace') as stream:
            fallback_time = datetime.fromtimestamp(log.stat().st_mtime)
            for number, line in enumerate(stream):
                match = best_pattern.search(line)
                if match:
                    value = float(match[1]), int(match[2])
                    if best is None or value[0] > best[0]:
                        best = value
                match = train_pattern.search(line)
                if match:
                    timestamp = time_pattern.search(line[:match.start()])
                    when = fallback_time
                    if timestamp:
                        try:
                            when = datetime.strptime(timestamp[0].replace('/', '-'),
                                                     '%Y-%m-%d %H:%M:%S')
                        except ValueError:
                            pass
                    key = (when, number)
                    if last_key is None or key > last_key:
                        last_key, last = key, int(match[1])
    return best, last


def best_score(logs):
    # Keep the matrix report's existing API unchanged.
    return log_summary(logs)[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('selection', nargs='?', default='routesearch')
    parser.add_argument('--runs-root', type=Path, default=ROOT / 'work_dirs/slurm_runs')
    args = parser.parse_args()
    header = f'{"Experiment":<25} {"Best mIoU":>10} {"Best iter":>12} {"Last iter":>12} {"Best file":>10} {"Seed":>12}'
    print(header)
    print('-' * len(header))
    for name in select_jobs(args.selection):
        records = sorted(args.runs_root.glob('*/' + name + '/run.json'))
        if not records:
            print(f'{name:<25} NO RUN')
            continue
        record = records[-1]
        meta = json.loads(record.read_text())
        work = Path(meta['work_dir'])
        logs = list(work.glob('*/*.log')) + list((record.parent / 'logs').glob('*.log'))
        best, last = log_summary(logs)
        score = f'{best[0]:.2f}' if best is not None else '--'
        iteration = str(best[1]) if best is not None else '--'
        present = '--'
        if best is not None:
            present = 'yes' if (work / f'best_mIoU_iter_{best[1]}.pth').is_file() else 'MISSING'
        progress = str(last) if last is not None else '--'
        print(f'{name:<25} {score:>10} {iteration:>12} {progress:>12} {present:>10} {str(meta.get("seed", "unknown")):>12}')


if __name__ == '__main__':
    main()
