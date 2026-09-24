"""Read best validation scores from logs; never load model checkpoints."""
import argparse
import json
from pathlib import Path
import re
from submit import ROOT, select_jobs


def best_score(logs):
    pattern = re.compile(r'The best checkpoint with ([\d.]+) mIoU at (\d+) iter')
    best = None
    for log in sorted(set(logs)):
        with log.open(errors='replace') as stream:
            for line in stream:
                match = pattern.search(line)
                if match:
                    value = float(match[1]), int(match[2])
                    if best is None or value[0] > best[0]:
                        best = value
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('selection', nargs='?', default='round3')
    parser.add_argument('--runs-root', type=Path, default=ROOT / 'work_dirs/slurm_runs')
    args = parser.parse_args()
    print(f'{"Experiment":<25} {"Best mIoU":>10} {"Iteration":>12} {"Best file":>10}')
    print('-' * 62)
    for name in select_jobs(args.selection):
        records = sorted(args.runs_root.glob('*/' + name + '/run.json'))
        if not records:
            print(f'{name:<25} NO RUN')
            continue
        record = records[-1]
        meta = json.loads(record.read_text())
        work = Path(meta['work_dir'])
        logs = list(work.glob('*/*.log')) + list((record.parent / 'logs').glob('*.log'))
        best = best_score(logs)
        if best is None:
            print(f'{name:<25} NO BEST RECORD: {work}')
            continue
        exists = (work / f'best_mIoU_iter_{best[1]}.pth').is_file()
        print(f'{name:<25} {best[0]:>10.2f} {best[1]:>12} {"yes" if exists else "MISSING":>10}')


if __name__ == '__main__':
    main()
