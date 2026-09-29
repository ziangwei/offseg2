"""3 datasets x 3 scales: distinguish historical reports from new log bests."""
import argparse
import json
from pathlib import Path

from results import best_score
from submit import ROOT


# Only the four existing ORIGINAL Route results, never PARSeg/Proto/variants.
# These are historical reports; no local checkpoint verification is implied.
HISTORY = {
    ('ade', 'b'): (48.49, 'best=last @160k; audited 48.4924'),
    ('stuff', 't'): (42.32, 'best=last @80k; screenshot'),
    ('stuff', 'b'): (44.75, 'last @80k; owner report, best unverified'),
    ('city', 'b'): (80.69, 'best; owner report, iteration unverified'),
}


def cell(dataset, scale, runs_root):
    if (dataset, scale) in HISTORY:
        score, source = HISTORY[dataset, scale]
        return f'{score:.2f}*', source
    name = f'route_{scale}_{dataset}'
    records = sorted(runs_root.glob(f'*/{name}/run.json'))
    if not records:
        return '--', name + ': no submitted run found'
    record = records[-1]
    meta = json.loads(record.read_text())
    work = Path(meta['work_dir'])
    logs = list(work.glob('*/*.log')) + list((record.parent / 'logs').glob('*.log'))
    best = best_score(logs)
    if best is None:
        return '--', name + ': no best record yet; ' + str(record.parent)
    score, iteration = best
    exists = (work / f'best_mIoU_iter_{iteration}.pth').is_file()
    return f'{score:.2f}', (f'{name}: best so far @{iteration}, '
                           f'best file={"yes" if exists else "MISSING"}; {record.parent}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs-root', type=Path, default=ROOT / 'work_dirs/slurm_runs')
    args = parser.parse_args()
    print(f'{"Dataset":<16} {"T (S1)":>12} {"B (S2)":>12} {"L":>12}')
    print('-' * 56)
    notes = []
    for dataset, label in [('ade', 'ADE20K'), ('stuff', 'COCO-Stuff164K'), ('city', 'Cityscapes')]:
        values = []
        for scale in ('t', 'b', 'l'):
            value, note = cell(dataset, scale, args.runs_root)
            values.append(value)
            notes.append(f'{label}/{scale.upper()}: {note}')
        print(f'{label:<16}' + ''.join(f' {value:>12}' for value in values))
    print('\n* Historical result with source below; not re-read from a new training.')
    print('New entries are best SO FAR; a score alone does not prove training finished.')
    for note in notes:
        print(note)


if __name__ == '__main__':
    main()
