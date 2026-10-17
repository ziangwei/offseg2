"""Scan bounded log/config locations, preserving individual runs and sources."""
import argparse
import re
import shutil
from pathlib import Path
from .common import ROOT, read_json, write_json, write_csv

TRAIN = re.compile(r'Iter\(train\)\s*\[\s*(\d+)\s*/\s*\d+\s*\]')
VAL = re.compile(r'Iter\(val\).*?mIoU:\s*([\d.]+)')
BEST = re.compile(r'The best checkpoint with ([\d.]+) mIoU at (\d+) iter')
NEEDLE = re.compile(r'\b(acc_(?:proto|iacs|acs|ccm)[\w]*):\s*([-+\deE.]+)')


def scan_log(path):
    current, curves, best, last, needles = None, [], None, None, {}
    with path.open(errors='replace') as stream:
        for line_number, line in enumerate(stream, 1):
            train = TRAIN.search(line)
            if train:
                current = last = int(train[1])
                needles.update({k: float(v) for k, v in NEEDLE.findall(line)})
            val = VAL.search(line)
            if val:
                curves.append(dict(iteration=current, miou=float(val[1]),
                                   source=str(path), line=line_number,
                                   step_source='preceding_train_line' if current else 'unknown'))
            hit = BEST.search(line)
            if hit and (best is None or float(hit[1]) > best['miou']):
                best = dict(miou=float(hit[1]), iteration=int(hit[2]), source=str(path), line=line_number)
    return dict(curves=curves, best=best, last_iter=last, final_logged_needles=needles)


def collect(repo, output):
    repo, output = Path(repo).resolve(), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name in ('AGENTS.md', 'THESIS_ROUTE.md', 'EXPERIMENTS.md', 'tools/slurm/experiments.json'):
        dest = output/'canonical'/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, dest)
    for source in (ROOT/'docs/thesis_handoff').glob('*.md'):
        shutil.copy2(source, output/source.name)
    records, inventory = [], []
    work = repo/'work_dirs'
    for record in sorted((work/'slurm_runs').glob('*/*/run.json')):
        meta = read_json(record)
        run = record.parent
        target = Path(meta['work_dir'])
        logs = sorted(set(target.glob('*/*.log')) | set((run/'logs').glob('*.log')))
        records.append((str(run.relative_to(work)), meta, logs, run/'source'/meta['config']))
    # Original, pre-Slurm runs. Never recurse into data/source/weights trees.
    if work.exists():
        for run in sorted(work.iterdir()):
            if not run.is_dir() or run.name.startswith(('slurm_', 'thesis_')):
                continue
            logs = sorted(set(run.glob('*.log')) | set(run.glob('*/*.log')))
            if not logs:
                continue
            configs = list(run.glob('*.py'))
            records.append((run.name, dict(experiment=run.name, seed='unknown',
                             source_label='legacy_directory; seed/config require inspection'),
                            logs, configs[0] if len(configs) == 1 else None))
    for run_id, meta, logs, config in records:
        folder = output/'runs'/run_id
        folder.mkdir(parents=True, exist_ok=True)
        write_json(folder/'run.json', meta)
        curves, scans = [], []
        for log in logs:
            item = scan_log(log)
            curves.extend(item['curves'])
            scans.append(dict(log=str(log), **{k:v for k,v in item.items() if k != 'curves'}))
        write_csv(folder/'validation_curve.csv', curves)
        write_json(folder/'log_sources.json', scans)
        if config and config.is_file():
            shutil.copy2(config, folder/'submitted_config.py')
            try:
                from mmengine.config import Config
                resolved = Config.fromfile(str(config), import_custom_modules=False)
                resolved.dump(str(folder/'resolved_config.py'))
                config_status = 'resolved; run.json seed/overrides take precedence'
            except Exception as exc:
                config_status = f'unresolved: {type(exc).__name__}: {str(exc)[:160]}'
        else:
            config_status = 'missing or ambiguous; do not infer from name'
        bests = [s['best'] for s in scans if s['best']]
        best = max(bests, key=lambda b:b['miou']) if bests else {}
        inventory.append(dict(run_id=run_id, experiment=meta.get('experiment', run_id),
                              seed=meta.get('seed', 'unknown'), git_sha=meta.get('git_sha', 'unknown'),
                              best_miou=best.get('miou'), best_iter=best.get('iteration'),
                              log_files=len(logs), config_status=config_status))
    write_csv(output/'run_inventory.csv', inventory)
    write_json(output/'collection.json', dict(repo=str(repo), runs=len(inventory),
        note='Every attempt stays separate. No maxima are stitched into a synthetic learning curve. '
             'A best checkpoint marker is not a final score; unknown step/seed remains unknown. '
             'Original large logs remain on the server; CSV rows retain file and line provenance.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    collect(args.repo, args.output)
