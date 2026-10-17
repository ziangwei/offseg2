"""Paired statistics and reproducible case selection from evaluated PNGs."""
import argparse
from pathlib import Path
import shutil
import sys
import numpy as np
from PIL import Image
from .common import read_json, write_json, write_csv, confusion, class_metrics


def region_counts(gt, base, route, classes):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from route_error_audit_b import boundary_band
    valid = gt < classes
    corrected = valid & (base != gt) & (route == gt)
    damaged = valid & (base == gt) & (route != gt)
    masks = dict(all=valid)
    for radius in (3, 5):
        boundary = boundary_band(gt, valid, radius)
        masks[f'boundary_r{radius}'] = boundary
        masks[f'interior_r{radius}'] = valid & ~boundary
    present = np.bincount(gt[valid], minlength=classes) > 0
    result = {}
    for name, mask in masks.items():
        result[name] = dict(pixels=int(mask.sum()), base_errors=int((mask & (base != gt)).sum()),
            route_errors=int((mask & (route != gt)).sum()), corrected=int((mask & corrected).sum()),
            damaged=int((mask & damaged).sum()),
            base_absent_fp=int((mask & ~present[base]).sum()),
            route_absent_fp=int((mask & ~present[route]).sum()))
    return result


def choose_cases(rows, count=24):
    """Largest gains, largest regressions, then evenly spaced gain ranks; explicit selection bias."""
    ordered = sorted(rows, key=lambda r:(r['pixel_accuracy_gain'], r['id']))
    count = min(count, len(rows))
    chosen = []
    for row in (ordered[-4:][::-1] + ordered[:4] +
                [ordered[i] for i in np.linspace(0, len(ordered)-1, count, dtype=int)] + ordered):
        if row['id'] not in {r['id'] for r in chosen}:
            chosen.append(row)
        if len(chosen) == count:
            break
    return chosen


def analyse(run):
    run = Path(run)
    base = read_json(run/'offseg/evaluation.json')
    route = read_json(run/'route/evaluation.json')
    a, b = {r['id']:r for r in base['images']}, {r['id']:r for r in route['images']}
    if set(a) != set(b) or base['test_cfg'] != route['test_cfg']:
        raise ValueError('Require identical validation images and slide protocol')
    if route['expected_check'] == 'MISMATCH':
        raise ValueError('Route reproduction failed; stop before selecting thesis evidence')
    names, palette = route['classes'], route['palette']
    classes = len(names)
    totals, rows = {}, []
    matrices = [np.zeros((classes, classes), dtype='int64') for _ in range(2)]
    for key in sorted(a):
        gt = np.array(Image.open(run/'offseg/gt'/f'{key}.png'))
        if not np.array_equal(gt, np.array(Image.open(run/'route/gt'/f'{key}.png'))):
            raise ValueError('Ground truth mismatch: '+key)
        predictions = [np.array(Image.open(run/label/'predictions'/f'{key}.png')) for label in ('offseg', 'route')]
        for matrix, pred in zip(matrices, predictions):
            matrix += confusion(gt, pred, classes)
        counts = region_counts(gt, *predictions, classes)
        for name, values in counts.items():
            totals.setdefault(name, {k:0 for k in values})
            for k, value in values.items():
                totals[name][k] += value
        c = counts['all']
        labels, area = np.unique(gt[gt < classes], return_counts=True)
        ids = [int(k) for k in labels[np.argsort(area)[-3:][::-1]]]
        rows.append(dict(**a[key], class_ids=ids,
                         pixel_accuracy_gain=100*(c['corrected']-c['damaged'])/max(c['pixels'], 1), **c))
    per_class = []
    scores = []
    for meta, matrix in zip((base, route), matrices):
        cls, score = class_metrics(matrix, names)
        if abs(score-meta['miou']) > 1e-8:
            raise ValueError('Saved PNGs do not reproduce exported confusion')
        per_class.append(cls)
        scores.append(score)
    differences = [dict(class_id=k, name=names[k], base_iou=per_class[0][k]['iou'],
                        route_iou=per_class[1][k]['iou'],
                        delta=per_class[1][k]['iou']-per_class[0][k]['iou']
                        if all(p[k]['iou'] is not None for p in per_class) else None,
                        gt_pixels=per_class[0][k]['gt_pixels']) for k in range(classes)]
    # Select directions solely by baseline error counts, never by improvement.
    baseline_pairs = sorted(((int(matrices[0][g,p]),g,p) for g in range(classes)
                             for p in range(classes) if g != p), reverse=True)[:10]
    pairs = [dict(gt=names[g], predicted=names[p], base_pixels=n,
                  route_pixels=int(matrices[1][g,p]), gt_pixels=int(matrices[0][g].sum()),
                  base_fraction=n/max(int(matrices[0][g].sum()),1),
                  route_fraction=int(matrices[1][g,p])/max(int(matrices[0][g].sum()),1))
             for n,g,p in baseline_pairs]
    for name, values in totals.items():
        n = max(values['pixels'], 1)
        values.update(base_error_rate=values['base_errors']/n,
                      route_error_rate=values['route_errors']/n,
                      net_corrected_fraction=(values['corrected']-values['damaged'])/n)
    selected = choose_cases(rows)
    output = run/'comparison'
    output.mkdir(exist_ok=True)
    write_csv(output/'per_class_delta.csv', differences)
    write_csv(output/'confusion_directions.csv', pairs)
    write_csv(output/'regions.csv', [dict(region=k, **v) for k,v in totals.items()])
    write_csv(output/'per_image.csv', [{k:v for k,v in r.items() if k not in ('class_ids','shape')} for r in rows])
    write_json(output/'cases.json', selected)
    write_json(output/'comparison.json', dict(base_miou=scores[0], route_miou=scores[1],
        delta=scores[1]-scores[0], image_count=len(rows), scope=route['scope'], regions=totals,
        selection='24 cases: four largest pixel-accuracy gains, four largest regressions, '
                  'then evenly spaced ranks; only selected cases illustrate mechanisms. '
                  'Global quantitative results include every evaluated image.',
        baseline_source=read_json(run/'run.json')['baseline_source'],
        definitions='Boundary uses both sides of valid four-neighbour GT edges, dilated '
                    'Chebyshev radius 3/5 ORIGINAL-image pixels. Pixel error rates are not mIoU. '
                    'Off-diagonal confusion rows=GT, columns=prediction; directional rates divide GT-class pixels.'))
    assets = run/'selected_images'
    assets.mkdir(exist_ok=True)
    for row in selected:
        Image.open(row['image']).convert('RGB').save(assets/(row['id']+'.png'))
    (output/'palette.json').write_text(__import__('json').dumps(palette))
    print(f"Evaluated pair: {scores[0]:.4f} -> {scores[1]:.4f}; selected {len(selected)} cases")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    analyse(parser.parse_args().run)
