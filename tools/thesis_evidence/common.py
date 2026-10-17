"""Portable formats shared by the server job and local figure scripts."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ROUTE_CONFIG = 'local_configs/offseg2/Base/offsegccmiacs_protoroute_r4_responsibility_ade20k_160k-512x512.py'
OFFSEG_CONFIG = 'local_configs/offseg/Base/offseg-b_ade20k_160k-512x512.py'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def image_id(path):
    # ADE validation basenames are unique; duplicate ids are rejected on merge.
    return Path(path).stem


def confusion(gt, pred, classes):
    import numpy as np
    if gt.shape != pred.shape or pred.min() < 0 or pred.max() >= classes:
        raise ValueError('Invalid prediction shape/class ids')
    valid = (gt >= 0) & (gt < classes)
    return np.bincount(classes * gt[valid].astype('int64') + pred[valid],
                       minlength=classes * classes).reshape(classes, classes)


def class_metrics(matrix, names):
    import numpy as np
    matrix = np.asarray(matrix)
    rows = []
    for k, name in enumerate(names):
        tp = int(matrix[k, k])
        total, predicted = int(matrix[k].sum()), int(matrix[:, k].sum())
        union = total + predicted - tp
        rows.append(dict(class_id=k, name=name, gt_pixels=total, predicted_pixels=predicted,
                         tp=tp, fp=predicted-tp, fn=total-tp,
                         iou=100 * tp / union if union else None))
    miou = float(np.mean([r['iou'] for r in rows if r['iou'] is not None]))
    return rows, miou


def crop_boxes(height, width, crop, stride):
    hc, wc = crop
    hs, ws = stride
    boxes = []
    for iy in range(max(height-hc+hs-1, 0)//hs+1):
        for ix in range(max(width-wc+ws-1, 0)//ws+1):
            y2, x2 = min(iy*hs+hc, height), min(ix*ws+wc, width)
            boxes.append((max(y2-hc, 0), max(x2-wc, 0), y2, x2))
    return boxes
