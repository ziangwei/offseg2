"""Read-only ADE Route error audit on normal, stitched validation outputs.

Ground truth is consumed only after prediction. No reranking, TTA, new loss,
test-time memory updates, or training. Counts are reduced before ratios.
"""
import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np


def boundary_band(gt, valid, radius):
    import cv2
    edge = np.zeros(gt.shape, dtype=np.uint8)
    different = valid[1:] & valid[:-1] & (gt[1:] != gt[:-1])
    edge[1:] |= different
    edge[:-1] |= different
    different = valid[:, 1:] & valid[:, :-1] & (gt[:, 1:] != gt[:, :-1])
    edge[:, 1:] |= different
    edge[:, :-1] |= different
    # Both sides of a four-neighbour semantic edge, dilated in Chebyshev distance.
    return cv2.dilate(edge, np.ones((2 * radius + 1,) * 2, np.uint8)).astype(bool) & valid


def region_sizes(gt, valid):
    import cv2
    result = np.zeros(gt.shape, dtype=np.int64)
    for k in np.unique(gt[valid]):
        mask = np.ascontiguousarray(valid & (gt == k), dtype=np.uint8)
        _, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        areas = stats[:, cv2.CC_STAT_AREA].copy()
        areas[0] = 0
        result[mask != 0] = areas[labels[mask != 0]]
    return result


class ErrorAudit:
    def __init__(self, classes):
        self.classes = classes
        self.confusion = np.zeros((classes, classes), dtype=np.int64)
        self.groups = {}
        self.per_image = []
        self.class_hits = np.zeros((classes, 3), dtype=np.int64)

    def consume(self, gt, top, image_path, prediction=None):
        gt, top = np.asarray(gt, dtype=np.int64), np.asarray(top, dtype=np.int64)
        if gt.ndim != 2 or top.shape != (5, *gt.shape):
            raise ValueError('Require GT [H,W] and sorted class ids [5,H,W]')
        if top.min() < 0 or top.max() >= self.classes:
            raise ValueError('Prediction id outside class range')
        valid = (gt >= 0) & (gt < self.classes)
        pred = top[0] if prediction is None else np.asarray(prediction)
        if pred.shape != gt.shape or pred.min() < 0 or pred.max() >= self.classes:
            raise ValueError('Invalid evaluated prediction')
        wrong = valid & (pred != gt)
        confusion = np.bincount(self.classes * gt[valid] + pred[valid],
                                minlength=self.classes**2).reshape(self.classes, self.classes)
        self.confusion += confusion
        hits = [np.any(top[:k] == gt[None], axis=0) & wrong for k in (2, 3, 5)]
        for i, hit in enumerate(hits):
            self.class_hits[:, i] += np.bincount(gt[hit], minlength=self.classes)
        b3, b5 = boundary_band(gt, valid, 3), boundary_band(gt, valid, 5)
        sizes = region_sizes(gt, valid)
        area = max(int(valid.sum()), 1)
        small = valid & (sizes <= area * .001)
        medium = valid & (sizes > area * .001) & (sizes <= area * .01)
        large = valid & (sizes > area * .01)
        present = np.bincount(gt[valid], minlength=self.classes) > 0
        absent = valid & ~present[pred]
        groups = dict(all=valid, boundary_r3=b3, interior_r3=valid & ~b3,
                      boundary_r5=b5, interior_r5=valid & ~b5,
                      small_region=small, medium_region=medium, large_region=large,
                      absent_prediction=absent, present_prediction=valid & ~absent)
        for name, size_mask in (('small', small), ('medium', medium), ('large', large)):
            groups[name + '_boundary_r3'] = size_mask & b3
            groups[name + '_interior_r3'] = size_mask & ~b3
        for name, mask in groups.items():
            counts = np.array([mask.sum(), (mask & wrong).sum(),
                               *[(mask & hit).sum() for hit in hits]], dtype=np.int64)
            self.groups[name] = self.groups.get(name, np.zeros(5, dtype=np.int64)) + counts
        tp = confusion.diagonal()
        union = confusion.sum(0) + confusion.sum(1) - tp
        self.per_image.append(dict(image=image_path, valid_pixels=int(valid.sum()),
                                   wrong_pixels=int(wrong.sum()),
                                   image_miou=100 * float(np.mean(tp[union > 0] / union[union > 0])) if union.any() else None,
                                   wrong_gt_in_top3=int(hits[1].sum()),
                                   boundary_r3_wrong=int((wrong & b3).sum()),
                                   small_region_wrong=int((wrong & small).sum())))

    def shard(self):
        return dict(confusion=self.confusion, groups=self.groups,
                    class_hits=self.class_hits, per_image=self.per_image)


def merge(shards, names):
    confusion = sum(s['confusion'] for s in shards)
    hits = sum(s['class_hits'] for s in shards)
    images = [i for s in shards for i in s['per_image']]
    paths = [i['image'] for i in images]
    if len(set(paths)) != len(paths):
        raise ValueError('Duplicate evaluation images (sampler padding or repeated input)')
    groups = {}
    for name in sorted(set().union(*(s['groups'] for s in shards))):
        counts = sum((s['groups'].get(name, np.zeros(5, dtype=np.int64)) for s in shards))
        n, wrong, *top = map(int, counts)
        groups[name] = dict(valid_pixels=n, wrong_pixels=wrong,
                            error_rate=wrong / n if n else None,
                            wrong_gt_in_topk=dict(zip(('2', '3', '5'), top)),
                            wrong_gt_in_topk_fraction={str(k):v / wrong if wrong else None for k,v in zip((2,3,5),top)})
    rows = []
    for k, name in enumerate(names):
        tp = int(confusion[k, k])
        gt, predicted = int(confusion[k].sum()), int(confusion[:, k].sum())
        fn, fp = gt - tp, predicted - tp
        union = tp + fn + fp
        rows.append(dict(class_id=k, name=name, iou=100 * tp / union if union else None,
                         gt_pixels=gt, predicted_pixels=predicted, tp=tp, fn=fn, fp=fp,
                         recall=tp / gt if gt else None, precision=tp / predicted if predicted else None,
                         wrong_gt_in_top3=int(hits[k, 1]),
                         wrong_gt_in_top3_fraction=float(hits[k, 1] / fn) if fn else None))
    pairs = [(int(confusion[g,p]), g,p) for g in range(len(names)) for p in range(len(names)) if g != p and confusion[g,p]]
    pairs.sort(reverse=True)
    return dict(images=len(images), miou=float(np.mean([r['iou'] for r in rows if r['iou'] is not None])),
                groups=groups, classes=rows, confusion=confusion.tolist(),
                top_confusions=[dict(gt=names[g], predicted=names[p], pixels=n) for n,g,p in pairs[:50]],
                per_image=sorted(images, key=lambda x: x['image']))


def write_report(result, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'error_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    for name, rows in (('per_class', result['classes']), ('per_image', result['per_image']),
                       ('confusion_pairs', result['top_confusions'])):
        if rows:
            with (output / (name + '.csv')).open('w', encoding='utf-8-sig', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    lines = [f"# Proto-route B error audit: mIoU {result['miou']:.4f}",
             f"Images: {result['images']}; expected-score check: {result['expected_score_check']}",
             '', '| Region | Pixels | Error rate | GT in top3 among errors |', '|---|---:|---:|---:|']
    for name, group in result['groups'].items():
        rate, recall = group['error_rate'], group['wrong_gt_in_topk_fraction']['3']
        lines.append(f"| {name} | {group['valid_pixels']} | {rate if rate is not None else 'NA'} | {recall if recall is not None else 'NA'} |")
    lines += ['', result['definition'], '', 'These are diagnostic associations, not attainable gains or a causal attribution.']
    (output / 'error_summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    import torch
    from mmengine.config import Config
    from mmengine.hooks import Hook
    from mmengine.runner import Runner
    from mmengine.model import is_model_wrapper
    from mmengine.dist import all_gather_object, get_rank, get_world_size

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--work-dir', required=True)
    parser.add_argument('--launcher', default='none', choices=('none', 'pytorch'))
    parser.add_argument('--local-rank', '--local_rank', type=int, default=0)
    parser.add_argument('--expected-miou', type=float, default=48.49)
    args = parser.parse_args()
    os.environ.setdefault('LOCAL_RANK', str(args.local_rank))
    cfg = Config.fromfile(args.config)
    if cfg.model.decode_head.type != 'OffSegCCMIACSProtoRoute':
        raise ValueError('Use the original Route config, not a losing variant')
    if not Path(args.checkpoint).is_file():
        raise FileNotFoundError(args.checkpoint)
    cfg.load_from, cfg.resume = args.checkpoint, False
    cfg.work_dir, cfg.launcher = args.work_dir, args.launcher
    runner = Runner.from_cfg(cfg)

    class AuditHook(Hook):
        def before_test(self, runner):
            self.model = runner.model.module if is_model_wrapper(runner.model) else runner.model
            if len(runner.test_dataloader.dataset) % get_world_size():
                raise ValueError('Dataset must divide evenly across ranks; do not count padded duplicates')
            # Strictly validate model keys, rather than accepting missing memory silently.
            payload = torch.load(args.checkpoint, map_location='cpu')
            state = payload.get('state_dict', payload)
            state = {k.removeprefix('module.'):v for k,v in state.items()}
            self.model.load_state_dict(state, strict=True)
            del payload, state
            self.saved = {k:v.detach().cpu().clone() for k,v in self.model.named_buffers()}
            self.audit = ErrorAudit(self.model.decode_head.num_classes)

        def after_test_iter(self, runner, batch_idx, data_batch=None, outputs=None):
            for sample in outputs:
                scores = sample.seg_logits.data.detach()
                if self.model.training or not torch.isfinite(scores).all():
                    raise RuntimeError('Require finite eval-mode predictions')
                gt = sample.gt_sem_seg.data.squeeze(0).cpu().numpy()
                top = scores.topk(5, dim=0).indices.cpu().numpy()
                prediction = sample.pred_sem_seg.data.squeeze(0)
                if not torch.equal(scores.gather(0, prediction.to(scores.device).unsqueeze(0))[0], scores.max(0).values):
                    raise RuntimeError('Evaluated prediction does not maximize logits')
                # Preserve the evaluator argmax in the presence of tied logits.
                self.audit.consume(gt, top, str(sample.img_path), prediction.cpu().numpy())

        def after_test_epoch(self, runner, metrics=None):
            for k,v in self.model.named_buffers():
                if not torch.equal(v.cpu(), self.saved[k]):
                    raise RuntimeError('Evaluation modified buffer: ' + k)
            result = merge(all_gather_object(self.audit.shard()), runner.test_dataloader.dataset.metainfo['classes'])
            if abs(result['miou'] - float(metrics['mIoU'])) > .011:
                raise RuntimeError('Diagnostic confusion does not reproduce evaluator mIoU')
            result.update(config=args.config, checkpoint=str(Path(args.checkpoint).resolve()),
                          metrics=metrics, test_cfg=dict(cfg.model.test_cfg),
                          expected_miou=args.expected_miou,
                          expected_score_check='PASS' if abs(result['miou']-args.expected_miou) <= .02 else 'MISMATCH',
                          definition='Original stitched/resized evaluation logits; GT used only after prediction. '
                          'Edges use valid four-neighbour class changes, dilated radius 3/5 original-image pixels on both sides. '
                          'Small/medium/large are 8-connected same-label regions covering <=0.1%, (0.1%,1%], >1% of valid image area; '
                          'these are semantic regions, not object instances. Boundary and size groups overlap; cross-groups are provided. '
                          'Confusion rows=GT, columns=prediction. Overall mIoU averages class IoUs; pixel error rates are not mIoU contributions. '
                          'Top-k uses torch.topk tie ordering and reports GT inclusion among already-wrong pixels.')
            if get_rank() == 0:
                write_report(result, runner.work_dir)
            if result['expected_score_check'] != 'PASS':
                raise RuntimeError('Checkpoint did not reproduce 48.49 within 0.02; inspect saved report before interpreting')

    runner.register_hook(AuditHook())
    runner.test()


if __name__ == '__main__':
    main()
