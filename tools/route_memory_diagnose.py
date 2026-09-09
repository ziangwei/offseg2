"""Read-only evaluation of an existing ProtoRoute checkpoint.

Normal predictions and mIoU are unchanged. Summaries are image/class-pair
means (equal window means within each image), split by full-image
ground-truth class presence (labels are used only AFTER
inference). No checkpoint editing, adaptation, or counterfactual scoring.
"""
import argparse
import json
import os
from pathlib import Path

import torch


class RouteProbe:
    def __init__(self, head):
        self.head = head
        self.pending = None
        self.batches = []
        self.head_calls = 0
        self.groups = {name: {'count': 0, 'sums': {}}
                       for name in ('all', 'present', 'absent')}
        self.images = self.valid_pixels = self.errors = self.absent_fp = 0
        self.saved = None

    def install(self):
        self.saved = {k: getattr(self.head, k).detach().clone()
                      for k in ('prototypes', 'proto_seen', 'proto_steps')}
        self.original_blend = self.head._blend_prototypes
        self.original_correction = self.head._subspace_correction

        def blend(masks, centres):
            if self.head.training or self.pending is not None:
                raise RuntimeError('Require eval mode and completed preceding head call')
            result = self.original_blend(masks, centres)
            with torch.no_grad():
                e = centres.detach().float()
                p = self.head.prototypes.detach().float().unsqueeze(0).expand_as(e)
                en, pn = e.norm(dim=-1), p.norm(dim=-1)
                support = masks.detach().float().softmax(dim=1).sum(dim=-1)
                n0 = torch.nn.functional.softplus(self.head.proto_n0_raw.detach().float())
                lam = n0 / (support + n0)
                if self.head.proto_fixed_lambda > 0:
                    lam = torch.full_like(lam, self.head.proto_fixed_lambda)
                lam *= (self.head.proto_seen > 0).view(1, -1)
                self.pending = {
                    'image_centre_norm': en, 'prototype_norm': pn,
                    'prototype_over_image_norm': pn / en.clamp_min(1e-8),
                    'prototype_image_cosine': (e * p).sum(-1) / (en * pn).clamp_min(1e-8),
                    'prototype_seen': (self.head.proto_seen > 0).float().view(1, -1).expand_as(en),
                    'support': support, 'lambda': lam,
                    'blended_centre_norm': result[0].detach().float().norm(dim=-1),
                }
            return result

        def correction(feat, centres, logits, **kwargs):
            result = self.original_correction(feat, centres, logits, **kwargs)
            with torch.no_grad():
                f, e = feat.detach().float(), centres.detach().float()
                raw = f @ e.transpose(1, 2)
                delta = result[0].detach().float()
                # Remove per-pixel class-common offsets: those do not change
                # class ordering and are removed by mask_norm's centering.
                raw = raw - raw.mean(dim=-1, keepdim=True)
                delta = delta - delta.mean(dim=-1, keepdim=True)
                raw_abs = raw.abs().mean(dim=1)
                delta_abs = delta.abs().mean(dim=1)
                self.pending.update(
                    feature_norm=f.norm(dim=-1).mean(dim=1, keepdim=True).expand_as(raw_abs),
                    raw_class_centred_abs=raw_abs,
                    correction_class_centred_abs=delta_abs,
                    correction_over_raw=delta_abs / raw_abs.clamp_min(1e-8))
                self.batches.append({k: v.cpu().double() for k, v in self.pending.items()})
                self.pending = None
                self.head_calls += 1
            return result

        self.head._blend_prototypes = blend
        self.head._subspace_correction = correction

    def consume(self, outputs):
        if self.pending is not None or not self.batches:
            raise RuntimeError('Expected completed head calls for these predictions')
        if any(len(outputs) != b['lambda'].shape[0] for b in self.batches):
            raise RuntimeError('Batch size changed within image inference; TTA is unsupported')
        # Preserve the original slide inference. Average diagnostics over its
        # windows, then weight images equally. These are NOT stitched pixel
        # maps; class presence below refers to the full image, not each crop.
        window_mean = {k: torch.stack([b[k] for b in self.batches]).mean(0)
                       for k in self.batches[0]}
        num_classes = self.head.num_classes
        for i, sample in enumerate(outputs):
            gt = sample.gt_sem_seg.data.squeeze().long()
            pred = sample.pred_sem_seg.data.squeeze().long().to(gt.device)
            if pred.shape != gt.shape:
                raise RuntimeError('Prediction/ground truth spatial mismatch')
            valid = (gt >= 0) & (gt < num_classes)
            present = torch.zeros(num_classes, dtype=torch.bool, device=gt.device)
            present[gt[valid].unique()] = True
            values = {k: v[i] for k, v in window_mean.items()}
            present = present.cpu()
            for name, mask in (('all', torch.ones_like(present)), ('present', present), ('absent', ~present)):
                group = self.groups[name]
                group['count'] += int(mask.sum())
                for key, value in values.items():
                    if not torch.isfinite(value).all():
                        raise RuntimeError('Non-finite diagnostic: ' + key)
                    group['sums'][key] = group['sums'].get(key, 0.0) + float(value[mask].sum())
            valid_pred = pred[valid]
            self.valid_pixels += int(valid.sum())
            self.errors += int((valid_pred != gt[valid]).sum())
            self.absent_fp += int((~present.to(pred.device)[valid_pred]).sum())
            self.images += 1
        self.pending = None
        self.batches = []

    def finish(self):
        if self.pending is not None or self.batches:
            raise RuntimeError('Unconsumed head diagnostics')
        for key, value in self.saved.items():
            if not torch.equal(value, getattr(self.head, key)):
                raise RuntimeError('Evaluation changed memory buffer: ' + key)
        self.head._blend_prototypes = self.original_blend
        self.head._subspace_correction = self.original_correction
        return dict(groups=self.groups, images=self.images, valid_pixels=self.valid_pixels,
                    errors=self.errors, absent_fp=self.absent_fp, head_calls=self.head_calls)


def summarise(shards):
    totals = {k: sum(s[k] for s in shards) for k in ('images', 'valid_pixels', 'errors', 'absent_fp', 'head_calls')}
    groups = {}
    for name in ('all', 'present', 'absent'):
        count = sum(s['groups'][name]['count'] for s in shards)
        keys = set().union(*(s['groups'][name]['sums'] for s in shards))
        groups[name] = dict(image_class_pairs=count, means={
            k: sum(s['groups'][name]['sums'].get(k, 0.0) for s in shards) / count if count else None
            for k in sorted(keys)})
    return dict(**totals, groups=groups,
                absent_fp_fraction_valid=totals['absent_fp'] / max(totals['valid_pixels'], 1),
                absent_fp_fraction_errors=totals['absent_fp'] / max(totals['errors'], 1))


def main():
    from mmengine.config import Config
    from mmengine.hooks import Hook
    from mmengine.runner import Runner
    from mmengine.model import is_model_wrapper
    from mmengine.dist import all_gather_object, get_rank, get_world_size

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--work-dir', required=True)
    parser.add_argument('--launcher', default='none', choices=('none', 'pytorch', 'slurm'))
    parser.add_argument('--local-rank', '--local_rank', type=int, default=0)
    args = parser.parse_args()
    os.environ.setdefault('LOCAL_RANK', str(args.local_rank))
    if not Path(args.checkpoint).is_file():
        raise FileNotFoundError(args.checkpoint)
    cfg = Config.fromfile(args.config)
    if cfg.model.decode_head.type != 'OffSegCCMIACSProtoRoute':
        raise ValueError('Only original ProtoRoute and its n0/EMA config variants are supported')
    if cfg.model.test_cfg.get('mode', 'whole') not in ('whole', 'slide'):
        raise ValueError('Require standard whole/slide evaluation; TTA is unsupported')
    cfg.load_from, cfg.resume = args.checkpoint, False
    cfg.work_dir, cfg.launcher = args.work_dir, args.launcher
    runner = Runner.from_cfg(cfg)

    class DiagnosticHook(Hook):
        def before_test(self, runner):
            model = runner.model.module if is_model_wrapper(runner.model) else runner.model
            # Standard ADE 2000-image validation divides evenly over 4 ranks.
            # Fail rather than count sampler-padded duplicates as real images.
            if len(runner.test_dataloader.dataset) % get_world_size():
                raise ValueError('Dataset size must divide evenly across ranks')
            self.probe = RouteProbe(model.decode_head)
            self.probe.install()

        def after_test_iter(self, runner, batch_idx, data_batch=None, outputs=None):
            self.probe.consume(outputs)

        def after_test_epoch(self, runner, metrics=None):
            result = summarise(all_gather_object(self.probe.finish()))
            result.update(config=args.config, checkpoint=args.checkpoint, metrics=metrics,
                          test_cfg=dict(cfg.model.test_cfg),
                          definition='Image/class-pair means, equal window means within each image. '
                                     'GT presence is full-image presence, used after prediction only. '
                                     'Feature statistics include padding and overlapping windows; they are not stitched maps. '
                                     'Cosine is zero for zero-norm vectors.')
            if get_rank() == 0:
                path = Path(runner.work_dir) / 'route_diagnostics.json'
                path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
                runner.logger.info('Read-only route diagnostics saved to %s', path)

    runner.register_hook(DiagnosticHook())
    runner.test()


if __name__ == '__main__':
    main()
