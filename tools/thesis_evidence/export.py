"""Strict, read-only original evaluation; class PNGs plus bounded Route probes."""
import argparse
import os
from pathlib import Path
import numpy as np
from PIL import Image
from .common import read_json, write_json, image_id, confusion, class_metrics
from .probe import RouteCapture


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
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--label', choices=('offseg', 'route'), required=True)
    parser.add_argument('--launcher', default='none', choices=('none', 'pytorch'))
    parser.add_argument('--local-rank', '--local_rank', type=int, default=0)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--cases', type=Path)
    parser.add_argument('--expected-miou', type=float)
    parser.add_argument('--owner-training-checkpoint', action='store_true',
                        help='Only for the owner-created full training checkpoint with MMEngine metadata; never public downloads')
    args = parser.parse_args()
    os.environ.setdefault('LOCAL_RANK', str(args.local_rank))
    cfg = Config.fromfile(args.config)
    expected_type = 'OffSegHead' if args.label == 'offseg' else 'OffSegCCMIACSProtoRoute'
    if cfg.model.decode_head.type != expected_type or cfg.test_dataloader.batch_size != 1:
        raise ValueError('Use original B ADE configuration, batch 1')
    cases = {r['id']: r for r in read_json(args.cases)} if args.cases else None
    if cases:
        cfg.test_dataloader.dataset.indices = [r['index'] for r in cases.values()]
    elif args.limit:
        cfg.test_dataloader.dataset.indices = list(range(args.limit))
    cfg.work_dir, cfg.launcher = str(args.output/'runner'), args.launcher
    # Strict safe tensor loading occurs in before_test; do not let Runner
    # invoke its unrestricted legacy checkpoint loader first.
    cfg.load_from, cfg.resume = None, False
    cfg.test_dataloader.sampler = dict(type='DefaultSampler', shuffle=False)
    cfg.test_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
    runner = Runner.from_cfg(cfg)

    class ExportHook(Hook):
        def before_test(self, runner):
            self.model = runner.model.module if is_model_wrapper(runner.model) else runner.model
            if len(runner.test_dataloader.dataset) % get_world_size():
                raise ValueError('Uneven distributed sample count would duplicate examples')
            if not cases and not args.limit and len(runner.test_dataloader.dataset) != 2000:
                raise ValueError('Full ADE validation must contain exactly 2000 images')
            # Official download is tensors-only. Owner-created full training
            # files additionally contain MMEngine HistoryBuffer metadata.
            state = torch.load(args.checkpoint, map_location='cpu',
                               weights_only=not args.owner_training_checkpoint)
            state = state.get('state_dict', state)
            self.model.load_state_dict({k.removeprefix('module.'): v for k, v in state.items()}, strict=True)
            del state
            self.saved = {k: v.detach().cpu().clone() for k, v in self.model.named_buffers()}
            self.rows, self.confusion = [], np.zeros((150, 150), dtype='int64')
            self.transparency_checked = False
            if self.model.decode_head.num_classes != 150:
                raise ValueError('This evidence job is ADE20K / 150 classes only')
            dataset = runner.test_dataloader.dataset
            self.indices = {str(dataset.get_data_info(i)['img_path']): i for i in range(len(dataset))}
            self.probe = RouteCapture(self.model, dense=bool(cases)) if args.label == 'route' else None
            if self.probe:
                self.probe.install()
            for name in ('predictions', 'gt', 'memory', 'mechanisms'):
                (args.output/name).mkdir(parents=True, exist_ok=True)

        def before_test_iter(self, runner, batch_idx, data_batch=None):
            if self.probe and cases:
                path = data_batch['data_samples'][0].img_path
                self.probe.class_ids = cases[image_id(path)]['class_ids']

        def after_test_iter(self, runner, batch_idx, data_batch=None, outputs=None):
            if self.model.training or len(outputs) != 1:
                raise RuntimeError('Require unmodified batch-1 evaluation')
            # The first actual validation image must have identical logits
            # with observation wrappers removed, not merely identical argmax.
            if self.probe and not self.transparency_checked:
                windows, geometry = self.probe.windows, self.probe.geometry
                self.probe.close()
                with torch.no_grad():
                    reference = self.model.test_step(data_batch)
                torch.testing.assert_close(outputs[0].seg_logits.data,
                                           reference[0].seg_logits.data, rtol=0, atol=0)
                self.probe.install()
                self.probe.windows, self.probe.geometry = windows, geometry
                self.transparency_checked = True
            for sample in outputs:
                if not torch.isfinite(sample.seg_logits.data).all():
                    raise RuntimeError('Non-finite predictions')
                pred = sample.pred_sem_seg.data[0].detach().cpu().numpy().astype('uint8')
                gt = sample.gt_sem_seg.data[0].detach().cpu().numpy().astype('uint8')
                path, key = str(sample.img_path), image_id(sample.img_path)
                index = cases[key]['index'] if cases else self.indices[path]
                self.confusion += confusion(gt, pred, 150)
                Image.fromarray(pred).save(args.output/'predictions'/f'{key}.png')
                Image.fromarray(gt).save(args.output/'gt'/f'{key}.png')
                self.rows.append(dict(id=key, index=index, image=path, shape=list(gt.shape)))
                if self.probe:
                    windows = self.probe.consume(gt)
                    np.savez_compressed(args.output/'memory'/f'{key}.npz',
                        support=np.stack([w['support'] for w in windows]),
                        lam=np.stack([w['lam'] for w in windows]),
                        gt_area=np.stack([w['gt_area'] for w in windows]),
                        valid_pixels=np.array([w['valid_pixels'] for w in windows]),
                        boxes=np.array([w['box'] for w in windows]))
                    if cases:
                        folder = args.output/'mechanisms'/key
                        folder.mkdir(exist_ok=True)
                        for number, window in enumerate(windows):
                            np.savez_compressed(folder/f'crop_{number:03d}.npz', **window)
                        write_json(folder/'geometry.json', self.probe.geometry)

        def after_test_epoch(self, runner, metrics=None):
            for k, v in self.model.named_buffers():
                if not torch.equal(v.cpu(), self.saved[k]):
                    raise RuntimeError('Evaluation mutated buffer: '+k)
            if self.probe:
                self.probe.close()
            shards = all_gather_object(dict(rows=self.rows, confusion=self.confusion))
            rows = sorted([r for s in shards for r in s['rows']], key=lambda r: r['index'])
            if len({r['id'] for r in rows}) != len(rows):
                raise ValueError('Duplicate images/ids in output')
            names = list(runner.test_dataloader.dataset.metainfo['classes'])
            matrix = sum(s['confusion'] for s in shards)
            per_class, score = class_metrics(matrix, names)
            if abs(score-float(metrics['mIoU'])) > .011:
                raise RuntimeError('PNG confusion and original evaluator disagree')
            full = not cases and not args.limit
            check = 'NOT_REQUESTED'
            if full and args.expected_miou is not None:
                check = 'PASS' if abs(score-args.expected_miou) <= .02 else 'MISMATCH'
            if get_rank() == 0:
                write_json(args.output/'evaluation.json', dict(
                    label=args.label, config=args.config, checkpoint=args.checkpoint,
                    image_count=len(rows), scope='full' if full else 'subset',
                    miou=score, metrics=metrics, expected_check=check,
                    classes=names, palette=runner.test_dataloader.dataset.metainfo['palette'],
                    per_class=per_class, confusion=matrix.tolist(), images=rows,
                    test_cfg=dict(cfg.model.test_cfg), buffers_unchanged=True,
                    observation_transparency='exact first-image logits verified' if self.probe else 'output-only observer',
                    learned_scalars=dict(n0=float(torch.nn.functional.softplus(self.model.decode_head.proto_n0_raw).detach()),
                                         iacs_mix=float(self.model.decode_head.acs.mix_logit.sigmoid().detach()))
                                    if args.label == 'route' else {},
                    gt_usage='Post-prediction diagnostics only; never fed to the model'))
            if check == 'MISMATCH':
                raise RuntimeError('Route did not reproduce 48.49 +/- 0.02; evidence saved, interpretation blocked')

    runner.register_hook(ExportHook())
    runner.test()


if __name__ == '__main__':
    main()
