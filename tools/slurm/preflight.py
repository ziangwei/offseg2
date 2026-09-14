"""Fail early on GPU/path/protocol mistakes in an allocated Slurm job."""
import argparse
import json
import os
from pathlib import Path
import sys


def main():
    import torch
    import mmcv.ops  # Check the installed compiled ops before starting 4 workers.
    from mmengine.config import Config
    p = argparse.ArgumentParser()
    p.add_argument('config')
    p.add_argument('--mode', choices=['train', 'eval'], required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--work-dir')
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    cfg = Config.fromfile(a.config)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 4:
        raise RuntimeError('Expected exactly 4 Slurm-visible CUDA GPUs')
    work = Path(a.work_dir or cfg.work_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    # Check image/annotation directories, including absolute DSS data roots.
    for split in ('train_dataloader', 'val_dataloader', 'test_dataloader'):
        ds = cfg[split]['dataset']
        root = Path(ds.get('data_root', '.'))
        for prefix in ds.get('data_prefix', {}).values():
            path = root / prefix
            if not path.is_dir():
                raise FileNotFoundError('{}: {}'.format(split, path))
    if a.mode == 'train' and not a.resume:
        weight = cfg.model.backbone.get('init_cfg', {}).get('checkpoint')
        if weight and not Path(weight).is_file():
            raise FileNotFoundError('Backbone weights: ' + weight)
        if any(work.glob('*.pth')) or (work / 'last_checkpoint').exists():
            raise RuntimeError('Work directory already has a checkpoint; explicitly resume or use a new directory')
    resume_checkpoint = None
    if a.resume:
        pointer = work / 'last_checkpoint'
        checkpoint = Path(pointer.read_text().strip())
        if not checkpoint.is_absolute():
            # Historical runs can record work_dirs/... relative to the old
            # checkout; this job now runs from its frozen source directory.
            local = work / checkpoint.name
            checkpoint = local if local.is_file() else Path.cwd() / checkpoint
        if not checkpoint.is_file():
            raise FileNotFoundError('Resume checkpoint: ' + str(checkpoint))
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if not all(key in saved for key in ('optimizer', 'param_schedulers', 'state_dict')):
            raise RuntimeError('Resume requires model, optimizer and scheduler state')
        if saved.get('meta', {}).get('iter', -1) >= cfg.train_cfg.max_iters:
            raise RuntimeError('Training already reached max_iters; use --evaluate-run to retry final evaluation')
        del saved
        resume_checkpoint = str(checkpoint.resolve())
    final = work / 'iter_{}.pth'.format(cfg.train_cfg.max_iters)
    if a.mode == 'eval' and not final.is_file():
        raise FileNotFoundError('Final checkpoint: ' + str(final))
    record = dict(config=a.config, work_dir=str(work), final_checkpoint=str(final),
                  max_iters=cfg.train_cfg.max_iters, seed=cfg.randomness.seed,
                  backbone=cfg.model.backbone.type, head=cfg.model.decode_head.type,
                  batch_per_gpu=cfg.train_dataloader.batch_size, gpu_count=4,
                  python=sys.executable, torch=torch.__version__, mode=a.mode,
                  job_id=os.environ.get('SLURM_JOB_ID'), resume=a.resume,
                  resume_checkpoint=resume_checkpoint)
    Path(a.output).write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
