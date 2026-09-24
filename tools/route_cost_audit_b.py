"""Same-device FP32 crop-forward cost comparison: OffSeg-B vs Proto-route-B.

Without --offseg-checkpoint, OffSeg uses the Route checkpoint's shared weights
ONLY for architectural cost measurement; no baseline accuracy is evaluated.
Unsupported traced operators are disclosed, never reported as zero-cost.
"""
import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path


def normalise_state(payload):
    state = payload.get('state_dict', payload)
    return {k.removeprefix('module.'):v for k,v in state.items()}


def prepare_eval(module):
    """Build EfficientFormer eval caches without an autograd graph.

    Its train(False) override computes a plain tensor `ab` from a parameter.
    Calling eval() outside no_grad leaves a requires-grad constant in tracing.
    This changes neither weights nor inference arithmetic.
    """
    import torch
    with torch.no_grad():
        module.eval()
    return module


def short_error(exc):
    # Tracer failures can include megabytes of a cached tensor's contents.
    return type(exc).__name__ + ': ' + str(exc).split('\nTensor:', 1)[0][:1500]


def write_summary(report, output):
    lines = ['# OffSeg-B / Proto-route-B cost audit', '', report['protocol'], '',
             '| Model | Parameters | Mean wall ms | Images/s | Peak allocated MiB | FLOPs status |',
             '|---|---:|---:|---:|---:|---|']
    for label, item in report['models'].items():
        latency = statistics.mean(r['wall_mean_ms'] for r in item['repeats'])
        peak = max(r['peak_allocated_bytes'] for r in item['repeats']) / 2**20
        lines.append(f"| {label} | {item['parameters']} | {latency:.3f} | {1000/latency:.2f} | {peak:.1f} | {item['complexity']['status']} |")
    lines += ['', 'Weight sources:', *[f"- {label}: {item['weights']}" for label,item in report['models'].items()]]
    (output/'cost_summary.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('\n'.join(lines))


def main():
    import torch
    from torch import nn
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmseg.registry import MODELS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('route_config')
    parser.add_argument('route_checkpoint')
    parser.add_argument('--offseg-config', default='local_configs/offseg/Base/offseg-b_ade20k_160k-512x512.py')
    parser.add_argument('--offseg-checkpoint')
    parser.add_argument('--work-dir', required=True)
    parser.add_argument('--warmup', type=int, default=30)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--worker-model', choices=('offseg', 'route'), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.warmup < 1 or args.iterations < 10:
        raise ValueError('Require warmup >=1 and at least ten timed calls')
    if not torch.cuda.is_available():
        raise RuntimeError('Timing requires a CUDA GPU')
    output = Path(args.work_dir)
    output.mkdir(parents=True, exist_ok=True)
    if args.worker_model is None:
        # Each model gets a new CUDA process. Neither tracing failures nor allocator
        # state from OffSeg can inflate Route's resident/peak memory measurement.
        report = None
        for worker in ('offseg', 'route'):
            cmd = [sys.executable, str(Path(__file__).resolve()), args.route_config,
                   args.route_checkpoint, '--offseg-config', args.offseg_config,
                   '--work-dir', str(output), '--warmup', str(args.warmup),
                   '--iterations', str(args.iterations), '--worker-model', worker]
            if args.offseg_checkpoint:
                cmd += ['--offseg-checkpoint', args.offseg_checkpoint]
            subprocess.run(cmd, check=True)
            result = json.loads((output / ('cost_' + worker + '.json')).read_text(encoding='utf-8'))
            if report is None:
                report = result
            else:
                for key in ('gpu', 'torch', 'cuda', 'cudnn', 'protocol'):
                    if result[key] != report[key]:
                        raise RuntimeError('Cost workers disagree on '+key)
                report['models'].update(result['models'])
            (output/'cost_audit.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
        write_summary(report, output)
        if any(item['complexity']['status']=='FAILED' for item in report['models'].values()):
            raise SystemExit('Cost timing saved, but FLOPs tracing FAILED; inspect cost_audit.json')
        return
    route_cfg, base_cfg = Config.fromfile(args.route_config), Config.fromfile(args.offseg_config)
    init_default_scope('mmseg')
    if route_cfg.model.decode_head.type != 'OffSegCCMIACSProtoRoute' or base_cfg.model.decode_head.type != 'OffSegHead':
        raise ValueError('Require original Route and original OffSeg')
    if route_cfg.model.backbone.to_dict() != base_cfg.model.backbone.to_dict():
        raise ValueError('Backbones must match')
    state = normalise_state(torch.load(args.route_checkpoint, map_location='cpu'))
    torch.manual_seed(1370346084)
    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    x = torch.randn(1, 3, 512, 512, device='cuda')
    metas = [dict(img_shape=(512,512), ori_shape=(512,512), pad_shape=(512,512),
                  padding_size=[0,0,0,0], flip=False)]

    class CropForward(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, image):
            return self.model.encode_decode(image, metas)

    report = dict(protocol='Batch1, 512x512, FP32, TF32 off, cudnn_benchmark=False, single GPU. '
                          'Prepared synthetic normalized-shape input; includes backbone, full head and output resize. '
                          'Excludes file I/O, preprocessing, argmax and full-image sliding-window assembly. '
                          'Latency is synchronized wall time; CUDA time is also supplied. Three repeated measurements, not training seeds. '
                          'Each model is measured in its own fresh process; eval caches are built under no_grad. '
                          'FLOPs use MMEngine convention (one fused multiply-add counts as one); counts with unsupported ops are partial.',
                  gpu=torch.cuda.get_device_name(0), torch=torch.__version__, python=platform.python_version(),
                  cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(), models={})
    for label, cfg in (('OffSeg-B', base_cfg), ('Proto-route-B', route_cfg)):
        if (label == 'OffSeg-B') != (args.worker_model == 'offseg'):
            continue
        model = MODELS.build(cfg.model)
        if label == 'OffSeg-B':
            if args.offseg_checkpoint:
                model.load_state_dict(normalise_state(torch.load(args.offseg_checkpoint, map_location='cpu')), strict=True)
                weight_source = str(Path(args.offseg_checkpoint).resolve())
            else:
                shared = {k:state[k] for k in model.state_dict() if k in state}
                model.load_state_dict(shared, strict=True)
                weight_source = 'Shared tensors from Route checkpoint; architecture cost only, NOT trained OffSeg accuracy'
        else:
            model.load_state_dict(state, strict=True)
            weight_source = str(Path(args.route_checkpoint).resolve())
        model.cuda()
        wrapper = prepare_eval(CropForward(model))
        saved = {k:v.detach().cpu().clone() for k,v in model.named_buffers()}
        entry = dict(weights=weight_source, parameters=sum(p.numel() for p in model.parameters()),
                     buffers_bytes=sum(b.numel()*b.element_size() for b in model.buffers()), repeats=[])
        with torch.no_grad():
            for _ in range(args.warmup):
                wrapper(x)
            torch.cuda.synchronize()
            for _ in range(3):
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                resident = torch.cuda.memory_allocated()
                walls, cuda_times = [], []
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                for _ in range(args.iterations):
                    torch.cuda.synchronize()
                    before = time.perf_counter()
                    start.record()
                    result = wrapper(x)
                    end.record()
                    torch.cuda.synchronize()
                    walls.append(1000 * (time.perf_counter() - before))
                    cuda_times.append(start.elapsed_time(end))
                    del result
                peak = torch.cuda.max_memory_allocated()
                entry['repeats'].append(dict(wall_mean_ms=statistics.mean(walls), wall_median_ms=statistics.median(walls),
                    wall_p95_ms=sorted(walls)[int(.95*(len(walls)-1))], cuda_mean_ms=statistics.mean(cuda_times),
                    images_per_second=1000/statistics.mean(walls), resident_bytes=resident,
                    peak_allocated_bytes=peak, incremental_peak_bytes=peak-resident))
            try:
                from mmengine.analysis import FlopAnalyzer
                analyzer = FlopAnalyzer(wrapper, (x,))
                total = analyzer.total()
                unsupported = dict(analyzer.unsupported_ops())
                entry['complexity'] = dict(counted_flops=total, unsupported_ops=unsupported,
                    by_operator=dict(analyzer.by_operator()), uncalled_modules=sorted(analyzer.uncalled_modules()),
                    status='PARTIAL_UNSUPPORTED_OPS' if unsupported else 'TRACED_ESTIMATE',
                    note='Do not publish as complete FLOPs without reviewing operator coverage, especially FreqFusion/CARAFE.')
            except Exception as exc:
                entry['complexity'] = dict(status='FAILED', error=short_error(exc))
            finally:
                if 'analyzer' in locals():
                    del analyzer  # Do not retain the first GPU model during the second measurement.
        for k,v in model.named_buffers():
            if not torch.equal(saved[k], v.cpu()):
                raise RuntimeError('Cost measurement changed model buffer: '+k)
        report['models'][label] = entry
        (output/('cost_' + args.worker_model + '.json')).write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
