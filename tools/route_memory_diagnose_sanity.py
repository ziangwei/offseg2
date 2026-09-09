"""Check probe transparency on the real head, GT grouping and rank merging."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
from route_memory_diagnose import RouteProbe, summarise
from proto_variants_sanity import load_components, make


def config_check():
    from mmengine.config import Config
    folder = Path(__file__).resolve().parents[1] / 'local_configs/offseg2/Base'
    for arm in ('protoroute', 'protoroute_fastmem'):
        cfg = Config.fromfile(str(folder / ('offsegccmiacs_' + arm +
                              '_r4_responsibility_ade20k_160k-512x512.py')),
                              import_custom_modules=False)
        assert cfg.model.test_cfg.mode == 'slide'
        assert tuple(cfg.model.test_cfg.crop_size) == (512, 512)
        assert tuple(cfg.model.test_cfg.stride) == (480, 480)
        assert cfg.model.decode_head.type == 'OffSegCCMIACSProtoRoute'
        assert cfg.test_dataloader.batch_size == 1
        assert not cfg.test_dataloader.sampler.shuffle
    print('PASS real resolved configs: original sliding-window ADE validation')


def main():
    torch.set_num_threads(1)
    _, variants, _ = load_components()
    model = make(variants.OffSegCCMIACSProtoRoute, warmup=1)
    x = torch.randn(2, 12, 4, 4)
    with torch.no_grad():
        model.train()(x)  # Fill a genuine memory before testing.
    model.eval()
    state = copy.deepcopy(model.state_dict())
    with torch.no_grad():
        reference = model(x)
        rng = torch.get_rng_state().clone()
        probe = RouteProbe(model)
        probe.install()
        measured = model(x)
        for key in reference:
            torch.testing.assert_close(reference[key], measured[key], rtol=0, atol=0)
        assert torch.equal(rng, torch.get_rng_state())
        assert all(torch.isfinite(v).all() for v in probe.batches[0].values())
        # GT exists only at consume(); it cannot feed into model inference.
        gts = [torch.tensor([[0, 0, 255, 255]] * 4), torch.ones(4, 4).long()]
        preds = [torch.full((4, 4), 2), torch.ones(4, 4).long()]
        outputs = [SimpleNamespace(gt_sem_seg=SimpleNamespace(data=g.unsqueeze(0)),
                                   pred_sem_seg=SimpleNamespace(data=p.unsqueeze(0)))
                   for g, p in zip(gts, preds)]
        expected_present = float(probe.batches[0]['lambda'][0, 0] + probe.batches[0]['lambda'][1, 1]) / 2
        probe.consume(outputs)
        shard = probe.finish()
    summary = summarise([shard])
    assert summary['images'] == 2
    assert summary['groups']['all']['image_class_pairs'] == 10
    assert summary['groups']['present']['image_class_pairs'] == 2
    assert summary['groups']['absent']['image_class_pairs'] == 8
    assert abs(summary['groups']['present']['means']['lambda'] - expected_present) < 1e-7
    assert summary['valid_pixels'] == 24
    assert summary['errors'] == summary['absent_fp'] == 8
    assert summary['absent_fp_fraction_valid'] == 1 / 3
    assert summary['absent_fp_fraction_errors'] == 1
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, state[key], rtol=0, atol=0)
    # Unequal rank counts must aggregate sums/counts, not rank means.
    empty = copy.deepcopy(shard)
    empty.update(images=0, valid_pixels=0, errors=0, absent_fp=0, head_calls=0)
    empty['groups'] = {k: dict(count=0, sums={}) for k in shard['groups']}
    assert summarise([shard, empty]) == summary
    doubled = summarise([shard, shard])
    assert doubled['groups']['present']['means'] == summary['groups']['present']['means']
    print('PASS exact logits/RNG/state preservation, ignored labels, GT presence, absent-FP and rank reduction')
    # Zero norm centres must produce finite diagnostics.
    model.prototypes.zero_()
    model.proto_seen.zero_()
    probe = RouteProbe(model)
    probe.install()
    with torch.no_grad():
        model(x)
        assert all(torch.isfinite(v).all() for v in probe.batches[0].values())
        # Simulate two slide windows before one output batch is consumed.
        model(x * 2)
        assert len(probe.batches) == 2
        expected = sum(float(b['image_centre_norm'].sum()) for b in probe.batches) / 20
        probe.consume(outputs)
    merged = summarise([probe.finish()])
    assert abs(merged['groups']['all']['means']['image_centre_norm'] - expected) < 1e-8
    assert merged['head_calls'] == 2
    print('PASS multiple sliding windows, zero/unseen prototype diagnostics and frozen eval buffers')


if __name__ == '__main__':
    config_check() if '--configs-only' in sys.argv else main()
