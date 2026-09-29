"""Resolve all five Route transfers against existing dataset/scale recipes."""
import copy
import json
from pathlib import Path
import tempfile

from submit import CATALOG, MATRIX, ROOT


def main():
    from mmengine.config import Config

    def read(path):
        return Config.fromfile(str(ROOT / path), import_custom_modules=False).to_dict()

    datasets = {'ade': ('ade20k', 160000, 512, 150, 4),
                'stuff': ('stuff164k', 80000, 512, 171, 4),
                'city': ('cityscapes', 160000, 1024, 19, 2)}
    sizes = {'t': ('Tiny', 's1', [32, 48, 120, 224], [32, 32, 64, 128]),
             'l': ('Large', 'l', [40, 80, 192, 384], [32, 64, 128, 256])}
    directories = set()
    for name in MATRIX:
        _, size, dataset = name.split('_')
        data, steps, crop, classes, batch = datasets[dataset]
        folder, backbone, channels, projected = sizes[size]
        item = CATALOG[name]
        cfg = read(item['config'])
        parent = read('local_configs/offseg2/Base/'
                      f'offsegccmiacs_protoroute_r4_responsibility_{data}_{steps//1000}k-{crop}x{crop}.py')
        expected = copy.deepcopy(parent)
        expected['model']['backbone'].update(
            type=f'efficientformerv2_{backbone}_feat',
            init_cfg=dict(type='Pretrained', checkpoint=item['backbone_checkpoint']))
        expected['model']['decode_head'].update(in_channels=channels, new_channels=projected)
        expected['work_dir'] = cfg['work_dir']
        assert cfg == expected, name + ': unexpected change beyond scale/work_dir'
        # Independently check the scale against its original OffSeg model recipe.
        original_steps = 160 if dataset == 'stuff' else steps // 1000
        original = read(f'local_configs/offseg/{folder}/offseg-{size}_{data}_{original_steps}k-{crop}x{crop}.py')
        assert cfg['model']['backbone'] == original['model']['backbone']
        for key in ('in_channels', 'new_channels', 'channels', 'num_classes'):
            assert cfg['model']['decode_head'][key] == original['model']['decode_head'][key]
        head = cfg['model']['decode_head']
        assert head['type'] == 'OffSegCCMIACSProtoRoute'
        assert head['num_classes'] == classes and head['acs_rank'] == 4
        assert head['ccm_rank'] == 64 and head['proto_warmup'] == 4000
        assert head['proto_momentum'] == .01 and head['proto_n0_init'] == 200.
        assert cfg['randomness']['seed'] == item['seed']
        assert cfg['train_cfg']['max_iters'] == cfg['param_scheduler'][-1]['end'] == steps
        assert cfg['train_dataloader']['batch_size'] == batch
        assert tuple(cfg['model']['data_preprocessor']['size']) == (crop, crop)
        assert cfg['load_from'] is None and cfg['resume'] is False
        hook = cfg['default_hooks']['checkpoint']
        assert hook['interval'] == cfg['train_cfg']['val_interval'] == steps // 20
        assert hook['max_keep_ckpts'] == 2 and hook['save_best'] == 'mIoU'
        assert f'_{size}_' in item['config'] and cfg['work_dir'].endswith(Path(item['config']).stem)
        assert cfg['work_dir'] not in directories
        directories.add(cfg['work_dir'])
        print(f'PASS {name}: {backbone}, {classes} classes, {steps} iterations, total batch {batch*4}')
    assert len(directories) == 5
    # Report source labels must not convert unfinished or missing runs to finals.
    from route_matrix import cell, HISTORY
    assert len(HISTORY) == 4
    with tempfile.TemporaryDirectory(prefix='route_matrix_') as temp:
        root = Path(temp)
        assert cell('ade', 't', root)[0] == '--'
        run = root / '20260929_001' / 'route_t_ade'
        work = run / 'checkpoints'
        log_dir = work / '20260929_002'
        log_dir.mkdir(parents=True)
        (run / 'run.json').write_text(json.dumps(dict(work_dir=str(work))))
        (log_dir / 'train.log').write_text(
            'The best checkpoint with 43.20 mIoU at 8000 iter\n'
            'The best checkpoint with 44.10 mIoU at 16000 iter\n')
        value, note = cell('ade', 't', root)
        assert value == '44.10' and 'MISSING' in note and 'best so far @16000' in note
        (work / 'best_mIoU_iter_16000.pth').touch()
        assert 'best file=yes' in cell('ade', 't', root)[1]
        assert cell('stuff', 'b', root)[0] == '44.75*'
        assert 'best unverified' in cell('stuff', 'b', root)[1]
    print('PASS 3x3 report: historical source labels, unfinished best and checkpoint presence')


if __name__ == '__main__':
    main()
