# -*- coding: utf-8 -*-
"""Active-class oracle: 量"场景/关系"这条原语的 mIoU 天花板。

思路:冻住已训练的 base,在 val 上拿到每张图的 dense logits;对每张图,
只保留 GT 里真实出现过的类(把缺席类的 logit 置 -inf)再 argmax。
得到的 mIoU 与原始 mIoU 之差 = "完美知道这张图里有哪些类"能带来的上界。
这是保守下界(只量了'在场'那半,'相邻/布局'那半的收益还叠在上面)。

自检:打印的 baseline mIoU 应当 ≈ 你已知的 48.2;若对得上,说明评测口径没问题,
oracle 的 delta 就可信。

用法(服务器单卡即可):
    cd offseg2
    # 先小样本确认能跑通:
    python tools/oracle_active_class.py \
        local_configs/offseg2/Base/parseg3_ade20k_160k-512x512.py \
        <你的base_ckpt>.pth --max-images 20
    # 再跑全量 val:
    python tools/oracle_active_class.py \
        local_configs/offseg2/Base/parseg3_ade20k_160k-512x512.py \
        <你的base_ckpt>.pth

不改任何现有文件,纯只读评测。
"""
import argparse

import torch
import torch.nn.functional as F
from mmengine.config import Config
from mmengine.runner import Runner

from mmseg.apis import init_model


def parse_args():
    p = argparse.ArgumentParser(description='Active-class oracle ceiling probe (read-only).')
    p.add_argument('config', help='base 的 config(如 parseg3 ade)')
    p.add_argument('checkpoint', help='已训练 base 的 .pth')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--max-images', type=int, default=-1, help='>0 时只跑前 N 张,用于 smoke')
    return p.parse_args()


def intersect_union(pred, label, num_classes, ignore_index, device):
    """标准 mmseg IoU 累加口径。"""
    mask = label != ignore_index
    pred = pred[mask].float()
    label = label[mask].float()
    inter = pred[pred == label]
    ai = torch.histc(inter, bins=num_classes, min=0, max=num_classes - 1).to(device)
    ap = torch.histc(pred, bins=num_classes, min=0, max=num_classes - 1).to(device)
    al = torch.histc(label, bins=num_classes, min=0, max=num_classes - 1).to(device)
    return ai, ap + al - ai


@torch.no_grad()
def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    model = init_model(cfg, args.checkpoint, device=args.device)
    model.eval()

    dev = args.device
    num_classes = int(model.decode_head.num_classes)
    ignore_index = int(getattr(model.decode_head, 'ignore_index', 255))
    val_loader = Runner.build_dataloader(cfg.val_dataloader)

    ai_v = torch.zeros(num_classes, device=dev)
    au_v = torch.zeros(num_classes, device=dev)
    ai_o = torch.zeros(num_classes, device=dev)
    au_o = torch.zeros(num_classes, device=dev)
    n = 0
    pruned_sum = 0

    for data in val_loader:
        results = model.test_step(data)
        for r in results:
            logits = r.seg_logits.data.float().to(dev)          # (C, H, W)
            gt = r.gt_sem_seg.data.squeeze(0).long().to(dev)    # (H, W)
            if logits.shape[-2:] != gt.shape[-2:]:
                logits = F.interpolate(
                    logits[None], size=gt.shape[-2:], mode='bilinear', align_corners=False)[0]

            present = torch.unique(gt[gt != ignore_index])

            pred_v = logits.argmax(0)

            # 只在'在场'类里竞争:缺席类 +(-inf)
            neg = torch.full((num_classes, 1, 1), float('-inf'), device=dev)
            neg[present] = 0.0
            pred_o = (logits + neg).argmax(0)

            a, u = intersect_union(pred_v, gt, num_classes, ignore_index, dev)
            ai_v += a; au_v += u
            a, u = intersect_union(pred_o, gt, num_classes, ignore_index, dev)
            ai_o += a; au_o += u

            pruned_sum += num_classes - int(present.numel())
            n += 1

        if n and n % 200 == 0:
            print(f'  ...{n} images', flush=True)
        if args.max_images > 0 and n >= args.max_images:
            break

    vv = au_v > 0
    vo = au_o > 0
    miou_v = (ai_v[vv] / au_v[vv]).mean().item() * 100
    miou_o = (ai_o[vo] / au_o[vo]).mean().item() * 100

    print('=' * 48)
    print(f'images evaluated      : {n}')
    print(f'avg classes pruned/img: {pruned_sum / max(n, 1):.1f} / {num_classes}')
    print(f'baseline mIoU         : {miou_v:.2f}   (应 ≈ 48.2 做自检)')
    print(f'active-class oracle   : {miou_o:.2f}')
    print(f'CEILING delta         : +{miou_o - miou_v:.2f}')
    print('=' * 48)
    print('判读:delta ≥ ~1.0 → 这条轴够同量级,值得搭子系统;'
          '接近 0 → 当场换轴,零成本。')


if __name__ == '__main__':
    main()
