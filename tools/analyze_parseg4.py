# -*- coding: utf-8 -*-
"""PARSeg4 体检脚本: 训完后在 val 上复算精修头中间量, 输出紧凑文本报告.

诊断对象(对应 MA/PARSeg4_理论分析_隐患与提升空间.md):
  [HEADS]  base/refine/final 各自像素准确率 + 2x2 对错格 + fusion 捕获率 —— 谁是瓶颈
  [A_USAGE] 分量使用率(P5 WTA 饥饿判读): 每类 responsibility-argmax 的有效分量数
  [B_SIGMA] σ²分布 + σ²-错误率相关(P4 loss-attenuation 判读)
  [C_FUSION] w_r=pr/(pb+pr) 分布与条件均值(P3/P8 判读, GT 类通道)
  [D_PI]   路由 π 的有效分量数(route 是否还在均匀附近)
  [E_EXTRA] 分量间方差预览(P2) + refine_var 预测 final 错误的 AUROC(R1 预览)

用法(服务器):
  python tools/analyze_parseg4.py \
      local_configs/offseg2/Base/parseg4_ade20k_160k-512x512.py \
      work_dirs/parseg4_ade20k_160k-512x512/iter_160000.pth \
      --max-images 250
输出: stdout + <ckpt所在目录>/parseg4_analysis_<ckpt名>.txt (身份信息含 work_dir 子文件夹名)

说明: 整图 forward(非 slide), 输入 pad 到 32 倍数, 仅用于统计诊断, 不复算官方 mIoU。
新文件, 不改任何现有文件。只适用于 PARSeg4 head。
"""
import argparse
import datetime
import math
import os

import numpy as np
import torch
import torch.nn.functional as F


def parse_args():
    p = argparse.ArgumentParser(description='PARSeg4 post-training analysis')
    p.add_argument('config')
    p.add_argument('checkpoint')
    p.add_argument('--max-images', type=int, default=250, help='val 抽样张数(等间隔)')
    p.add_argument('--chunk', type=int, default=30, help='类维分块大小(控显存)')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--sample-px', type=int, default=4000, help='AUROC 每图采样像素数')
    p.add_argument('--out', default=None, help='报告输出路径(默认存 ckpt 同目录)')
    return p.parse_args()


def rankdata(x):
    """平均秩(处理并列), 仅 numpy."""
    order = np.argsort(x, kind='mergesort')
    ranks = np.empty(len(x), dtype=np.float64)
    sx = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def pearson(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if len(a) < 3 or a.std() < 1e-12 or b.std() < 1e-12:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a, b):
    return pearson(rankdata(np.asarray(a)), rankdata(np.asarray(b)))


def auroc(scores, labels):
    """labels: 1=正类(预测错误). 秩公式, 含并列平均."""
    s = np.asarray(scores, np.float64)
    y = np.asarray(labels, np.float64)
    npos, nneg = y.sum(), (1 - y).sum()
    if npos < 10 or nneg < 10:
        return float('nan')
    r = rankdata(s)
    return float((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


@torch.no_grad()
def main():
    args = parse_args()
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    cfg = Config.fromfile(args.config)  # fromfile 会处理 custom_imports(注册 PARSeg4)
    init_default_scope('mmseg')
    from mmseg.registry import DATASETS
    from mmseg.apis import init_model
    from mmseg.models.decode_heads.PARSeg4 import LAMBDA_PROBIT  # 保险再触发一次注册

    model = init_model(args.config, args.checkpoint, device=args.device)
    model.eval()
    head = model.decode_head
    mrh = head.prototype_attribute_refinement
    use_sigma = bool(getattr(mrh, 'use_sigma', False))
    fusion_mode = getattr(head, 'fusion_mode', 'gate')
    inv_var_on = use_sigma and getattr(head, 'fusion_inv_var', None) is not None \
        and fusion_mode == 'inv_var'
    tau = float(mrh.tau)
    mss = int(getattr(mrh, 'match_stride_scale', 1))

    ds_cfg = cfg.val_dataloader.dataset if 'val_dataloader' in cfg else cfg.test_dataloader.dataset
    dataset = DATASETS.build(ds_cfg)
    n_total = len(dataset)
    n_imgs = min(args.max_images, n_total)
    indices = np.linspace(0, n_total - 1, n_imgs).astype(int)

    # checkpoint 元信息(身份)
    iter_meta = '?'
    try:
        meta = torch.load(args.checkpoint, map_location='cpu').get('meta', {})
        iter_meta = str(meta.get('iter', '?'))
    except Exception:
        pass

    da = cfg.model.decode_head
    Nc, A = int(da.num_classes), int(da.cls_attributes)
    ignore = 255

    # ---------------- 累加器 ----------------
    usage = torch.zeros(Nc, A, dtype=torch.float64)          # GT类像素的获胜分量计数
    sig_sum = torch.zeros(Nc, A, dtype=torch.float64)        # σ² 累加(图级)
    sig_n = torch.zeros(Nc, dtype=torch.float64)             # 该类出现的图数
    err_win = torch.zeros(Nc, A, dtype=torch.float64)        # 获胜像素中 refine 判错数
    pi_effn_sum, pi_effn_n = 0.0, 0
    cells = dict(n=0, b=0, r=0, f=0, bo_ro=0, bo_rw=0, bw_ro=0, bw_rw=0,
                 cap_bw_ro=0, cap_bo_rw=0)
    wr_hist = torch.zeros(20, dtype=torch.float64)
    wr_cell_sum = dict(bo_ro=0.0, bo_rw=0.0, bw_ro=0.0, bw_rw=0.0)
    bvar_sum, bvar_n = 0.0, 0
    btw_sum, btw_n = 0.0, 0
    au_s, au_y = [], []
    sig_all = []

    from mmengine.dataset import pseudo_collate
    LAM = LAMBDA_PROBIT

    for k, idx in enumerate(indices):
        data = pseudo_collate([dataset[int(idx)]])
        data = model.data_preprocessor(data, False)
        inputs = data['inputs']
        if isinstance(inputs, (list, tuple)):
            inputs = inputs[0].unsqueeze(0)
        inputs = inputs.to(args.device)
        gt = data['data_samples'][0].gt_sem_seg.data.to(args.device)  # [1,H,W]

        # pad 到 32 倍数(整图 forward 需要; gt 同步 pad ignore)
        H, W = inputs.shape[-2:]
        ph, pw = (32 - H % 32) % 32, (32 - W % 32) % 32
        if ph or pw:
            inputs = F.pad(inputs, (0, pw, 0, ph), value=0.0)
            gt = F.pad(gt, (0, pw, 0, ph), value=ignore)

        # ---- 特征准备(复刻 PARSeg4.forward, 只读) ----
        feats = model.extract_feat(inputs)
        x = head._transform_inputs(list(feats))
        new_inputs = [head.pre[i](x[i]) for i in range(len(x))]
        lowres = new_inputs[-1]
        for hires, ff in zip(new_inputs[:-1][::-1], head.freqfusions):
            _, hires, lowres = ff(hr_feat=hires, lr_feat=lowres)
            b, _, h, w = hires.shape
            lowres = torch.cat([hires.reshape(b * 4, -1, h, w),
                                lowres.reshape(b * 4, -1, h, w)], dim=1).reshape(b, -1, h, w)
        feat_aligned = head.align(lowres)
        base = head.offset_learning(feat_aligned)                      # [1,Nc,h4,w4]

        # ---- 精修头中间量(复刻 MixtureRefineHead.forward) ----
        rf = mrh.refinement_feat_proj(feat_aligned)
        attr = mrh.spatial_attribute_decoder(refinement_feats=rf, base_head_logits=base)
        cal, proto = mrh.proto_refiner(attr_tokens=attr, refinement_feats=rf,
                                       base_head_logits=base)
        route = mrh.route_mlp(proto.detach()) + mrh.route_class_bias.weight.unsqueeze(0)
        log_pi = F.log_softmax(route, dim=-1)                          # [1,Nc,A]
        pi = log_pi.exp()
        ent = -(pi * log_pi).sum(-1).squeeze(0)                        # [Nc]

        rf_m = rf if mss <= 1 else F.avg_pool2d(rf, mss, mss)
        seg = F.normalize(rf_m.permute(0, 2, 3, 1), p=2, dim=-1, eps=1e-6)  # [1,hm,wm,D]
        comp = F.normalize(cal, p=2, dim=-1, eps=1e-6)                 # [1,Nc,A,D]
        hm, wm = seg.shape[1:3]
        h4, w4 = base.shape[-2:]

        var_c = None
        if use_sigma:
            lv = mrh.comp_logvar(cal).squeeze(-1).clamp(-8, 8)         # [1,Nc,A]
            var_c = lv.exp()

        gt_m = gt[0, ::4 * mss, ::4 * mss][:hm, :wm]                   # [hm,wm]
        gt4 = gt[0, ::4, ::4][:h4, :w4]                                # [h4,w4]
        present = torch.unique(gt_m)
        present = [int(c) for c in present if int(c) != ignore and int(c) < Nc]

        refine_m = torch.empty(1, Nc, hm, wm, device=args.device)
        refvar_m = torch.empty(1, Nc, hm, wm, device=args.device) if use_sigma else None
        win_cache = {}  # 本图: 类 -> 该类GT像素的获胜分量 [n]

        for c0 in range(0, Nc, args.chunk):
            c1 = min(c0 + args.chunk, Nc)
            sim = torch.einsum('bhwd,bcad->bchwa', seg, comp[:, c0:c1])    # [1,nc,hm,wm,A]
            if use_sigma:
                v = var_c[:, c0:c1, None, None, :]
                score = log_pi[:, c0:c1, None, None, :] + (sim / tau) / torch.sqrt(1 + LAM * v)
            else:
                score = log_pi[:, c0:c1, None, None, :] + sim / tau
            refine_m[:, c0:c1] = torch.logsumexp(score, dim=-1)
            resp = F.softmax(score, dim=-1)
            if use_sigma:
                refvar_m[:, c0:c1] = (resp * var_c[:, c0:c1, None, None, :]).sum(-1)

            for c in present:
                if not (c0 <= c < c1):
                    continue
                mask = gt_m == c
                if mask.sum() < 10:
                    continue
                rc = resp[0, c - c0][mask]                                  # [n,A]
                win = rc.argmax(-1)
                usage[c] += torch.bincount(win, minlength=A).double().cpu()
                # 分量间方差预览(P2): 未调制 sim/τ 的 resp 加权方差
                s_t = sim[0, c - c0][mask] / tau                            # [n,A]
                m1 = (rc * s_t).sum(-1)
                btw = ((rc * s_t * s_t).sum(-1) - m1 * m1).clamp_min(0)
                btw_sum += float(btw.mean()) * mask.sum().item()
                btw_n += int(mask.sum())
                # σ-错误配对的获胜数缓存(错误数在拿到 refine_pred 后补)
                if use_sigma:
                    sig_sum[c] += var_c[0, c].double().cpu()
                    sig_n[c] += 1
                win_cache[c] = win

        # refine 回 stride4(同 head 行为)
        if mss > 1:
            refine4 = F.interpolate(refine_m, size=(h4, w4), mode='bilinear', align_corners=False)
            refvar4 = F.interpolate(refvar_m, size=(h4, w4), mode='bilinear', align_corners=False) if use_sigma else None
        else:
            refine4, refvar4 = refine_m, refvar_m

        # ---- fusion ----
        w_r = None
        if inv_var_on:
            bvar = head.base_logvar_head(feat_aligned).clamp(-8, 8).exp()
            fus = head.fusion_inv_var
            bl = fus.scale_b * base + fus.shift_b
            rl = fus.scale_r * refine4 + fus.shift_r
            pb = 1.0 / (bvar * fus.scale_b ** 2 + 1e-6)
            pr = 1.0 / (refvar4 * fus.scale_r ** 2 + 1e-6)
            final = (pb * bl + pr * rl) / (pb + pr)
            w_r = pr / (pb + pr)                                       # [1,Nc,h4,w4]
            bvar_sum += float(bvar.mean())
            bvar_n += 1
        else:
            final = head.fusion(base, refine4)

        # ---- 正确性统计(stride4) ----
        valid4 = gt4 != ignore
        gtc = gt4.clamp(0, Nc - 1).long()
        bp = base[0].argmax(0)
        rp = refine4[0].argmax(0)
        fp_ = final[0].argmax(0)
        bo = (bp == gt4) & valid4
        ro = (rp == gt4) & valid4
        fo = (fp_ == gt4) & valid4
        nv = int(valid4.sum())
        cells['n'] += nv
        cells['b'] += int(bo.sum()); cells['r'] += int(ro.sum()); cells['f'] += int(fo.sum())
        m_bo_ro = bo & ro; m_bo_rw = bo & ~ro & valid4
        m_bw_ro = ~bo & ro & valid4; m_bw_rw = ~bo & ~ro & valid4
        cells['bo_ro'] += int(m_bo_ro.sum()); cells['bo_rw'] += int(m_bo_rw.sum())
        cells['bw_ro'] += int(m_bw_ro.sum()); cells['bw_rw'] += int(m_bw_rw.sum())
        cells['cap_bw_ro'] += int((fo & m_bw_ro).sum())
        cells['cap_bo_rw'] += int((fo & m_bo_rw).sum())

        if w_r is not None:
            wr_gt = w_r[0].gather(0, gtc.unsqueeze(0)).squeeze(0)      # [h4,w4] GT通道
            wr_hist += torch.histc(wr_gt[valid4].float(), bins=20, min=0, max=1).double().cpu()
            for key, m in (('bo_ro', m_bo_ro), ('bo_rw', m_bo_rw),
                           ('bw_ro', m_bw_ro), ('bw_rw', m_bw_rw)):
                if int(m.sum()) > 0:
                    wr_cell_sum[key] += float(wr_gt[m].sum())

        # σ-错误配对: 获胜像素中 refine 判错(用 m 分辨率的 refine_pred)
        if use_sigma and win_cache:
            rp_m = refine_m[0].argmax(0)                                # [hm,wm]
            for c, win in win_cache.items():
                mask = gt_m == c
                wrong = (rp_m[mask] != c)
                for a in range(A):
                    sel = win == a
                    if int(sel.sum()) > 0:
                        err_win[c, a] += int((wrong & sel).sum())

        # π 有效分量数(仅 present 类)
        for c in present:
            pi_effn_sum += float(torch.exp(ent[c]))
            pi_effn_n += 1

        # AUROC 采样: refine_var 预测 final 错误
        if use_sigma:
            vmask = valid4.flatten()
            sc = refvar4[0].gather(0, gtc.unsqueeze(0)).squeeze(0).flatten()[vmask]
            yy = (~fo).flatten()[vmask].float()
            n_s = min(args.sample_px, sc.numel())
            sel = torch.randperm(sc.numel(), device=sc.device)[:n_s]
            au_s.append(sc[sel].cpu().numpy()); au_y.append(yy[sel].cpu().numpy())
            sig_all.append(var_c.flatten().cpu().numpy())

        if (k + 1) % 50 == 0:
            print(f'... {k + 1}/{n_imgs} images')

    # ---------------- 汇总 ----------------
    L = []
    wd = os.path.basename(os.path.dirname(os.path.abspath(args.checkpoint)))
    L.append('[ID]')
    L.append(f'WORKDIR={wd}')
    L.append(f'CONFIG={os.path.splitext(os.path.basename(args.config))[0]}')
    L.append(f'CKPT={os.path.basename(args.checkpoint)} ITER={iter_meta}')
    aa = da.get('args', {})
    L.append(f'FLAGS tau={tau} A={A} heads={aa.get("mix_decoder_heads")} sigma={use_sigma} '
             f'fusion={fusion_mode} mss={mss}')
    L.append(f'EVAL n_images={n_imgs}/{n_total} date={datetime.date.today()} mode=whole-image(非slide,仅诊断)')

    n = max(cells['n'], 1)
    L.append('[HEADS]')
    L.append(f'acc base={cells["b"]/n:.4f} refine={cells["r"]/n:.4f} final={cells["f"]/n:.4f}')
    L.append(f'cells占比 bo_ro={cells["bo_ro"]/n:.4f} bo_rw={cells["bo_rw"]/n:.4f} '
             f'bw_ro={cells["bw_ro"]/n:.4f} bw_rw={cells["bw_rw"]/n:.4f}')
    L.append(f'oracle(任一头对)={1 - cells["bw_rw"]/n:.4f}')
    cap1 = cells['cap_bw_ro'] / max(cells['bw_ro'], 1)
    cap2 = cells['cap_bo_rw'] / max(cells['bo_rw'], 1)
    L.append(f'fusion捕获率 base错refine对时final对={cap1:.4f} | base对refine错时final对={cap2:.4f}')

    L.append('[A_USAGE](P5判读: 每类GT像素的获胜分量分布)')
    tot = usage.sum(1)
    ok = tot >= 500
    effn = []
    for c in range(Nc):
        if ok[c]:
            p = (usage[c] / tot[c]).numpy()
            p = p[p > 0]
            effn.append(float(np.exp(-(p * np.log(p)).sum())))
    if effn:
        e = np.array(effn)
        L.append(f'eff_comp/类(A={A}): mean={e.mean():.2f} p10={np.percentile(e,10):.2f} '
                 f'p50={np.percentile(e,50):.2f} p90={np.percentile(e,90):.2f} n类={len(e)}')
        L.append(f'塌缩类占比 eff<1.5: {(e<1.5).mean():.3f} | eff<2: {(e<2).mean():.3f}')
        share = (usage / tot.clamp_min(1)[:, None])[ok]
        L.append(f'top1分量平均占有率={share.max(1).values.mean():.3f}')

    L.append('[B_SIGMA](P4判读)')
    if use_sigma:
        sv = np.concatenate(sig_all) if sig_all else np.array([1.0])
        L.append(f'σ² 分位 p5={np.percentile(sv,5):.3f} p50={np.percentile(sv,50):.3f} '
                 f'p95={np.percentile(sv,95):.3f} | 触free-bits带边占比 '
                 f'低(<0.18)={ (sv<0.18).mean():.3f} 高(>3.1)={(sv>3.1).mean():.3f}')
        xs, ys = [], []
        for c in range(Nc):
            if sig_n[c] < 3:
                continue
            for a in range(A):
                if usage[c, a] >= 200:
                    xs.append(float(sig_sum[c, a] / sig_n[c]))
                    ys.append(float(err_win[c, a] / usage[c, a]))
        L.append(f'σ²-错误率相关 (c,a)对n={len(xs)}: pearson={pearson(xs,ys):.3f} '
                 f'spearman={spearman(xs,ys):.3f}  (强正=σ被用来回避难样本)')
    else:
        L.append('sigma=off, 跳过')

    L.append('[C_FUSION](P3/P8判读, GT类通道)')
    if inv_var_on:
        hs = wr_hist.numpy()
        hp = hs / max(hs.sum(), 1)
        mean_wr = float((np.arange(20) / 20 + 0.025) @ hp)
        L.append(f'w_r(信refine权重) mean={mean_wr:.3f} | 直方图20bin%={np.round(hp*100,1).tolist()}')
        for key in ('bo_ro', 'bo_rw', 'bw_ro', 'bw_rw'):
            cnt = cells[key]
            L.append(f'w_r|{key}={wr_cell_sum[key]/max(cnt,1):.3f}')
        L.append(f'(理想: bw_ro高、bo_rw低; 四值接近=门控未分化) base_var均值={bvar_sum/max(bvar_n,1):.3f}')
    else:
        L.append('fusion=gate(师兄式), w_r 不适用')

    L.append('[D_PI]')
    L.append(f'π有效分量数 mean={pi_effn_sum/max(pi_effn_n,1):.2f} / A={A} (≈A=还在均匀附近; ≈1=路由塌缩)')

    L.append('[E_EXTRA]')
    L.append(f'分量间方差(P2预览, GT类, sim/τ尺度) mean={btw_sum/max(btw_n,1):.3f}')
    if use_sigma and au_s:
        L.append(f'refine_var预测final错误 AUROC={auroc(np.concatenate(au_s), np.concatenate(au_y)):.3f} '
                 f'(0.5=无信息)')

    report = '\n'.join(L)
    print('\n' + report)
    out = args.out or os.path.join(
        os.path.dirname(os.path.abspath(args.checkpoint)),
        f'parseg4_analysis_{os.path.splitext(os.path.basename(args.checkpoint))[0]}.txt')
    with open(out, 'w') as f:
        f.write(report + '\n')
    print(f'\nsaved -> {out}')


if __name__ == '__main__':
    main()
