"""Produce PDF + PNG scientific figures; retain underlying CSV/NPZ data."""
import argparse
import csv
from pathlib import Path
import numpy as np
from PIL import Image
from .common import read_json, write_json


def render(run):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize, TwoSlopeNorm
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':9,
                         'axes.spines.top':False, 'axes.spines.right':False,
                         'pdf.fonttype':42, 'savefig.dpi':200})
    run = Path(run)
    out = run/'figures'
    out.mkdir(exist_ok=True)
    route = read_json(run/'route/evaluation.json')
    palette = np.array(route['palette']+[[0,0,0]]*(256-len(route['palette'])), dtype='uint8')
    cases = read_json(run/'comparison/cases.json')
    figures = []

    def save(fig, name, caption, **details):
        fig.savefig(out/(name+'.pdf'), bbox_inches='tight')
        fig.savefig(out/(name+'.png'), bbox_inches='tight')
        plt.close(fig)
        figures.append(dict(name=name, caption=caption, **details))

    def table(path):
        with path.open(encoding='utf-8-sig') as stream:
            return list(csv.DictReader(stream))

    rows = [r for r in table(run/'comparison/per_class_delta.csv') if r['delta'] != '']
    ordered = sorted(rows, key=lambda r:float(r['delta']))
    chosen = ordered[:10]+ordered[-10:]
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    ax.barh(range(len(chosen)), [float(r['delta']) for r in chosen],
            color=['#ce5260' if float(r['delta']) < 0 else '#208c83' for r in chosen])
    ax.set_yticks(range(len(chosen)), [r['name'] for r in chosen])
    ax.axvline(0, color='#555555', lw=.8)
    ax.set_xlabel('IoU change (percentage points)')
    save(fig, 'per_class_extremes', 'Ten largest decreases and ten largest increases, selected from all 150 ADE20K classes. Full table accompanies the figure.')
    fig, ax = plt.subplots(figsize=(9, 2.8))
    ax.bar(range(len(rows)), [float(r['delta']) for r in rows], width=1, color='#286c99')
    ax.axhline(0, color='#333333', lw=.6)
    ax.set(xlabel='ADE20K class id', ylabel='IoU change (pp)')
    save(fig, 'per_class_all', 'All 150 class differences in fixed dataset class order, including decreases.')

    pairs = table(run/'comparison/confusion_directions.csv')
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    y = np.arange(len(pairs))
    ax.barh(y-.18, [100*float(r['base_fraction']) for r in pairs], height=.34, label='OffSeg', color='#8b9aab')
    ax.barh(y+.18, [100*float(r['route_fraction']) for r in pairs], height=.34, label='Proto-route', color='#217cad')
    ax.set_yticks(y, [r['gt']+' → '+r['predicted'] for r in pairs])
    ax.set_xlabel('Fraction of GT-class pixels confused in this direction (%)')
    ax.legend(frameon=False)
    save(fig, 'confusion_directions', 'Ten directional confusions ranked by baseline error count only. Rates divide the GT-class pixel count, not the total image area.')

    # Four gain-ranked examples and four regression-ranked examples on separate pages.
    for page, selection in enumerate((cases[:4], cases[4:8]), 1):
        if not selection:
            continue
        fig, axes = plt.subplots(len(selection), 5, figsize=(12.5, 2.25*len(selection)), squeeze=False)
        for y, row in enumerate(selection):
            key = row['id']
            rgb = np.array(Image.open(run/'selected_images'/f'{key}.png'))
            gt = np.array(Image.open(run/'route/gt'/f'{key}.png'))
            base_pred = np.array(Image.open(run/'offseg/predictions'/f'{key}.png'))
            route_pred = np.array(Image.open(run/'route/predictions'/f'{key}.png'))
            valid = gt < 150
            corrected = valid & (base_pred != gt) & (route_pred == gt)
            damaged = valid & (base_pred == gt) & (route_pred != gt)
            overlay = (rgb.astype('float32')*.32).astype('uint8')
            overlay[corrected], overlay[damaged] = [33,181,130], [220,65,83]
            for ax, img in zip(axes[y], (rgb, palette[gt], palette[base_pred], palette[route_pred], overlay)):
                ax.imshow(img)
                ax.axis('off')
            axes[y,0].set_ylabel(key, fontsize=7)
        for ax, title in zip(axes[0], ('Image', 'Ground truth', 'OffSeg', 'Proto-route', 'Corrected / damaged')):
            ax.set_title(title)
        fig.subplots_adjust(wspace=.025, hspace=.08)
        save(fig, f'qualitative_{page}',
             ('Four largest pixel-accuracy improvements.' if page == 1 else 'Four largest pixel-accuracy regressions.')+
             ' Green: baseline wrong and Route correct; red: baseline correct and Route wrong. '
             'Selected cases are illustrative; use full-set tables for quantitative claims.', ids=[r['id'] for r in selection])

    # Equal weight for each image/crop/class observation, with absent crops separated.
    xs, ys, absent = [], [], []
    total_pairs = 0
    for path in sorted((run/'route/memory').glob('*.npz')):
        with np.load(path) as data:
            area = data['gt_area']/np.maximum(data['valid_pixels'][:,None], 1)
            lam = data['lam']
            present = area > 0
            # Deterministic bounded display sample; raw arrays retain every observation.
            xs.extend(area[present].tolist())
            ys.extend(lam[present].tolist())
            absent.extend(lam[~present][::max(1, int((~present).sum())//100)].tolist())
            total_pairs += area.size
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), gridspec_kw={'width_ratios':[3,1]})
    if xs:
        index = np.linspace(0, len(xs)-1, min(80000,len(xs)), dtype=int)
        axes[0].hexbin(np.log10(np.array(xs)[index]), np.array(ys)[index], gridsize=45, mincnt=1, cmap='Blues')
    axes[0].set(xlabel='log10(GT fraction within evaluated crop)', ylabel='Memory mixing weight λ', ylim=(0,1))
    if absent:
        axes[1].boxplot(absent, showfliers=False)
    axes[1].set(xticks=[1], xticklabels=['GT absent'], ylim=(0,1))
    save(fig, 'memory_vs_gt_area', 'GT area is measured after inference on the resized, padded actual crop. '
         'Present crop/class pairs are shown separately from absent ones. This is an association, not a causal gain. '
         'Absent box plot uses a deterministic bounded sample; full arrays are provided.', total_crop_class_pairs=total_pairs)

    # One actual crop for each of four successful and two unsuccessful cases.
    for row in cases[:4]+cases[4:6]:
        key = row['id']
        files = sorted((run/'mechanism/mechanisms'/key).glob('crop_*.npz'))
        if not files:
            continue
        geometry = read_json(files[0].parent/'geometry.json')
        rgb = Image.open(run/'selected_images'/f'{key}.png')
        h, w = geometry['img_shape']
        rgb = np.array(rgb.resize((w,h), Image.Resampling.BILINEAR))
        left, right, top, bottom = geometry['padding']
        rgb = np.pad(rgb, ((top,bottom),(left,right),(0,0)))
        with np.load(files[0]) as data:
            fh, fw = map(int, data['feature_shape'])
            y1,x1,y2,x2 = data['box']
            target = int(data['class_ids'][0])
            responsibility = data['responsibility'][0].reshape(fh,fw)*(fh*fw)
            spatial = data['spatial_weight'][0].reshape(fh,fw)*(fh*fw)
            vmax = max(float(responsibility.max()), float(spatial.max()), 1e-6)
            margin = data['delta_margin'][0].reshape(fh,fw)
            limit = max(float(np.abs(margin).max()), 1e-6)
            images = [data['rgb_input'] if 'rgb_input' in data else rgb[y1:y2,x1:x2], palette[data['gt_grid']], responsibility, spatial,
                      data['candidate_count'].reshape(fh,fw), data['delta'][0].reshape(fh,fw), margin]
            titles = ['Actual input crop', 'GT at feature grid', 'Responsibility × N',
                      'Spatial softmax × N', 'CCM candidate count', 'Raw correction Δ', 'Δ(target) − Δ(rival)']
            fig, axes = plt.subplots(2, 4, figsize=(12,5.5))
            for i,(img,title) in enumerate(zip(images,titles)):
                ax = axes.flat[i]
                kwargs = {}
                if i in (2,3):
                    kwargs = dict(cmap='magma', norm=Normalize(0,vmax))
                elif i in (4,5):
                    kwargs = dict(cmap='viridis')
                elif i == 6:
                    kwargs = dict(cmap='RdBu_r', norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit))
                artist = ax.imshow(img, **kwargs)
                ax.set_title(title, fontsize=9)
                ax.axis('off')
                if i >= 2:
                    fig.colorbar(artist, ax=ax, fraction=.046, pad=.03)
            ax = axes.flat[7]
            artist = ax.imshow(data['metric4x4'][0], cmap='RdBu_r')
            ax.set_title('Image-conditioned 4×4 metric')
            fig.colorbar(artist, ax=ax, fraction=.046, pad=.03)
            fig.suptitle(f'{key} | class: {route["classes"][target]} | crop 0: {tuple(data["box"])}', fontsize=10)
            fig.tight_layout()
            save(fig, 'mechanism_'+key, 'Actual first inference crop, not a stitched full-image attribution. '
                 'Responsibility and spatial softmax use the same trained Route logits and a common scale; '
                 'the spatial alternative is diagnostic, not a separately trained ablation. '
                 'Candidate count controls CCM context, not final label admissibility. '
                 'Δ and its rival margin are raw pre-normalization corrections, not probabilities or causal attributions. '
                 'The global learned four directions are fixed at inference; this 4×4 metric is recomputed per crop.',
                 class_id=target, crop=0)

    # Each run/log is plotted separately. Never join different attempts or seeds.
    for label in ('offseg', 'route'):
        metadata = read_json(run/'run.json')
        checkpoint = Path(metadata.get(label+'_original_checkpoint',metadata[label+'_checkpoint']))
        candidates = list((run/'handoff/runs').rglob('run.json'))
        matching = []
        for record in candidates:
            meta = read_json(record)
            if meta.get('work_dir') and checkpoint.parent == Path(meta['work_dir']):
                matching.append(record.parent)
            elif record.parent.name == checkpoint.parent.name:
                matching.append(record.parent)
        for number, folder in enumerate(matching):
            curves = table(folder/'validation_curve.csv')
            fig, ax = plt.subplots(figsize=(5.2,3))
            sources = sorted(set(r['source'] for r in curves))
            used = False
            for i, source in enumerate(sources):
                values = [r for r in curves if r['source'] == source and r['iteration'] not in ('','None')]
                if not values:
                    continue
                ax.plot([int(r['iteration'])/1000 for r in values], [float(r['miou']) for r in values], marker='.', label=f'log {i+1}')
                used = True
            if used:
                ax.set(xlabel='Training iteration (thousands)',ylabel='Validation mIoU (%)',title=label)
                ax.legend(frameon=False)
                save(fig, f'curve_{label}_{number}', 'Existing training validation readings, with individual log segments kept separate. Official downloaded weights may have no local training curve.')
            else:
                plt.close(fig)
    write_json(out/'captions.json', figures)
    print(f'Wrote {len(figures)} figure pairs to {out}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    render(parser.parse_args().run)
