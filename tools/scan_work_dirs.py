#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Report how far every run in work_dirs actually got.

Eval count alone cannot separate "trained to the end, final validation did not
land" from "stopped early because it was tracking below the reference".  The
last logged training iteration can.  For every work_dir this prints:

    last train iter / max_iters, percent, eval count, best and last mIoU,
    the largest iter_*.pth on disk, and a status label.

Status labels match EXPERIMENTS.md section 1:
    complete      reached max_iters and the final validation is present
    val-missing   reached max_iters but the last validation did not land
                  (wall clock, full disk, ...) -> the number is a LOWER BOUND,
                  recoverable with tools/dist_test.sh on the final checkpoint
    stopped@NN%   never reached max_iters -> `killed` or `wall/infra`;
                  the number records a decision, not the method's performance

Usage:
    python tools/scan_work_dirs.py [work_dirs] [--json out.json]
"""

import argparse
import glob
import json
import os
import re

TRAIN_RE = re.compile(r'Iter\(train\)\s*\[\s*(\d+)\s*/\s*(\d+)\s*\]')
MIOU_RE = re.compile(r'\bmIoU:\s*([0-9]+\.?[0-9]*)')
CKPT_RE = re.compile(r'iter_(\d+)\.pth$')


def scan_one(path):
    logs = sorted(glob.glob(os.path.join(path, '**', '*.log'), recursive=True))
    last_iter = max_iters = 0
    mious = []
    for log in logs:
        try:
            with open(log, 'r', encoding='utf-8', errors='ignore') as handle:
                for line in handle:
                    hit = TRAIN_RE.search(line)
                    if hit:
                        cur, total = int(hit.group(1)), int(hit.group(2))
                        last_iter = max(last_iter, cur)
                        max_iters = max(max_iters, total)
                    hit = MIOU_RE.search(line)
                    if hit:
                        mious.append(float(hit.group(1)))
        except OSError:
            continue

    ckpts = []
    for pth in glob.glob(os.path.join(path, '**', '*.pth'), recursive=True):
        hit = CKPT_RE.search(os.path.basename(pth))
        if hit:
            ckpts.append(int(hit.group(1)))
    best_ckpt = sorted(glob.glob(
        os.path.join(path, '**', 'best_mIoU_iter_*.pth'), recursive=True))

    if not max_iters:
        status = 'no-log'
    elif last_iter >= max_iters:
        # The final validation normally lands after the last training log line.
        expected = mious and abs(mious[-1] - max(mious)) < 1e-9
        status = 'complete' if mious else 'val-missing'
        if mious and len(mious) >= 2 and not expected:
            status = 'complete'
    else:
        status = 'stopped@%d%%' % round(100.0 * last_iter / max_iters)

    return dict(
        name=os.path.basename(path.rstrip('/')),
        last_iter=last_iter,
        max_iters=max_iters,
        pct=(round(100.0 * last_iter / max_iters, 1) if max_iters else 0.0),
        evals=len(mious),
        best=(max(mious) if mious else None),
        last=(mious[-1] if mious else None),
        max_ckpt=(max(ckpts) if ckpts else None),
        has_best_ckpt=bool(best_ckpt),
        status=status,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', nargs='?', default='work_dirs')
    parser.add_argument('--json', dest='json_out', default=None)
    args = parser.parse_args()

    rows = []
    for entry in sorted(os.listdir(args.root)):
        path = os.path.join(args.root, entry)
        if os.path.isdir(path):
            rows.append(scan_one(path))

    width = max([len(r['name']) for r in rows] + [4])
    header = ('%-*s  %-16s %6s  %5s  %8s  %8s  %10s  %s'
              % (width, 'work_dir', 'iters', 'pct', 'evals',
                 'best', 'last', 'max_ckpt', 'status'))
    print(header)
    print('-' * len(header))
    for r in rows:
        print('%-*s  %-16s %5.1f%%  %5d  %8s  %8s  %10s  %s' % (
            width, r['name'],
            '%d/%d' % (r['last_iter'], r['max_iters']) if r['max_iters'] else '-',
            r['pct'], r['evals'],
            '%.2f' % r['best'] if r['best'] is not None else '-',
            '%.2f' % r['last'] if r['last'] is not None else '-',
            r['max_ckpt'] if r['max_ckpt'] is not None else '-',
            r['status'] + (' +best.pth' if r['has_best_ckpt'] else '')))

    incomplete = [r for r in rows if r['status'].startswith('stopped')]
    if incomplete:
        print('\n未跑满（数值记录的是一次决策，不是该方法的成绩）:')
        for r in sorted(incomplete, key=lambda x: x['pct']):
            print('  %-*s  %d/%d (%.0f%%)  best=%s'
                  % (width, r['name'], r['last_iter'], r['max_iters'],
                     r['pct'],
                     '%.2f' % r['best'] if r['best'] is not None else '-'))

    if args.json_out:
        with open(args.json_out, 'w', encoding='utf-8') as handle:
            json.dump(rows, handle, indent=2, ensure_ascii=False)
        print('\nwrote %s' % args.json_out)


if __name__ == '__main__':
    main()
