# PARSeg-SDR SMOKE TEST ONLY -- 2k iters, ~20-30 min on 4 GPUs.
#
# Purpose: verify on the server that the SDR head builds, all five new loss
# terms appear in the log with sane finite magnitudes, validation runs, and
# nothing NaNs. Warm-starts from the PARSeg3 try1 checkpoint ONLY because
# that makes the val number interpretable as a crash check.
#
# ⚠ The mIoU printed here is NOT a gain signal for SDR (warm-start screens
# are structurally blind to training-time mechanisms -- that is the whole
# point of SDR). Expected val: anywhere in the 46-49 band. Only investigate
# if it is far outside that band or NaN.
_base_ = ['./parsegsdr_ade20k_160k-512x512.py']

load_from = 'work_dirs/parseg3_ade20k_160k-512x512_4x4_try1/iter_160000.pth'

train_cfg = dict(max_iters=2000, val_interval=1000)
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=1000, type='CheckpointHook'))

model = dict(
    decode_head=dict(
        args=dict(
            # shorter ramp so the smoke run actually exercises the kd/margin
            # losses at non-trivial weight before it ends
            sdr_warmup_iters=500,
        )))
