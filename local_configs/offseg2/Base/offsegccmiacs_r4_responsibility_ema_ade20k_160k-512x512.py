# The 47.79 model with an exponential moving average of the weights.
#
# The only axis this project has never touched: the training recipe.  Every
# one of the ~22 ADE variants changed the architecture; none changed how the
# weights are produced.  Weight EMA is standard, costs nothing at inference
# (the averaged weights replace the raw ones), adds no parameters, no loss and
# no branch, and does not alter the backbone, crop or schedule.  Under this
# project's reporting convention -- best across evals -- a free +0.2 is a free
# +0.2.
#
# It is a training aid, not a contribution, and must be reported as such: if
# it helps, every number in the ablation table has to be regenerated with it
# or none of them can use it.  That is the real cost of this run, and the
# reason to test it on the main model first rather than adopting it silently.
#
# momentum 0.0002 with interval 1 gives an effective averaging window of about
# 5000 iterations against the 160k schedule; the poly LR reaches zero at the
# end, so late-training weights are already slow-moving and the window mainly
# suppresses residual batch-to-batch jitter.
#
# Read-out:
#   >= 48.0     free gain that clears the 48 line without any new mechanism.
#               Then it must be applied uniformly across the ablation table.
#   47.8-48.0   small positive; decide whether regenerating the whole table is
#               worth it.
#   <  47.8     no effect at this schedule; report as a one-line negative and
#               never revisit the training-recipe axis.
#
# Compute: 4 GPUs, about 24 h.  Identical to the main model in every other
# respect, so this is a strict single variable.
_base_ = ['./offsegccmiacs_r4_responsibility_ade20k_160k-512x512.py']

custom_hooks = [
    dict(type='EMAHook', momentum=2e-4, interval=1, priority=49),
]

work_dir = \
    './work_dirs/offsegccmiacs_r4_responsibility_ema_ade20k_160k-512x512'
