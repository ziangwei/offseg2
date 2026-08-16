# Independent second draw of the 47.79 main model.
#
# The original run did not set `randomness`, so MMEngine drew its seed and it
# was never recorded.  Fixing an explicit seed here makes this an independent
# draw, not a replay of the same trajectory -- which is what a variance
# estimate needs.  Nothing else changes.
#
# Read-out: the pair (47.79, this) is the project's first spread estimate for
# the main model.  A result >= 48.0 also clears the 48 line on the same
# mechanism, without a new component.
_base_ = ['./offsegccmiacs_r4_responsibility_ade20k_160k-512x512.py']

randomness = dict(seed=2026, deterministic=False)

work_dir = \
    './work_dirs/offsegccmiacs_r4_responsibility_seed2026_ade20k_160k-512x512'
