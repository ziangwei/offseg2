# Fresh Proto-route run; submit.py assigns one unique seed before sbatch.
# seed=None is intentional: preflight refuses a run without the recorded seed.
_base_ = ['./offsegccmiacs_protoroute_l_r4_responsibility_cityscapes_160k-1024x1024.py']

randomness = dict(seed=None, deterministic=False)

# Only the optimizer base LR changes: 6e-5 -> 4e-5; all multipliers remain.
optim_wrapper = dict(optimizer=dict(lr=4e-5))

load_from = None
resume = False
work_dir = './work_dirs/offsegccmiacs_protoroute_lr4_l_r4_responsibility_cityscapes_160k-1024x1024'
