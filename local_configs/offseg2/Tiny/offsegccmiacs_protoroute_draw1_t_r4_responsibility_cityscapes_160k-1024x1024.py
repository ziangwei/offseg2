# Fresh Proto-route run; submit.py assigns one unique seed before sbatch.
# seed=None is intentional: preflight refuses a run without the recorded seed.
_base_ = ['./offsegccmiacs_protoroute_t_r4_responsibility_cityscapes_160k-1024x1024.py']

randomness = dict(seed=None, deterministic=False)

load_from = None
resume = False
work_dir = './work_dirs/offsegccmiacs_protoroute_draw1_t_r4_responsibility_cityscapes_160k-1024x1024'
