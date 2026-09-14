# Isolated Slurm experiment. B=S2, T=S1; inherited model recipe retained.
_base_ = ['./offsegccmiacs_protoroute_relation_r4_responsibility_ade20k_160k-512x512.py']
randomness = dict(seed=1370346084, deterministic=False)
load_from = None
resume = False
work_dir = './work_dirs/offsegccmiacs_protoroute_relation_b_r4_responsibility_ade20k_160k-512x512'
