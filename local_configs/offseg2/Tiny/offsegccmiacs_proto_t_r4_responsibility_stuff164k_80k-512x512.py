# Isolated Slurm experiment. B=S2, T=S1; inherited model recipe retained.
_base_ = ['./offsegccmiacs_protoroute_r4_responsibility_stuff164k_80k-512x512.py']
model = dict(decode_head=dict(type='OffSegCCMIACSProto'))
randomness = dict(seed=2000199364, deterministic=False)
load_from = None
resume = False
work_dir = './work_dirs/offsegccmiacs_proto_t_r4_responsibility_stuff164k_80k-512x512'
