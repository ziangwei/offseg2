# One independent arm against ProtoRoute 48.49.
# Replace the shared IACS mix scalar with one learned scalar per class.
# All start at the original 0.10. Keep all candidates and full IACS.
# Memory n0=200 / EMA=.01, routing and both CE losses stay unchanged.
# Fresh 160k run from the inherited backbone pretraining, not a route resume.
_base_ = ['./offsegccmiacs_protoroute_r4_responsibility_ade20k_160k-512x512.py']

model = dict(decode_head=dict(iacs_classwise_mix=True))
load_from = None
resume = False
work_dir = './work_dirs/offsegccmiacs_protoroute_classmix_r4_responsibility_ade20k_160k-512x512'
