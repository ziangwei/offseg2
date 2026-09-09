# Independent capacity arm against ProtoRoute r4 48.49; no classmix stacking.
# The old non-memory IACS-r8 lost 0.65: this is a conditional re-test after
# adding responsibility, prototype memory and memory-based CCM routing.
# Change only rank. Initial correction energy is NOT held constant.
_base_ = ['./offsegccmiacs_protoroute_r4_responsibility_ade20k_160k-512x512.py']

model = dict(decode_head=dict(acs_rank=8))
load_from = None
resume = False
work_dir = './work_dirs/offsegccmiacs_protoroute_r8_responsibility_ade20k_160k-512x512'
