# Control: is CCM necessary?
#
# OffSeg + IACS-r4 + competitive responsibility, with CCM's context-conditioned
# feature transform replaced by the identity.  Single variable against the
# 47.79 main model: rank, statistics mode, assignment, stage-1 CE weight,
# optimiser keys and run settings are all inherited unchanged.
#
# Pre-registered read-out:
#   >= 47.6      CCM contributes nothing measurable.  Delete it: the method
#                loses ~0.11M parameters and its least explicable component.
#   47.0 - 47.6  CCM contributes but is not load bearing.  Report the number
#                and narrow the claim instead of calling CCM essential.
#   <  47.0      CCM is load bearing.  The chain keeps it and the ablation
#                table can finally say so with evidence.
_base_ = ['./offsegccmiacs_r4_responsibility_ade20k_160k-512x512.py']

custom_imports = dict(
    imports=[
        'mmseg.models.decode_heads.OffSegACS',
        'mmseg.models.decode_heads.OffSegNoCCM',
    ],
    allow_failed_imports=False)

model = dict(decode_head=dict(type='OffSegIACSNoCCM'))

work_dir = './work_dirs/offsegiacs_r4_responsibility_noccm_ade20k_160k-512x512'
