# Shared residual-direction dictionary under the 47.79 decision side.
#
# Single variable vs offsegccmiacs_r4_responsibility: the per-class basis
# U_k is no longer a private [C, r] block but a rank-r combination of m=64
# shared atoms, U_k = orthonormalise(D^T A_k).  Projection, per-image second
# moment, competitive responsibility, stop-gradient, scale and mix are all
# inherited unchanged; no loss is added.
#
# Parameters in the basis: 150*256*4 = 153,600  ->  64*256 + 150*64*4 = 54,784
# (about 2.8x fewer, and the module drops from roughly 0.26M to 0.16M).
#
# Two measurements motivate it:
#   * RGE + class-wise grouped SE = 46.49, i.e. -1.07 against the shared
#     version -- per-class freedom overfits in this decoder, and the basis is
#     the largest purely per-class block left;
#   * the Stuff paired gain is +0.42 at T but +0.07 at B, and Stuff has 171
#     classes at half the ADE schedule, which is exactly where private
#     per-class blocks get the least data each.
#
# Pre-registered read-out:
#   >= 47.79    sharing wins outright: better number AND 2.8x fewer basis
#               parameters.  Becomes the main model, and the Stuff-B rerun is
#               the immediate follow-up.
#   47.4-47.79  iso-performance at 2.8x fewer parameters -- still a good trade
#               for an efficiency thesis; then try dict_size=32 (5.6x).
#   <  47.4     the class subspaces genuinely need private directions; report
#               and close the sharing axis.
# Live needle `acc_dict_atom_usage`: participation ratio per class.  Near 64
# means classes spread over the whole dictionary; near 4 means each class
# collapsed onto its own atoms and the sharing is nominal only -- in that case
# a null result says nothing about sharing.
_base_ = ['./offsegccmiacs_r4_responsibility_ade20k_160k-512x512.py']

custom_imports = dict(
    imports=[
        'mmseg.models.decode_heads.OffSegACS',
        'mmseg.models.decode_heads.OffSegDictBasis',
    ],
    allow_failed_imports=False)

model = dict(decode_head=dict(type='OffSegCCMIACSDict', dict_size=64))

optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys={
            'acs.mix_logit': dict(lr_mult=10.0, decay_mult=0.0),
        }))

work_dir = \
    './work_dirs/offsegccmiacs_dict64_r4_responsibility_ade20k_160k-512x512'
