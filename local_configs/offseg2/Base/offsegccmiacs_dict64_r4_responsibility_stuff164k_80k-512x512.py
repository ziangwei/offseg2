# Shared residual-direction dictionary on COCO-Stuff164K-B.
#
# This is where the hypothesis actually lives.  The per-class basis U_k is a
# private [C, r] block, 153,600 parameters at 150 classes and 171 at Stuff,
# and every class learns its four directions from its own pixels alone.  The
# measured Stuff paired gains are +0.42 at T but only +0.07 at B, and Stuff
# runs 171 classes on half the ADE schedule -- exactly the regime where
# private per-class blocks get the least data each.  Replacing them with a
# rank-r combination of m=64 shared atoms lets every class's directions be
# estimated from all classes' gradients.
#
# Single variable vs Base/offsegccmiacs_r4_responsibility_stuff164k_80k:
# the basis parameterisation.  Projection, per-image second moment,
# competitive responsibility, stop-gradient, scale and mix are inherited.
#
# Read this together with the ADE run of the same head
# (offsegccmiacs_dict64_r4_responsibility_ade20k_160k-512x512.py), which is
# the control at 150 classes and a full schedule:
#   Stuff-B up, ADE flat   the hypothesis holds -- private bases are
#                          undertrained where classes are many and the
#                          schedule short.  Fixes the weakest cell in the
#                          results table and comes with a mechanism story.
#   both up                sharing is simply the better parameterisation;
#                          adopt everywhere at 2.8x fewer basis parameters.
#   both flat              iso-performance at fewer parameters; report as an
#                          efficiency note, not a gain.
#   both down              class subspaces genuinely need private directions;
#                          close the sharing axis.
# Live needle `acc_dict_atom_usage`: participation ratio per class.  Near 64
# means classes really share the dictionary; near 4 means each class collapsed
# onto its own atoms and a null result says nothing about sharing.
_base_ = ['./offsegccmiacs_r4_responsibility_stuff164k_80k-512x512.py']

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
    './work_dirs/offsegccmiacs_dict64_r4_responsibility_b_stuff164k_80k-512x512'
