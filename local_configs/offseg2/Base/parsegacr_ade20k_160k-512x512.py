# PARSeg-ACR: auto-context re-decision. From scratch 160k, recipe-pure.
#
# The untried subsystem-level move: a SECOND decision round that sees the
# spatial arrangement of round-1's beliefs (which classes sit where, in
# what shapes, next to whom). Targets the diagnosed error mass directly:
# confident REGIONAL collapses among layout classes (ceiling<->wall etc.)
# are exactly the errors a layout-aware second look can flip; per-pixel
# one-shot decisions structurally cannot.
#
# Attribution discipline (owner requirement): training recipe is
# byte-identical to base (same losses/weights on round 1, same optimizer,
# schedule, crop, batch). The ONLY additions are structural: layout
# encoder + round-2 decision + bounded blend gate (init 0.1, max 1.0 --
# the model may interpolate all the way to "round 2 is the decision").
#
# Live monitoring: `acc_acr_gate` in the train log. Gate rising = the
# model trusts the second round; stuck near 0.1 = layout adds nothing.
# Read-out vs base try1 (48.17). Kill: 96k-128k clearly below try1 curve.
_base_ = ['./parseg3_ade20k_160k-512x512.py']

custom_imports = dict(
    imports=['mmseg.models.decode_heads.PARSegACR'],
    allow_failed_imports=False)

model = dict(
    decode_head=dict(
        type='PARSegACR',
        args=dict(
            # PARSeg3 args inherited unchanged by config deep-merge.
            acr_layout_dim=64,
            acr_gate_max=1.0,
            acr_gate_init=0.1,
            acr_r2w=1.0,
            acr_r1w=0.5,
        )))
