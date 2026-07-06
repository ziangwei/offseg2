# PARSeg-PAT: description-grounded PAL attribute tokens. From scratch 160k.
#
# The last standing text hypothesis after the name-granularity axis died
# (LTA 46.95 / PTA 46.97, cone mechanism measured; LTC 48.48 = neutralized):
# attribute DESCRIPTIONS carry discriminative content that names lack, and
# PAL's attribute tokens are the thesis's own landing site for them.
# Training-only InfoNCE between present classes' calibrated attribute
# tokens and frozen per-class description sets (never averaged). Forward
# and inference are exactly PARSeg3; no text model in any graph.
#
# Asset (generate ONCE on the server before training):
#   HF_HOME=/path/to/cache python tools/gen_text_descriptions.py
#
# Read-out vs base try1 (48.17). Kill rule: 96k-128k clearly below try1's
# same-iter curve. Precedent for the bet: PALX's GT-anchoring of PAL
# geometry produced brief positive spikes; PAT is the language version with
# a stationary (frozen-text) target.
_base_ = ['./parseg3_ade20k_160k-512x512.py']

custom_imports = dict(
    imports=['mmseg.models.decode_heads.PARSegPAT'],
    allow_failed_imports=False)

model = dict(
    decode_head=dict(
        type='PARSegPAT',
        args=dict(
            # PARSeg3 args inherited unchanged by config deep-merge.
            pat_desc_path='assets/text_anchors/ade20k_clip_vitb32_desc6.pt',
            pat_w=0.15,
            pat_tau=0.1,
            pat_warmup_iters=8000,
            pat_token_dim=256,
        )))
