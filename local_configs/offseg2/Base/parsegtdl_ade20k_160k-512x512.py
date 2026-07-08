# PARSeg-TDL: text dictionary lookup. From scratch 160k.
#
# Text as CONTENT: refinement features retrieve from a frozen 900-entry
# attribute-description dictionary via single-head cross-attention, added
# back through a bounded scalar gate (init 0.05, max 0.5). DTFormer/TSAM
# lineage made train/inference-symmetric with an offline constant bank --
# no text model in any graph, no per-image text, no leakage surface. Base
# head untouched; enrichment on the PAL side only.
#
# Retrieval is explicitly ANCHORED (v2, pre-launch fix of the "supervision
# too indirect -> degenerates into a generic feature bias" failure mode):
# a training-only GT-routed alignment loss pushes each pixel's attention
# mass onto its GT class's own K dictionary entries (weight 0.1, ramped),
# so the dictionary must be used semantically, not as a random basis.
#
# Live monitoring: `acc_tdl_gt_mass` in the train log = mean attention mass
# on the GT class's entries. Uniform baseline ~ 1/150 = 0.0067; it rising
# toward 0.1+ means semantic retrieval is forming.
#
# Requires (once, server): python tools/gen_text_descriptions.py
# Read-out vs base try1 (48.17). Post-training forensics: learned gate
# (tdl_lookup.gate_alpha; gate -> 0 = dictionary rejected). If TDL wins,
# the decisive text-content ablation is one line: tdl_random_bank=True
# (same-shape fixed random bank). Kill: 96k-128k clearly below try1.
_base_ = ['./parseg3_ade20k_160k-512x512.py']

custom_imports = dict(
    imports=['mmseg.models.decode_heads.PARSegTDL'],
    allow_failed_imports=False)

model = dict(
    decode_head=dict(
        type='PARSegTDL',
        args=dict(
            # PARSeg3 args inherited unchanged by config deep-merge.
            tdl_desc_path='assets/text_anchors/ade20k_clip_vitb32_desc6.pt',
            tdl_attn_dim=64,
            tdl_gate_max=0.5,
            tdl_gate_init=0.05,
            tdl_alignw=0.1,
            tdl_warmup_iters=8000,
            tdl_random_bank=False,
        )))
