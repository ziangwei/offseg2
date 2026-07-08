# PARSeg-TDL: text dictionary lookup. From scratch 160k.
#
# Text as CONTENT: refinement features retrieve from a frozen 900-entry
# attribute-description dictionary via single-head cross-attention, added
# back through a bounded scalar gate (init 0.05, max 0.5). DTFormer/TSAM
# lineage made train/inference-symmetric with an offline constant bank --
# no text model in any graph, no per-image text, no leakage surface. Base
# head untouched; enrichment on the PAL side only.
#
# Requires (once, server): python tools/gen_text_descriptions.py
# Read-out vs base try1 (48.17). Watch the learned gate in the ckpt
# (prototype_attribute_refinement.tdl_lookup.gate_alpha): gate -> 0 means
# the model rejected the dictionary. Kill: 96k-128k clearly below try1.
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
        )))
