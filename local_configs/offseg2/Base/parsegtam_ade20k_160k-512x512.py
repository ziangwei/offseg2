# PARSeg-TAM: text-derived per-class attribute metric. From scratch 160k.
#
# Text as METRIC: the refinement cosine gets a per-class diagonal channel
# weighting w_c = 1 + 0.5*tanh(W(desc_mean_c) + r_c), W and r zero-init so
# the model starts EXACTLY as PARSeg3. Language seeds which channels matter
# for each class's similarity; bounded positive weights carry no cross-class
# geometry constraint (the name-cone failure cannot apply). Zero new losses;
# inference adds one [C, D] multiply.
#
# Requires (once, server): python tools/gen_text_descriptions.py
# Read-out vs base try1 (48.17). Cheap forensic after training: the spread
# of w_c across classes (all w stuck near 1 = metric rejected). Kill:
# 96k-128k clearly below try1 curve.
_base_ = ['./parseg3_ade20k_160k-512x512.py']

custom_imports = dict(
    imports=['mmseg.models.decode_heads.PARSegTAM'],
    allow_failed_imports=False)

model = dict(
    decode_head=dict(
        type='PARSegTAM',
        args=dict(
            # PARSeg3 args inherited unchanged by config deep-merge.
            tam_desc_path='assets/text_anchors/ade20k_clip_vitb32_desc6.pt',
            tam_scale=0.5,
        )))
