# PARSeg-LDR: language-described reranking. From scratch 160k.
#
# Text as EVIDENCE: the proven LCR core (candidate reranking, best local
# result 48.60) keeps everything and additionally sees, per candidate,
# three description-match scalars (max/mean/gap of cosine between the pixel
# feature and the candidate's 6 described attributes). Augment, never
# constrain: the learned class embedding stays; useless text evidence gets
# ignored by the MLP. Frozen desc asset only; no text model anywhere.
#
# Requires (once, server): python tools/gen_text_descriptions.py
# Read-out vs LCR v1 (48.60/48.48 band): isolates the text-evidence effect
# on the surviving mechanism. Kill: 96k-128k clearly below the LCR curve.
_base_ = ['./parseglcr_ade20k_160k-512x512.py']

custom_imports = dict(
    imports=['mmseg.models.decode_heads.PARSegLDR'],
    allow_failed_imports=False)

model = dict(
    decode_head=dict(
        type='PARSegLDR',
        args=dict(
            # all lcr_* args inherited unchanged (topk5/dim64/hidden128/
            # gate 0.35-0.05/aux 0.2/rank 0.2-0.2-2.0)
            ldr_desc_path='assets/text_anchors/ade20k_clip_vitb32_desc6.pt',
        )))
