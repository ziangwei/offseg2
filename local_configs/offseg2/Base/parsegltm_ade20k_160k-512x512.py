# PARSeg-LTM: LCR x TAM stack. From scratch 160k. THE 49+ push.
#
# The project's two positives composed: LCR candidate rerank (48.60/48.48,
# base-logits subsystem) + TAM language metric (48.73, best number,
# refinement-cosine subsystem). Different subsystems, no shared params,
# both zero-init/identity-start with their exact original hyperparameters.
# Additive case lands ~49.1-49.3.
#
# Requires desc6 asset (already on server). Read-out vs TAM (48.73) and
# LCR (48.60): must beat BOTH to validate the stack. Kill: 96k-128k
# clearly below the TAM/LCR same-iter curves.
_base_ = ['./parseglcr_ade20k_160k-512x512.py']

custom_imports = dict(
    imports=['mmseg.models.decode_heads.PARSegLTM'],
    allow_failed_imports=False)

model = dict(
    decode_head=dict(
        type='PARSegLTM',
        args=dict(
            # all lcr_* args inherited unchanged from the LCR config
            tam_desc_path='assets/text_anchors/ade20k_clip_vitb32_desc6.pt',
            tam_scale=0.5,
            tam_use_text=True,
        )))
