# PARSeg-LRP final integrated candidate.
#
# The entire tuned PARSeg3 path and the validated LCR correction are inherited.
# LRP changes only PGAC's prototype source: frozen language descriptions form
# class retrieval queries, which obtain image-conditioned VISUAL prototypes
# without using base-logit masks.  Text is not a classifier and no image-
# specific text or text encoder is required at inference.
#
# Generate the frozen description asset once on the training machine:
#   python tools/gen_text_descriptions.py
_base_ = ['./parseglcr_ade20k_160k-512x512.py']

custom_imports = dict(
    imports=['mmseg.models.decode_heads.PARSegLRP'],
    allow_failed_imports=False)

model = dict(
    decode_head=dict(
        type='PARSegLRP',
        args=dict(
            # All PARSeg3 and LCR arguments are inherited unchanged.
            lrp_desc_path=(
                'assets/text_anchors/ade20k_clip_vitb32_desc6.pt'),
            lrp_num_heads=8,
            lrp_grid_size=16,
            lrp_center_text=True,

            # Begin close to LCR/PARSeg3, while leaving enough range for the
            # independent prototype to take over where base masks are wrong.
            lrp_gate_init=0.10,
            lrp_gate_max=0.50,
        )))

train_cfg = dict(val_interval=8000)
default_hooks = dict(
    checkpoint=dict(
        by_epoch=False,
        interval=8000,
        save_last=True,
        type='CheckpointHook'))
find_unused_parameters = True
