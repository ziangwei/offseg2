_base_ = ['../../offseg/Tiny/offseg-t_cityscapes_160k-1024x1024.py']

model = dict(
    decode_head=dict(
        type='PARSeg3',
        cls_attributes=8,
        args=dict(
            basew=2.0,
            refinementw=1.5,
            fusionw=1.0,
            intra_div=0.1,
            tau=0.07,
            proto_topk_div=256,
            refinement_focusw=0.75,
            proto_residual_scale=1.0,
            fusion_mode='AGCF',
            use_class_prototypes=True,
        )))

# 4 GPUs x 2 images/GPU for Cityscapes.
train_dataloader = dict(batch_size=2)
