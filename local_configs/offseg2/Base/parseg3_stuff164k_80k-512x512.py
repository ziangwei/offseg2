_base_ = ['../../offseg/Base/offseg-b_stuff164k_80k-512x512.py']

coco_data_root = '/dss/dssfs05/pn39qo/pn39qo-dss-0001/di97fer/xjn/coco_stuff164k'

model = dict(
    decode_head=dict(
        type='PARSeg3',
        cls_attributes=12,
        args=dict(
            basew=2.0,
            refinementw=1.5,
            fusionw=1.0,
            intra_div=0.1,
            tau=0.07,
            proto_topk_div=64,
            refinement_focusw=0.75,
            proto_residual_scale=1.0,
            fusion_mode='AGCF',
            use_class_prototypes=True,
        )))

# 4 GPUs x 4 images/GPU for COCO-Stuff.
train_dataloader = dict(
    batch_size=4, dataset=dict(data_root=coco_data_root))
val_dataloader = dict(dataset=dict(data_root=coco_data_root))
test_dataloader = dict(dataset=dict(data_root=coco_data_root))
