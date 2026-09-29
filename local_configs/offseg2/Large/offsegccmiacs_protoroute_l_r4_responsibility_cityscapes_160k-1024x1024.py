# Original Proto-route, transferred to L/cityscapes; no new method variant.
# Keep this dataset's B Route training/evaluation recipe; change only scale.
_base_ = ['../Base/offsegccmiacs_protoroute_r4_responsibility_cityscapes_160k-1024x1024.py']

model = dict(
    backbone=dict(
        type='efficientformerv2_l_feat',
        init_cfg=dict(type='Pretrained',
                      checkpoint='pretrained/eformer_v2/eformer_l_450.pth')),
    decode_head=dict(
        in_channels=[40,80,192,384],
        new_channels=[32,64,128,256]))

load_from = None
resume = False
work_dir = './work_dirs/offsegccmiacs_protoroute_l_r4_responsibility_cityscapes_160k-1024x1024'
