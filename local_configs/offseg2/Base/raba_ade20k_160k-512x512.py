"""RABA on ADE20K: end-to-end, fixed 150-label output, no text/distill.

The data pipeline, EfficientFormerV2-S2 backbone, crop, batch size, 160k
schedule, validation interval, and slide-inference protocol are inherited from
the PARSeg3 reference.  The decoder is replaced in full to prevent PARSeg3's
language/refinement keys from leaking into this independent model family.
"""

_base_ = ['./parseg3_ade20k_160k-512x512.py']

custom_imports = dict(
    imports=[
        'mmdet.models',
        'mmseg.models.decode_heads.region_attribute_bialign_head'
    ],
    allow_failed_imports=False)

num_classes = 150
embed_dims = 256
num_feature_levels = 3

model = dict(
    decode_head=dict(
        _delete_=True,
        type='RegionAttributeBiAlignHead',
        in_channels=[32, 64, 144, 288],
        feat_channels=embed_dims,
        out_channels=embed_dims,
        num_classes=num_classes,
        num_queries=100,
        num_transformer_feat_level=num_feature_levels,
        align_corners=False,
        pixel_decoder=dict(
            type='mmdet.P3FreqFusionPixelDecoder',
            new_channels=[32, 64, 128, 256],
            num_outs=num_feature_levels,
            norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
            act_cfg=dict(type='ReLU', inplace=True),
            # MMDetection 3.x checks this value before it constructs any
            # pixel decoder.  RABA accepts and ignores the remaining encoder
            # config: it uses P3 FreqFusion, not deformable attention.
            encoder=dict(
                layer_cfg=dict(
                    self_attn_cfg=dict(num_levels=num_feature_levels)))),
        enforce_decoder_input_project=False,
        positional_encoding=dict(num_feats=128, normalize=True),
        transformer_decoder=dict(
            return_intermediate=True,
            num_layers=3,
            layer_cfg=dict(
                self_attn_cfg=dict(
                    embed_dims=embed_dims,
                    num_heads=8,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=True),
                cross_attn_cfg=dict(
                    embed_dims=embed_dims,
                    num_heads=8,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=True),
                ffn_cfg=dict(
                    embed_dims=embed_dims,
                    feedforward_channels=1024,
                    num_fcs=2,
                    act_cfg=dict(type='ReLU', inplace=True),
                    ffn_drop=0.0,
                    dropout_layer=None,
                    add_identity=True)),
            init_cfg=None),
        num_attributes=4,
        attribute_temperature=0.20,
        alignment_temperature=0.20,
        classification_temperature=0.07,
        attribute_residual_scale=0.50,
        offset_scale=0.10,
        mask_pool_eps=1e-6,
        final_only_loss=True,
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=2.0,
            reduction='mean',
            class_weight=[1.0] * num_classes + [0.1]),
        loss_mask=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=5.0),
        loss_dice=dict(
            type='mmdet.DiceLoss',
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            eps=1.0,
            loss_weight=5.0),
        train_cfg=dict(
            num_points=12544,
            oversample_ratio=3.0,
            importance_sample_ratio=0.75,
            assigner=dict(
                type='mmdet.HungarianAssigner',
                match_costs=[
                    dict(type='mmdet.ClassificationCost', weight=2.0),
                    dict(
                        type='mmdet.CrossEntropyLossCost',
                        weight=5.0,
                        use_sigmoid=True),
                    dict(
                        type='mmdet.DiceCost',
                        weight=5.0,
                        pred_act=True,
                        eps=1.0)
                ]),
            sampler=dict(type='mmdet.MaskPseudoSampler'))),
    # Keep PARSeg3's 512 crop / 480 stride comparison protocol.
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(480, 480)))

# A transformer decoder trained from scratch is unstable under PARSeg3's
# inherited "head x10" rule (effective 6e-4).  Use the standard Mask2Former
# scale while retaining the same data, batch, iteration count, and scheduler.
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.0001,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.05),
    clip_grad=dict(max_norm=0.01, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1, decay_mult=1.0),
            'query_embed': dict(decay_mult=0.0),
            'query_feat': dict(decay_mult=0.0),
            'level_embed': dict(decay_mult=0.0),
            'class_centers': dict(decay_mult=0.0),
            'attribute_deltas': dict(decay_mult=0.0),
            'no_object_repr': dict(decay_mult=0.0),
        },
        norm_decay_mult=0.0))

find_unused_parameters = False
