# PARSeg4.3 = 4.2a-lite + 异源上下文精修。动机 / 做法见 mmseg/models/decode_heads/PARSeg43.py。
# 继承 4.2a-lite 的全部设置(refinement_focusw=0.75 / batch_size=8 / 8000 验证+存档 / cudnn /
# loadbal_w=0 / mix_temp_start=1.0), 只把精修头换成 PARSeg43 并加 ctx_dilations。
# 不 override forward / 不改输出 dict → analyze_parseg4.py 原样适用(不动脚本)。
# 退回值: ctx_gate 学到 0 ≈ 4.2a。
_base_ = ['./parseg42a-lite_ade20k_160k-512x512.py']

custom_imports = dict(
    imports=['mmseg.models.decode_heads.PARSeg41',
             'mmseg.models.decode_heads.PARSeg43'],
    allow_failed_imports=False)

model = dict(
    decode_head=dict(
        type='PARSeg43',
        args=dict(ctx_dilations=(1, 6, 12))))
