# Strong readable decoder: object-context feedback refines pixels, then the
# four-map RGE scorer handles class-relative response diversity.  Still no
# CCM, full IACS matrix, second prediction branch, or extra loss.
_base_ = ['./offsegrge_r4_noccm_ade20k_160k-512x512.py']

model = dict(
    decode_head=dict(
        type='OffSegOCFRGE',
        context_hidden=128,
    ))

work_dir = './work_dirs/offsegocfrge_r4_ade20k_160k-512x512'
