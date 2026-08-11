# Response-FFN RGE: a shared residual 4->8->4 pointwise channel mixer acts on
# the live residual response maps before responsibility gather/excitation.
# Its final layer is zero-init, so the initial computation equals RGE-r4.
_base_ = ['./offsegccmrge_r4_ade20k_160k-512x512.py']

model = dict(
    decode_head=dict(
        rge_excitation_hidden=0,
        rge_excitation_classwise=False,
        rge_response_hidden=8,
    ))

work_dir = './work_dirs/offsegccmrge_responseffn_r4_ade20k_160k-512x512'
