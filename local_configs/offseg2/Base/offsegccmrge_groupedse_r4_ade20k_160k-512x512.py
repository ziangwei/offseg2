# Grouped-SE RGE: every class owns a tiny 4->8->4 excitation MLP because its
# four learned residual response channels form a class-specific coordinate
# system.  The final grouped 1x1 layer is zero-init, so this starts from RGE.
_base_ = ['./offsegccmrge_r4_ade20k_160k-512x512.py']

model = dict(
    decode_head=dict(
        rge_excitation_hidden=8,
        rge_excitation_classwise=True,
        rge_response_hidden=0,
    ))

work_dir = './work_dirs/offsegccmrge_groupedse_r4_ade20k_160k-512x512'
