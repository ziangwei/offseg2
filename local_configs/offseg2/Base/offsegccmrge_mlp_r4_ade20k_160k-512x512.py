# RGE refinement: masked-GAP -> four-channel shared MLP -> excitation.
# The MLP is shared by all classes and images.  Its final layer is zero-init,
# so the model starts exactly from the measured RGE-r4 computation while
# adding a conventional cross-channel excitation stage (only 76 parameters).
_base_ = ['./offsegccmrge_r4_ade20k_160k-512x512.py']

model = dict(
    decode_head=dict(
        rge_excitation_hidden=8,
    ))

work_dir = './work_dirs/offsegccmrge_mlp_r4_ade20k_160k-512x512'
