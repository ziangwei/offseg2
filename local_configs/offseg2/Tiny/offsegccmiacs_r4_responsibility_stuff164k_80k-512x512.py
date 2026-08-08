# OffSeg-T responsibility-IACS-r4 on COCO-Stuff164K (171 classes).
# The parent already fixes the Stuff protocol to 80k iterations with
# validation/checkpoints every 4000 and total batch size 16.
_base_ = ['./offsegccmiacs_r4_stuff164k_80k-512x512.py']

# Tiny and Base configs share a basename, so their default MMEngine work_dir
# would collide without an explicit scale-specific directory.
work_dir = './work_dirs/offsegccmiacs_r4_responsibility_t_stuff164k_80k-512x512'

model = dict(
    decode_head=dict(
        iacs_center_statistics=False,
        iacs_assignment='posterior',
        iacs_reliability_shrink=False,
        iacs_persistent_spectrum=False,
        iacs_classwise_mix=False,
        iacs_candidate_topk=0,
    ))
