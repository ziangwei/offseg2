# PARSeg5-CPM: PARSeg3 + cross-image prototype memory + evidence-aware fusion.
# Independent evidence comes from a dataset-level GT prototype bank (EMA), not
# from the same-image features that base/refine/EAF-context all share. Injected
# at the fusion end exactly like EAF, so EAF vs CPM isolates the evidence source.
# Inherits all run settings from the parseg3 config (batch_size=4, 4-card,
# val/ckpt interval 8000, cudnn_benchmark, optimizer, schedule) via _base_;
# the args dict below is deep-merged into parseg3's args, so basew/refinementw/
# fusionw/intra_div/tau/proto_*/use_class_prototypes are all preserved.
_base_ = ['./parseg3_ade20k_160k-512x512.py']

custom_imports = dict(
    imports=['mmseg.models.decode_heads.PARSeg5CPM'],
    allow_failed_imports=False)

model = dict(
    decode_head=dict(
        type='PARSeg5CPM',
        args=dict(
            cpm_emb_dim=256,
            cpm_tau=0.1,
            cpm_momentum=0.999,
            cpm_update_min_count=1,
            globalw=0.4,
            global_focusw=0.2,
            global_focus_err_weight=1.0,
            global_focus_unc_weight=0.5,
            global_focus_class_balance=True,
            refinement_focusw=0.75,
        )))
