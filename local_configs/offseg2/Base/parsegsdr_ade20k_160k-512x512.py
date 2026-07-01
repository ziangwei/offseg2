# PARSeg-SDR: scene-discriminative refinement with a GT-anchored teacher.
#
# FROM-SCRATCH 160k run. This is deliberately NOT a warm-start FT probe:
# SDR changes how the features and the decision surface are TRAINED
# (GT-anchored teacher prototypes + self-distillation + in-scene margins),
# and a warm-start around an already-converged 48.2 checkpoint structurally
# cannot show that class of effect (see PARSeg_experiment_summary.log --
# every warm-start add-on this year was flat, spiked +0.1, or collapsed).
#
# Inference graph and parameter set are EXACTLY PARSeg3: zero new params,
# zero inference cost. The teacher branch exists only during training.
#
# Pre-registered read-out (compare against try1's mid-run validation curve
# from work_dirs/parseg3_ade20k_160k-512x512_4x4_try1):
#   - 32k: must not be below try1@32k        -> otherwise kill the run.
#   - 64k: must be >= try1@64k + 0.3         -> otherwise kill the run.
#   - 160k: report final vs 48.17/48.2.
_base_ = ['./parseg3_ade20k_160k-512x512.py']

custom_imports = dict(
    imports=['mmseg.models.decode_heads.PARSegSDR'],
    allow_failed_imports=False)

model = dict(
    decode_head=dict(
        type='PARSegSDR',
        args=dict(
            # PARSeg3 args (basew/refinementw/fusionw/intra_div/tau/...)
            # are inherited unchanged by config deep-merge.
            sdr_teacherw=1.0,     # CE on GT-anchored teacher refinement
            sdr_kdw=0.5,          # student->teacher self-distillation
            sdr_kd_temp=1.0,
            sdr_rivalw=0.2,       # in-scene co-present rival margin (final logits)
            sdr_absentw=0.1,      # absent-class suppression (final logits)
            sdr_margin=0.5,
            sdr_purity=0.75,      # margins only on clean interior 4x4 cells
            sdr_warmup_iters=8000,  # linear 0->1 ramp for kd/margin losses
        )))

# fixed seed so later SDR ablations (loss on/off) can be run as
# matched-seed A/B against this run
randomness = dict(seed=2026)
