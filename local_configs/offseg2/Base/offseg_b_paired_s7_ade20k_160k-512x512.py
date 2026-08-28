# Paired OffSeg-B baseline, draw 7 of two.
#
# Context: an independent re-draw of the 47.79 main model returned 46.82, so
# run-to-run spread on this line is about 1.0 mIoU -- larger than the entire
# CCM -> ACS -> IACS -> responsibility chain (0.99).  Every single-run delta
# inside that chain is therefore unmeasurable, and the only claim left with a
# chance of exceeding the noise is the coarse one: whole method vs plain
# OffSeg in this environment.
#
# These two draws serve two purposes at once:
#   1. a baseline MEAN to compare against the method's mean (47.79, 46.82);
#   2. the baseline's own SPREAD, which says whether ~1.0 is a property of
#      this dataset/schedule/environment or specifically of our head.
#      Wide baseline spread  -> the noise is environmental, our head is fine.
#      Narrow baseline spread -> our head is the unstable part, which is a
#      mechanism finding and is fixable.
#
# Head is plain OffSegHead; every run setting matches the CCM/ACS/IACS chain.
_base_ = ['./offseg_b_paired_ade20k_160k-512x512.py']

randomness = dict(seed=7, deterministic=False)

work_dir = './work_dirs/offseg_b_paired_s7_ade20k_160k-512x512'
