# Repository research instructions

Before proposing, implementing, or interpreting any thesis experiment in this repository:

1. Read `THESIS_ROUTE.md` completely. It is the canonical research context.
2. Read `EXPERIMENTS.md` for result/status labels. Never infer a result from a config name.
3. Inspect the current implementation/config before repeating formulas from old chat context.
4. Treat unrelated tracked changes and all untracked files as user-owned; do not delete, stage,
   or rewrite them unless explicitly in scope.

Non-negotiable defaults are recorded in `THESIS_ROUTE.md`: ADE uses the fixed
EfficientFormerV2-S2 backbone; no distillation or external model; do not rely on a new loss or a
heavy decoder; preserve clear independence from PARSeg's attribute branch and gated fusion.

Research integrity rules:

- Distinguish paper results, user-reported final results, interim/peak readings, probes, and
  config-ready experiments.
- Do not claim that OffSeg 45.9 is a matched local baseline, that 47.79 is multi-seed, or that
  Stuff generalization/FLOPs are complete until the ledger says so.
- Attribute GCR-style class subspaces, soft responsibilities/moments, OffSeg, FreqFusion, RCM,
  NMF and other published mechanisms. The canonical contribution boundary is in
  `THESIS_ROUTE.md`; never rename borrowed components as original work.
- A failed implementation closes only that implementation, not an entire research family.
- When the user reports a new result, first update `EXPERIMENTS.md`, then synchronize the status,
  one-page conclusion, and evidence gaps in `THESIS_ROUTE.md`.
- Record the exact config, dataset/schedule, stage (final/peak/interim), seed if known, mIoU,
  correct control delta, relevant final needles, checkpoint/log location, and commit if known.

Historical PARSeg/Dual/NMF configs and `PARSeg_experiment_summary.log` are retained for audit.
They are non-canonical unless a fact is promoted with an explicit source label in the ledger.

## Inference rules learned the hard way

These were each violated once in this repository and cost a full training slot or a
retraction. They are not general advice; they are specific to how this project fails.

1. **Never read a mechanism conclusion out of a difference between two losing arms.**
   `pairwhiten 46.95` vs `pairraw 46.19` was written up as "whitening is the right
   object, only the penalty form failed" and used three times to keep the pair line
   alive. Both arms were below their control. The line's third attempt scored 46.00,
   the worst addition in the project. Only when at least one arm beats the control may
   the between-arm difference be given a mechanism reading.
2. **A two-class separability probe does not predict 150-way trainability.** "The top
   confusion pairs are 98-100% linearly separable in the frozen features" motivated a
   component that lost 1.79. Fitting a direction for a named pair in 256 dimensions is
   easy; keeping the other 148 classes right while using it is the problem. The probe
   table in `EXPERIMENTS.md` §3 also states that its numbers come from the PARSeg3
   48.17 checkpoint, not from the current model; it may not be used alone as a design
   basis.
3. **Grep the ledger before writing "this has never been tried."** `iacs_candidate_topk`
   was proposed as an untouched axis; `offsegccmiacs_r4_top3` had already measured
   47.08. Check `grep -n <keyword> EXPERIMENTS.md` and `ls local_configs/offseg2/*/`
   first. This is rule 2 of the section above, restated because it was broken twice.
4. **Do not tune a hyperparameter of something that has not yet produced a positive
   result.** Tuning comes after a component works.
5. **Prefer a needle from the current best model over a probe from an older one.** The
   only positive result to date (proto, 48.12) came from the winner's own
   `effective_support`; the component built on the older probe table lost by 1.79.
6. **Verify new head code numerically before submitting it.** Install CPU torch, stub
   the parent class, copy the real file in, and assert on shapes and behaviour.
   Reading the code has missed three real bugs.
