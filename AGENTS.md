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
