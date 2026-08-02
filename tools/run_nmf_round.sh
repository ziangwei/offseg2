#!/usr/bin/env bash
# Submit the three-slot product round from the repository root.
# Order is intentional: the product bet first, then its NMF-only attribution
# row, then the already prepared CCM+PCE alternative.

set -euo pipefail

scripts=(
    tools/slurm/nmf2_ccm.sh
    tools/slurm/nmf1_only.sh
    tools/slurm/ev3_ccm_pce.sh
)

for script in "${scripts[@]}"; do
    job_id=$(sbatch --parsable "$script")
    printf '%-34s job %s\n' "$script" "$job_id"
done
