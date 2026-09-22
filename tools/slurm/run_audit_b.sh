#!/bin/bash
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Use submit_audit_b.py / sbatch.'; exit 2; }
record=${1:?Missing run.json}
# Read metadata with system Python before activating the recorded environment.
readarray -t settings < <(python3 - "$record" <<'PY'
import json,sys
m=json.load(open(sys.argv[1]))
for k in ('conda_base','conda_env','run_dir','config','checkpoint','offseg_checkpoint'):
    print(m.get(k) or '')
PY
)
set +u
source "${settings[0]}/etc/profile.d/conda.sh"
conda activate "${settings[1]}"
set -u
run=${settings[2]}
config=${settings[3]}
checkpoint=${settings[4]}
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
unset RANK WORLD_SIZE LOCAL_RANK MASTER_ADDR MASTER_PORT GROUP_RANK ROLE_RANK
python -c 'import torch,cv2; assert torch.cuda.device_count()==4; print(torch.__version__,cv2.__version__)'
exec 9>"$run/.audit.lock"
flock -n 9 || { echo 'This audit is already running'; exit 2; }
status=0
python -m torch.distributed.run --rdzv-backend=c10d --rdzv-endpoint=localhost:0 \
    "--rdzv-id=audit-$SLURM_JOB_ID" --nnodes=1 --nproc_per_node=4 --max_restarts=0 \
    tools/route_error_audit_b.py "$config" "$checkpoint" \
    --launcher pytorch --work-dir "$run/errors" || status=$?
extra=()
if [[ -n "${settings[5]}" ]]; then extra+=(--offseg-checkpoint "${settings[5]}"); fi
python tools/route_cost_audit_b.py "$config" "$checkpoint" \
    --work-dir "$run/cost" "${extra[@]}" || status=$?
python - "$run" "$status" <<'PY'
import json,sys,zipfile
from pathlib import Path
run=Path(sys.argv[1])
(run/'audit_exit.json').write_text(json.dumps({'exit_code':int(sys.argv[2])}))
with zipfile.ZipFile(run/'route_audit_b.zip','w',zipfile.ZIP_DEFLATED) as z:
    files=[run/'run.json',run/'audit_exit.json']
    for folder in ('errors','cost','logs'):
        files += [p for p in (run/folder).rglob('*') if p.is_file() and p.suffix in ('.json','.csv','.md','.log','.py')]
    for p in files:
        z.write(p,str(p.relative_to(run)))
print('Send back:',run/'route_audit_b.zip')
PY
exit "$status"
