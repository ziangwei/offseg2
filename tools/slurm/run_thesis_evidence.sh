#!/bin/bash
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Submit with submit_thesis_evidence.py'; exit 2; }
record=${1:?Missing run.json}
readarray -t settings < <(python3 - "$record" <<'PY'
import json,sys
m=json.load(open(sys.argv[1]))
for k in ('conda_base','conda_env','run_dir'):
    print(m[k])
PY
)
set +u
source "${settings[0]}/etc/profile.d/conda.sh"
conda activate "${settings[1]}"
set -u
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
unset RANK WORLD_SIZE LOCAL_RANK MASTER_ADDR MASTER_PORT GROUP_RANK ROLE_RANK
exec 9>"${settings[2]}/.evidence.lock"
flock -n 9 || { echo 'Evidence job already running'; exit 2; }
python -c 'import torch,cv2,matplotlib; assert torch.cuda.device_count()==4; print(torch.__version__,matplotlib.__version__)'
exec python -m tools.thesis_evidence.pipeline "$record"
