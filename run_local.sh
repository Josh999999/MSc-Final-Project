#!/bin/bash -l
# Run the experiments directly (login node or interactive session), no SLURM.
# For quick tests only - submit real runs with submit_all.sh.
#
#   bash run_local.sh          # all experiments
#   bash run_local.sh 1 3      # only experiments 1 and 3
set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

module load conda/python3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME:-grn}" || {
    echo "ERROR: environment not active. Run:  source ./_setup.sh" >&2
    exit 1
}

TODO=("$@")
[ ${#TODO[@]} -eq 0 ] && TODO=(1 2 3)

for n in "${TODO[@]}"; do
    echo
    echo "======================================================================"
    echo " Experiment $n"
    echo "======================================================================"
    start=$SECONDS
    if python "Experiment${n}.py"; then
        echo "  finished in $(( SECONDS - start ))s"
    else
        echo "  FAILED after $(( SECONDS - start ))s" >&2
    fi
done
