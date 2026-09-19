#!/bin/bash -l
# Submit every experiment as a separate SLURM job.
#
#   bash submit_all.sh
#
# Each job writes to out/<job-name>-<jobid>.out
set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p out

JOBS=(
    job_experiment1.sh
    job_experiment2.sh
    job_experiment3.sh
)

for job in "${JOBS[@]}"; do
    if [ ! -f "$job" ]; then
        echo "SKIP  $job (not found)" >&2
        continue
    fi
    id=$(sbatch --parsable "$job") && echo "submitted $job  -> job $id" \
        || echo "FAILED to submit $job" >&2
done

echo
echo "Check progress with:  squeue -u \$USER"
echo "Logs appear in:       $SCRIPT_DIR/out/"
