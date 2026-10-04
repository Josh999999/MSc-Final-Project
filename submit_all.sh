#!/bin/bash -l
# Submit experiments as separate SLURM jobs.
#
#   bash submit_all.sh              # experiments 1-7 (8 is an array: see job_experiment8.sh)
#   bash submit_all.sh 1 3          # only experiments 1 and 3
#   bash submit_all.sh 8            # the design-grid array plus its dependent summarise job
#
# If the .py files live elsewhere (e.g. these scripts are in scripts/):
#   PROJECT_DIR=$HOME/MSc-Final-Project bash submit_all.sh
set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORK_DIR="${PROJECT_DIR:-$SCRIPT_DIR}"

cd "$WORK_DIR" || exit 1
mkdir -p out

TODO=("$@")
[ ${#TODO[@]} -eq 0 ] && TODO=(1 2 3 4 5 6 7)

for n in "${TODO[@]}"; do
    job="${SCRIPT_DIR}/job_experiment${n}.sh"

    if [ ! -f "$job" ]; then
        echo "SKIP  experiment $n (no $job)" >&2
        continue
    fi

    if [ ! -f "${WORK_DIR}/Experiment${n}.py" ]; then
        echo "SKIP  experiment $n (no ${WORK_DIR}/Experiment${n}.py)" >&2
        continue
    fi

    if [ "$n" = "8" ]; then
        # size the array from the grid, then chain the summarise step
        count=$(python Experiment8.py --count) || { echo "FAILED to count Experiment 8 tasks" >&2; continue; }
        if id=$(sbatch --parsable --array=0-$((count-1))%20 --export=ALL,PROJECT_DIR="$WORK_DIR" "$job"); then
            echo "submitted experiment 8 array ($count tasks) -> job $id"
            sid=$(sbatch --parsable --dependency=afterok:"$id" --export=ALL,PROJECT_DIR="$WORK_DIR" "${SCRIPT_DIR}/job_experiment8_summarise.sh") \
                && echo "submitted experiment 8 summarise      -> job $sid (runs when the array completes)"
        else
            echo "FAILED to submit experiment 8" >&2
        fi
        continue
    fi

    if id=$(sbatch --parsable --export=ALL,PROJECT_DIR="$WORK_DIR" "$job"); then
        echo "submitted experiment $n  -> job $id"
    else
        echo "FAILED to submit experiment $n" >&2
    fi
done

echo
echo "Check progress with:  squeue -u \$USER"
echo "Logs appear in:       ${WORK_DIR}/out/   (named grn-expN-<jobid>.out)"
