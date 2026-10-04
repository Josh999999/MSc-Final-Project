#!/bin/bash -l
#SBATCH --job-name=grn-exp8
#SBATCH -A ecsstudents
#SBATCH -p ecsstudents_l4
#SBATCH --nodes=1
#SBATCH -c 1
#SBATCH --mem=2G
#SBATCH --time=08:00:00
#SBATCH --array=0-169%20
#SBATCH --output=out/%x-%A_%a.out
#SBATCH --error=out/%x-%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jm2e25@soton.ac.uk
#
# Experiment 8: the design grid, one array task per (configuration, seed) pair
#     mkdir -p out && sbatch job_experiment8.sh
#
# Pure NumPy, single-threaded: one core and 2 GB are enough (measured runs
# used ~120 MB). No --gres=gpu. ecsstudents_l4 has a 24 h wall-clock limit.
# Expected wall time: ~1 h per task at rounds = 10 (~5 h at rounds = 50).
#
#     bash submit_all.sh 8      (sizes the array from the grid and chains the summarise job)
# or by hand:
#     python Experiment8.py --count          # prints the number of tasks, N (17 configurations x 10 seeds = 170)
#     sbatch --array=0-$((N-1))%20 job_experiment8.sh
#     sbatch --dependency=afterok:<array job id> job_experiment8_summarise.sh
#
# Each task writes Experiment8/scratch/<row>_<seed>.json; the summarise job
# builds both tables from those files without re-running anything. %20 caps
# the number of tasks running at once; raise it if the queue allows. Delete
# Experiment8/scratch/ before a fresh full run so old tasks are not mixed in.

module load conda/python3

# Define the `conda activate` shell function (module load alone does not).
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME:-grn}"

# Work from the directory holding the Experiment*.py files. Set PROJECT_DIR if
# the .py files live somewhere other than this script's folder.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR:-$SCRIPT_DIR}" || exit 1

if [ ! -f "Experiment8.py" ]; then
    echo "ERROR: Experiment8.py not found in $(pwd)" >&2
    echo "       sbatch --export=ALL,PROJECT_DIR=/path/to/project job_experiment1.sh" >&2
    exit 1
fi

mkdir -p out

echo "host      : $(hostname)"
echo "workdir   : $(pwd)"
echo "started   : $(date)"
echo "python    : $(which python)"
python -c "import numpy, matplotlib; print('numpy', numpy.__version__, '| matplotlib', matplotlib.__version__)"
echo "git       : $(git rev-parse --short HEAD 2>/dev/null || echo 'not a git checkout')"
echo "----------------------------------------------------------------------"

# The regression suite gates every cluster run: no figure is produced by a
# build that does not pass its own tests.
python tests.py || { echo "REGRESSION TESTS FAILED -- not running Experiment 8"; exit 1; }
echo "----------------------------------------------------------------------"

python Experiment8.py "${SLURM_ARRAY_TASK_ID}"
status=$?

echo "----------------------------------------------------------------------"
echo "finished  : $(date)  (exit ${status})"
exit ${status}
