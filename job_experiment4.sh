#!/bin/bash -l
#SBATCH --job-name=grn-exp4
#SBATCH -A ecsstudents
#SBATCH -p ecsstudents_l4
#SBATCH --nodes=1
#SBATCH -c 1
#SBATCH --mem=2G
#SBATCH --time=12:00:00
#SBATCH --output=out/%x-%j.out
#SBATCH --error=out/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jm2e25@soton.ac.uk
#
# Experiment 4: r-round induction surfaces over (T1, T2), 21x21 grid
#     mkdir -p out && sbatch job_experiment4.sh
#
# Pure NumPy, single-threaded: one core and 2 GB are enough (measured runs
# used ~120 MB). No --gres=gpu. ecsstudents_l4 has a 24 h wall-clock limit.
# Expected wall time: ~2 h at rounds = 10 (~10 h at rounds = 50).

module load conda/python3

# Define the `conda activate` shell function (module load alone does not).
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME:-grn}"

# Work from the directory holding the Experiment*.py files. Set PROJECT_DIR if
# the .py files live somewhere other than this script's folder.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR:-$SCRIPT_DIR}" || exit 1

if [ ! -f "Experiment4.py" ]; then
    echo "ERROR: Experiment4.py not found in $(pwd)" >&2
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
python tests.py || { echo "REGRESSION TESTS FAILED -- not running Experiment 4"; exit 1; }
echo "----------------------------------------------------------------------"

python Experiment4.py
status=$?

echo "----------------------------------------------------------------------"
echo "finished  : $(date)  (exit ${status})"
exit ${status}
