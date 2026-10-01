#!/bin/bash -l
#SBATCH --job-name=grn-exp1
#SBATCH --nodes=1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=out/%x-%j.out
#SBATCH --error=out/%x-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jm2e25@soton.ac.uk
#SBATCH -A ecsstudents
#SBATCH -p ecsstudents_l4
#SBATCH --time=24:00:00
#
# Run Experiment 1.
#   mkdir -p out && sbatch job_experiment1.sh
#
# Pure NumPy, so no GPU is requested; CPU partitions queue faster.

module load conda/python3

# Define the `conda activate` shell function (module load alone does not).
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME:-grn}"

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

echo "host      : $(hostname)"
echo "started   : $(date)"
echo "python    : $(which python)"
python -c "import numpy, matplotlib; print('numpy', numpy.__version__, '| matplotlib', matplotlib.__version__)"
echo "----------------------------------------------------------------------"

python Experiment1.py
status=$?

echo "----------------------------------------------------------------------"
echo "finished  : $(date)  (exit ${status})"
exit ${status}
