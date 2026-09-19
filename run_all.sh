#!/usr/bin/env bash
# Convenience wrapper around run_experiments.py
#
#   ./run_all.sh              # experiments 1, 2, 4 at paper scale + 3 reduced
#   ./run_all.sh --quick      # fast smoke run of all four
#
# Any arguments are passed straight through to run_experiments.py.
set -euo pipefail
cd "$(dirname "$0")"

if [ "$#" -gt 0 ]; then
    exec python3 run_experiments.py "$@"
fi

# Default: full scale for the tractable experiments, reduced for the large one.
python3 run_experiments.py 1 2 4 --outdir figures
python3 run_experiments.py 3 --gens 2000 --outdir figures
