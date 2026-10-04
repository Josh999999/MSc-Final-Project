#!/usr/bin/env bash
#
# Iridis environment setup.
#
#   source ./_setup.sh          <- MUST be sourced, not executed
#
# Creates the conda environment on first use, then activates it.
# Override the name with:  ENV_NAME=other source ./_setup.sh

ENV_NAME="${ENV_NAME:-grn}"
PY_VERSION="${PY_VERSION:-3.12}"

if [ -n "${BASH_SOURCE[0]:-}" ]; then
    PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
else
    PROJECT_DIR="$(pwd)"
fi

module load conda/python3 || {
    echo "ERROR: 'module load conda/python3' failed. Try: module avail conda" >&2
    return 1 2>/dev/null || exit 1
}

# `module load` puts conda on PATH but does not define the `conda activate`
# shell function.  Sourcing conda.sh does, and works in both interactive
# shells and batch jobs -- this is more reliable than relying on `conda init`.
CONDA_BASE="$(conda info --base 2>/dev/null)"
if [ -n "$CONDA_BASE" ] && [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
else
    echo "ERROR: could not locate conda.sh under '${CONDA_BASE}'." >&2
    return 1 2>/dev/null || exit 1
fi

# Make sure at least one channel exists, or `conda create` cannot resolve.
if ! conda config --show channels 2>/dev/null | grep -q "conda-forge"; then
    echo "Adding conda-forge channel ..."
    conda config --add channels conda-forge
fi

# Create the environment the first time only.
if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "Creating conda environment '${ENV_NAME}' (first run only) ..."
    conda create -n "${ENV_NAME}" -c conda-forge \
        "python=${PY_VERSION}" numpy matplotlib -y || {
        echo "ERROR: conda create failed." >&2
        return 1 2>/dev/null || exit 1
    }
fi

conda activate "${ENV_NAME}" || {
    echo "ERROR: could not activate '${ENV_NAME}'." >&2
    return 1 2>/dev/null || exit 1
}

python -c "import numpy, matplotlib; print('OK:', 'numpy', numpy.__version__, '| matplotlib', matplotlib.__version__)" || {
    echo "ERROR: numpy/matplotlib not importable in '${ENV_NAME}'." >&2
    echo "       Try: pip install numpy matplotlib" >&2
    return 1 2>/dev/null || exit 1
}

echo "Environment '${ENV_NAME}' ready.  python = $(which python)"
