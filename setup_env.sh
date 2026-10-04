#!/usr/bin/env bash
#
# Configure conda, build the project environment and test it.
# Falls back to a venv (built on the module's python) if conda fails.
#
#   source ./setup_env.sh          <- MUST be sourced, not executed
#
# Options:  ENV_NAME=grn  PY_VERSION=3.12  FORCE_VENV=1

ENV_NAME="${ENV_NAME:-grn}"
PY_VERSION="${PY_VERSION:-3.12}"
VENV_DIR="${VENV_DIR:-$HOME/${ENV_NAME}-env}"

_ok()   { echo "  [ ok ] $*"; }
_info() { echo "  [info] $*"; }
_fail() { echo "  [FAIL] $*" >&2; }

echo "======================================================================"
echo " 1. Loading the conda module"
echo "======================================================================"
if module load conda/python3 2>/dev/null; then
    _ok "module load conda/python3"
else
    _fail "module load conda/python3 failed"
    _info "available conda modules:"
    module avail conda 2>&1 | sed 's/^/        /'
    return 1 2>/dev/null || exit 1
fi

echo
echo "======================================================================"
echo " 2. Conda information"
echo "======================================================================"
CONDA_BIN="$(command -v conda || true)"
if [ -z "$CONDA_BIN" ]; then
    _fail "no 'conda' on PATH after module load"
    return 1 2>/dev/null || exit 1
fi
_ok "conda binary : $CONDA_BIN"

CONDA_BASE="$(conda info --base 2>/dev/null)"
_ok "conda base   : ${CONDA_BASE:-<unknown>}"

PROFILE="${CONDA_BASE}/etc/profile.d/conda.sh"
if [ -f "$PROFILE" ]; then
    # This is what defines the `conda activate` shell function.
    # `module load` alone does not, which is why activate can fail.
    # shellcheck disable=SC1090
    source "$PROFILE"
    _ok "sourced $PROFILE"
else
    _fail "conda.sh not found at $PROFILE"
fi

echo
echo "======================================================================"
echo " 3. Channels"
echo "======================================================================"
if conda config --show channels 2>/dev/null | grep -q conda-forge; then
    _ok "conda-forge already configured"
else
    _info "adding conda-forge"
    conda config --add channels conda-forge && _ok "added" || _fail "could not add channel"
fi
conda config --show channels 2>/dev/null | sed 's/^/        /'

echo
echo "======================================================================"
echo " 4. Environment '${ENV_NAME}'"
echo "======================================================================"
CONDA_OK=0
if [ "${FORCE_VENV:-0}" = "1" ]; then
    _info "FORCE_VENV=1, skipping conda environment"
else
    if conda env list 2>/dev/null | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
        _ok "environment already exists"
    else
        _info "creating (first run, may take a few minutes) ..."
        conda create -n "${ENV_NAME}" -c conda-forge \
              "python=${PY_VERSION}" numpy matplotlib -y \
            && _ok "created" || _fail "conda create failed"
    fi

    if conda activate "${ENV_NAME}" 2>/dev/null; then
        _ok "activated '${ENV_NAME}'"
        CONDA_OK=1
    else
        _fail "could not activate '${ENV_NAME}'"
    fi
fi

echo
if [ "$CONDA_OK" -ne 1 ]; then
    echo "======================================================================"
    echo " 4b. Falling back to a venv on the module's python"
    echo "======================================================================"
    if [ ! -d "$VENV_DIR" ]; then
        _info "creating venv at $VENV_DIR"
        python3 -m venv "$VENV_DIR" || { _fail "venv creation failed"; return 1 2>/dev/null || exit 1; }
    fi
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate" || { _fail "venv activate failed"; return 1 2>/dev/null || exit 1; }
    _ok "activated venv $VENV_DIR"
    _info "installing numpy matplotlib"
    python -m pip install --quiet --upgrade pip
    python -m pip install --quiet numpy matplotlib && _ok "installed" || _fail "pip install failed"
fi

echo
echo "======================================================================"
echo " 5. Testing the environment"
echo "======================================================================"
_ok "python  : $(command -v python)"
_ok "version : $(python -V 2>&1)"

python - <<'PYEOF'
import sys
ok = True
for mod in ("numpy", "matplotlib"):
    try:
        m = __import__(mod)
        print(f"  [ ok ] import {mod:<11} {m.__version__}")
    except Exception as e:
        ok = False
        print(f"  [FAIL] import {mod:<11} {e}")
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    fig, ax = plt.subplots()
    ax.plot(np.arange(5), np.arange(5) ** 2)
    fig.savefig("_env_test.png", dpi=50)
    plt.close(fig)
    print("  [ ok ] headless plot written (_env_test.png)")
except Exception as e:
    ok = False
    print(f"  [FAIL] plotting: {e}")
sys.exit(0 if ok else 1)
PYEOF
status=$?

echo
if [ "$status" -eq 0 ]; then
    rm -f _env_test.png
    echo "======================================================================"
    echo " ENVIRONMENT READY"
    echo "======================================================================"
    echo " Next:   mkdir -p out && bash submit_all.sh"
else
    echo "======================================================================"
    echo " ENVIRONMENT NOT READY - see the [FAIL] lines above"
    echo "======================================================================"
fi
