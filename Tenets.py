"""External Imports (Libraries and APIs)"""
import numpy as np
 
 
"""Local Imports"""
from Config import Config
from GRN import DTYPE, _cos, handle_develop, evaluate_fitness, masked_matrix
from Induction import handle_induction
from Plastic_Induction import energy, differential_energy
 
 
 
 
def _unit(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x)
 
 
    return x / n if n > 0 else x
 
 
 
 
def _orthogonal_component(v: np.ndarray, u: np.ndarray) -> np.ndarray:
    v = v - (v @ u) * u
 
 
    return _unit(v)
 
 
 
 
def _interpolate_to_cosine(u: np.ndarray, v: np.ndarray, a: float) -> np.ndarray:
    """Unit vector x with cos(x, u) == a exactly, for orthonormal u, v."""
 
 
    return a * u + np.sqrt(max(0.0, 1.0 - a * a)) * v
 
 
 
 
def ideal_hebbian(cfg: Config, S: np.ndarray = None) -> np.ndarray:
    S = cfg.targets if S is None else S
    S = np.atleast_2d(np.asarray(S, dtype = DTYPE))
 
    H = S.T @ S
 
 
 
    return masked_matrix(H, cfg.mask)
 
 
 
 
def measure_tenet1(B: np.ndarray, cfg: Config, S: np.ndarray = None) -> float:
    B = np.asarray(B, dtype = DTYPE)
    H = ideal_hebbian(cfg, S)
 
 
    return _cos(B[cfg.mask], H[cfg.mask])
 
 
 
 
def measure_tenet2(G: np.ndarray, cfg: Config, S: np.ndarray = None) -> float:
    S = cfg.target if S is None else S
 
 
    return _cos(G, S)
 
 
 
def synthesise_B(a1: float, cfg: Config, rng: np.random.Generator, S: np.ndarray = None) -> np.ndarray:
    idx = cfg.allowed()
    symmetric = bool(np.array_equal(cfg.mask, cfg.mask.T))
 
    H = ideal_hebbian(cfg, S)
    u = _unit(H.ravel()[idx])
 
    R = rng.normal(size = (cfg.N, cfg.N))
 
 
    if symmetric:
        R = (R + R.T) / 2.0
 
 
    R = masked_matrix(R, cfg.mask)
    v = _orthogonal_component(R.ravel()[idx], u)
 
    vals = _interpolate_to_cosine(u, v, a1)
 
    B = np.zeros((cfg.N, cfg.N))
    B.ravel()[idx] = vals
 
 
    if symmetric:               # projection can perturb symmetry slightly
        B = masked_matrix((B + B.T) / 2.0, cfg.mask)
 
 
    return B / np.linalg.norm(B)
 
 
 
 
def synthesise_G(
        a2: float,
        cfg: Config,
        rng: np.random.Generator,
        S: np.ndarray = None,
        amplitude: float = 1.0
    ) -> np.ndarray:
    S = cfg.target if S is None else np.asarray(S, dtype = DTYPE)
 
    u = _unit(S)
    v = _orthogonal_component(rng.normal(size = S.shape), u)
    G = _interpolate_to_cosine(u, v, a2)
 
 
    return G * amplitude * np.sqrt(S.size)
 
 
 
 
def fitness_surface(
        cfg: Config,
        rng: np.random.Generator,
        S_eval: np.ndarray = None,
        n_seeds: int = 8,
        amplitude: float = 1.0,
        grid: int = 21,
        induction_measure: str = "auc_inner",
        measure: str = "fitness"
    ) -> dict:
    S_eval = cfg.target if S_eval is None else np.asarray(S_eval, dtype = DTYPE)
 
 
    if S_eval.size != cfg.N:
 
        raise ValueError(f"S_eval has {S_eval.size} genes but N={cfg.N}")
 
 
    a1_grid = np.linspace(-1, 1, grid)
    a2_grid = np.linspace(-1, 1, grid)
 
    Z = np.zeros((a2_grid.size, a1_grid.size))
    check1, check2 = [], []
 
 
    for i, a2 in enumerate(a2_grid):
 
        for j, a1 in enumerate(a1_grid):
 
            acc = 0.0
 
 
            for _ in range(n_seeds):
                B = cfg.Y * synthesise_B(a1, cfg, rng)
                G = synthesise_G(a2, cfg, rng, S = S_eval, amplitude = amplitude)
 
                # Check alignment BEFORE applying the gain, so the diagnostic
                # is about direction only. cos is scale-invariant either way.
                check1.append(measure_tenet1(B, cfg) - a1)
                check2.append(measure_tenet2(G, cfg, S_eval) - a2)
 
                P = handle_develop(G, B, cfg, induction = False)
 
 
                if cfg.induction:
                    history = handle_induction(B, P, G, cfg, rng, S = S_eval)
                    M = history[induction_measure]
                    acc += M

                else:
                    measure = measure.strip().lower()


                    if measure == "fitness":
                        acc += evaluate_fitness(P, S_eval, cfg)

                    elif measure == "energy":
                        acc += energy(P, B, cfg, normalise = cfg.normalise_energy, normalise_interactions = cfg.energy_normalise_interactions)

                    elif measure == "diff_energy":
                        acc += differential_energy(P, G, B, cfg, normalise = cfg.normalise_energy, normalise_interactions = cfg.energy_normalise_interactions)

                    else:
                        acc += 1
 
 
            Z[i, j] = acc / n_seeds
 
 
    return {
        "Z": Z,
        "a1_grid": a1_grid,
        "a2_grid": a2_grid,
        "Y": cfg.Y,
        "max_tenet1_error": float(np.max(np.abs(check1))),
        "max_tenet2_error": float(np.max(np.abs(check2))),
        "config": cfg.copy(),   # snapshot: cfg is mutable and may change after this call
    }
 
 
 
def plastic_measure_surface(
        cfg: Config,
        rng: np.random.Generator,
        measure: str = "auc_inner",
        S_eval: np.ndarray = None,
        n_seeds: int = 4,
        amplitude: float = 1.0,
        grid: int = 15
    ) -> dict:
    S_eval = cfg.target if S_eval is None else np.asarray(S_eval, dtype = DTYPE)
 
 
    if S_eval.size != cfg.N:
 
        raise ValueError(f"S_eval has {S_eval.size} genes but N={cfg.N}")
 
 
    a1_grid = np.linspace(-1, 1, grid)
    a2_grid = np.linspace(-1, 1, grid)
 
    Z = np.zeros((a2_grid.size, a1_grid.size))
    check1, check2 = [], []
 
 
    for i, a2 in enumerate(a2_grid):
 
        for j, a1 in enumerate(a1_grid):
 
            acc = 0.0
 
 
            for _ in range(n_seeds):
                B = cfg.Y * synthesise_B(a1, cfg, rng)
                G = synthesise_G(a2, cfg, rng, S = S_eval, amplitude = amplitude)
 
                check1.append(measure_tenet1(B, cfg) - a1)
                check2.append(measure_tenet2(G, cfg, S_eval) - a2)
 
                # Develop under the SCALED matrix, then run the plastic search
                # on that phenotype with the same matrix.
                P = handle_develop(G, B, cfg, induction = False)
                history = handle_induction(B, P, G, cfg, rng, S = S_eval, limit_return = False)
 
 
                if measure not in history:
                    raise KeyError(
                        f"'{measure}' is not reported by the induction process. "
                        f"Available: {sorted(k for k in history if np.isscalar(history[k]))}"
                    )
 
 
                acc += float(history[measure])
 
 
            Z[i, j] = acc / n_seeds
 
 
    return {
        "Z": Z,
        "a1_grid": a1_grid,
        "a2_grid": a2_grid,
        "Y": cfg.Y,
        "measure": measure,
        "max_tenet1_error": float(np.max(np.abs(check1))),
        "max_tenet2_error": float(np.max(np.abs(check2))),
        "config": cfg.copy(),   # snapshot: cfg is mutable and may change after this call
    }
 