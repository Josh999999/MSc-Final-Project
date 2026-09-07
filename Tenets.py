"""External Imports (Libraries and APIs)"""
import numpy as np
 
 
"""Local Imports"""
from Config import Config
from GRN import _cos, handle_develop, evaluate_fitness, mask_indices, masked_matrix
from Induction import handle_induction
 
 
 
 
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
    S = np.atleast_2d(np.asarray(S, dtype = float))
 
    H = S.T @ S
 
    np.fill_diagonal(H, 0.0)
 
 
    return masked_matrix(H, cfg.mask)
 
 
 
 
def measure_tenet1(B: np.ndarray, cfg: Config, S: np.ndarray = None) -> float:
    B = np.asarray(B, dtype = float).copy()
    np.fill_diagonal(B, 0.0)
 
    H = ideal_hebbian(cfg, S)
 
 
    return _cos(B[cfg.mask], H[cfg.mask])
 
 
 
 
def measure_tenet2(G: np.ndarray, cfg: Config, S: np.ndarray = None) -> float:
    S = cfg.target if S is None else S
 
 
    return _cos(G, S)
 
 
 
def measure_phenotype_alignment(P: np.ndarray, B: np.ndarray, cfg: Config) -> float:
    P = np.asarray(P, dtype = float)
    B = np.asarray(B, dtype = float)
 
    PP = np.outer(P, P).copy()
    np.fill_diagonal(PP, 0.0)
 
 
    return _cos(PP[cfg.mask], B[cfg.mask])
 
 
 
def split_gain(B: np.ndarray) -> tuple:
    """Separate B into direction (unit Frobenius) and magnitude."""
    B = np.asarray(B, dtype = float)
    Y = float(np.linalg.norm(B))
 
 
    return (B / Y if Y > 0 else B.copy()), Y
 
 
 
 
def synthesise_B(a1: float, cfg: Config, rng: np.random.Generator, S: np.ndarray = None) -> np.ndarray:
    """Unit-Frobenius B whose Tenet 1 score is exactly a1."""
    idx = mask_indices(cfg.mask)
    symmetric = bool(np.array_equal(cfg.mask, cfg.mask.T))
 
    H = ideal_hebbian(cfg, S)
    u = _unit(H.ravel()[idx])
 
    R = rng.normal(size = (cfg.N, cfg.N))
 
 
    if symmetric:
        R = (R + R.T) / 2.0
 
 
    np.fill_diagonal(R, 0.0)
    R = masked_matrix(R, cfg.mask)
    v = _orthogonal_component(R.ravel()[idx], u)
 
    vals = _interpolate_to_cosine(u, v, a1)
 
    B = np.zeros((cfg.N, cfg.N))
    B.ravel()[idx] = vals
 
 
    if symmetric:               # projection can perturb symmetry slightly
        B = masked_matrix((B + B.T) / 2.0, cfg.mask)
        np.fill_diagonal(B, 0.0)
 
 
    return B / np.linalg.norm(B)
 
 
 
 
def synthesise_G(
        a2: float,
        cfg: Config,
        rng: np.random.Generator,
        S: np.ndarray = None,
        amplitude: float = 1.0
    ) -> np.ndarray:
    """Genotype whose Tenet 2 score is exactly a2, with per-gene RMS amplitude."""
    S = cfg.target if S is None else np.asarray(S, dtype = float)
 
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
        grid: int = 21
    ) -> dict:
    S_eval = cfg.target if S_eval is None else np.asarray(S_eval, dtype = float)
 
 
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
 
                P = handle_develop(G, B, cfg, induction = cfg.induction)
 
 
                if cfg.induction:
                    history = handle_induction(B, P, G, cfg, rng, S = S_eval)
                    F = history["auc"]
                    acc += F
                else:
                    acc += evaluate_fitness(P, S_eval, cfg)
 
 
            Z[i, j] = acc / n_seeds
 
 
    return {
        "Z": Z,
        "a1_grid": a1_grid,
        "a2_grid": a2_grid,
        "Y": cfg.Y,
        "max_tenet1_error": float(np.max(np.abs(check1))),
        "max_tenet2_error": float(np.max(np.abs(check2))),
        "config": cfg,
    }