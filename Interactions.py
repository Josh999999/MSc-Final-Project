"""External Imports (Libraries and APIs)"""
import numpy as np
 
 
"""Local Imports"""
from Config import Config
from GRN import masked_matrix
 
 
 
 
def _finalise(B: np.ndarray, cfg: Config, symmetric: bool = None, normalise: bool = None) -> np.ndarray:
    symmetric = cfg.symmetric_mask if symmetric is None else symmetric
    normalise = cfg.interactions_norm if normalise is None else normalise
 
    B = np.asarray(B, dtype = float).copy()
    np.fill_diagonal(B, 0.0)
 
 
    if symmetric:
        B = (B + B.T) / 2.0
        np.fill_diagonal(B, 0.0)
 
 
    B = masked_matrix(B, cfg.mask)
 
 
    if normalise:
        r = np.linalg.norm(B, ord = "fro")
        B = B * (cfg.Y / r) if r > 0 else np.zeros_like(B)
 
 
    return B
 
 
 
 
def random_interactions(cfg: Config, rng: np.random.Generator, normalise: bool = None) -> np.ndarray:
    signs = rng.choice([-1.0, 1.0], size = (cfg.N, cfg.N), p = [1 - cfg.prob_pos_B, cfg.prob_pos_B])
 
 
    if cfg.uniform:
        B = signs
 
    else:
        B = rng.uniform(low = 0.0, high = 1.0, size = (cfg.N, cfg.N)) * signs
 
 
    return _finalise(B, cfg, normalise = normalise)
 
 
 
 
def appropriate_interactions(
        cfg: Config,
        rng: np.random.Generator,
        S: np.ndarray = None,
        inappropriate: bool = False,
        normalise: bool = None
    ) -> np.ndarray:
 
    S = cfg.target if S is None else np.asarray(S, dtype = float)
 
    # Set value of signs in alignment with the target
    signs = np.outer(S, S)
 
 
    if inappropriate:
        signs = -signs
 
 
    if cfg.uniform:
        B = signs.astype(float)
 
    else:
        B = rng.uniform(low = 0.0, high = 1.0, size = (cfg.N, cfg.N)) * signs
 
 
    return _finalise(B, cfg, normalise = normalise)
 
 
 
 
def small_appropriate_interactions(cfg: Config, rng: np.random.Generator, S: np.ndarray = None) -> np.ndarray:
 
    B = appropriate_interactions(cfg, rng, S = S, normalise = False)
 
    # Magnitude is forced regardless of cfg.interactions_norm: a "small"
    # matrix that ignored its own magnitude would be meaningless.
    r = np.linalg.norm(B, ord = "fro")
 
 
    return B * (cfg.small_magnitude / r) if r > 0 else B
 
 
 
 
def noisy_appropriate_interactions(
        cfg: Config,
        rng: np.random.Generator,
        S: np.ndarray = None,
        normalise: bool = None
    ) -> np.ndarray:
 
    S = cfg.target if S is None else np.asarray(S, dtype = float)
 
    signs = np.outer(S, S)
    flip = rng.random(size = (cfg.N, cfg.N)) < cfg.flip_frac
 
 
    if cfg.symmetric_mask:
        flip = np.triu(flip, 1)
        flip = flip | flip.T
 
 
    signs = signs * np.where(flip, -1.0, 1.0)
 
 
    if cfg.uniform:
        B = signs.astype(float)
 
    else:
        B = rng.uniform(low = 0.0, high = 1.0, size = (cfg.N, cfg.N)) * signs
 
 
    return _finalise(B, cfg, normalise = normalise)
 
 
 
 
def modular_interactions(cfg: Config, rng: np.random.Generator, normalise: bool = None) -> np.ndarray:
    module_of = module_assignment(cfg.N, cfg.n_modules)
    same_module = module_of[:, None] == module_of[None, :]
 
    B = np.where(same_module, cfg.intra, cfg.inter)
    signs = rng.choice([-1.0, 1.0], size = (cfg.N, cfg.N), p = [1 - cfg.prob_pos_B, cfg.prob_pos_B])
 
 
    return _finalise(B * signs, cfg, normalise = normalise)
 
 
 
 
def modular_appropriate_interactions(
        cfg: Config,
        rng: np.random.Generator,
        S: np.ndarray = None,
        inappropriate: bool = False,
        normalise: bool = None
    ) -> np.ndarray:
    B = appropriate_interactions(cfg, rng, S = S, inappropriate = inappropriate, normalise = False)
 
    module_of = module_assignment(cfg.N, cfg.n_modules)
    same_module = module_of[:, None] == module_of[None, :]
 
    B = np.where(same_module, B, B * cfg.inter)
 
 
    return _finalise(B, cfg, normalise = normalise)
 
 
 
 
def module_assignment(N: int, n_modules: int) -> np.ndarray:
 
    return np.repeat(np.arange(n_modules), module_sizes(N, n_modules))
 
 
 
 
def module_sizes(N: int, n_modules: int) -> np.ndarray:
    base = N // n_modules
    sizes = np.full(n_modules, base, dtype = int)
    sizes[: N - base * n_modules] += 1
 
 
    return sizes
 
 
 
 
def adjust_interaction_magnitude(B: np.ndarray, cfg: Config, Y: float = None) -> np.ndarray:
    Y = cfg.Y if Y is None else Y
 
    B = masked_matrix(B, cfg.mask)
    r = np.linalg.norm(B, ord = "fro")
 
 
    return B * (Y / r) if r > 0 else np.zeros_like(B)
 
 
 
 
def normalise_interactions(B: np.ndarray, cfg: Config) -> np.ndarray:    
    
    return adjust_interaction_magnitude(B, cfg, Y = 1)