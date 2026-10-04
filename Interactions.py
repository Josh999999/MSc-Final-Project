"""External Imports (Libraries and APIs)"""
import numpy as np
 
 
"""Local Imports"""
from Config import Config
from GRN import DTYPE, masked_matrix
 

 
 
def _finalise(B: np.ndarray, cfg: Config, symmetric: bool = None) -> np.ndarray:
    """
    Every builder ends here: optional symmetrisation, the topology mask, and a
    rescaling to Frobenius norm cfg.Y. A builder therefore supplies the SIGN
    pattern and the relative magnitudes of the interactions, and Y alone sets
    the overall magnitude -- the same meaning Y has for the synthesised
    matrices of the tenet surfaces.
    """
    symmetric = cfg.symmetric_mask if symmetric is None else symmetric
 
    B = np.asarray(B, dtype = DTYPE).copy()
 
 
    if symmetric:
        B = (B + B.T) / 2.0
 
 
    B = masked_matrix(B, cfg.mask, inplace = True)
 
 
    return adjust_interaction_magnitude(B, cfg, Y = cfg.Y)
 
 
 
 
def random_interactions(cfg: Config, rng: np.random.Generator) -> np.ndarray:
    signs = rng.choice([-1.0, 1.0], size = (cfg.N, cfg.N), p = [1 - cfg.prob_pos_B, cfg.prob_pos_B])
 
 
    if cfg.uniform:
        B = signs
 
    else:
        B = rng.uniform(low = 0.0, high = 1.0, size = (cfg.N, cfg.N)) * signs
 
 
    return _finalise(B, cfg)
 
 
 
 
def appropriate_interactions(cfg: Config, rng: np.random.Generator, S: np.ndarray = None, inappropriate: bool = False) -> np.ndarray: 
    S = cfg.target if S is None else np.asarray(S, dtype = DTYPE)
 
    # Set value of signs in alignment with the target
    signs = np.outer(S, S)
 
 
    if inappropriate:
        signs = -signs
 
 
    if cfg.uniform:
        B = signs.astype(float)
 
    else:
        B = rng.uniform(low = 0.0, high = 1.0, size = (cfg.N, cfg.N)) * signs
 
 
    return _finalise(B, cfg)
 
 
 
 
def noisy_appropriate_interactions(cfg: Config, rng: np.random.Generator, S: np.ndarray = None, inappropriate: bool = False) -> np.ndarray: 
    S = cfg.target if S is None else np.asarray(S, dtype = DTYPE)
 
    signs = np.outer(S, S)
    flip = rng.random(size = (cfg.N, cfg.N)) < cfg.flip_frac
 
 
    if cfg.symmetric_mask:
        flip = np.triu(flip, 1)
        flip = flip | flip.T
 
 
    signs = signs * np.where(flip, -1.0, 1.0)


    if inappropriate:
        signs = -signs

 
    if cfg.uniform:
        B = signs.astype(float)
 
    else:
        B = rng.uniform(low = 0.0, high = 1.0, size = (cfg.N, cfg.N)) * signs
 
 
    return _finalise(B, cfg)
 
 
 
 
def modular_interactions(cfg: Config, rng: np.random.Generator) -> np.ndarray:
    module_of = module_assignment(cfg.N, cfg.n_modules)
    same_module = module_of[:, None] == module_of[None, :]
 
    B = np.where(same_module, cfg.intra, cfg.inter)
    signs = rng.choice([-1.0, 1.0], size = (cfg.N, cfg.N), p = [1 - cfg.prob_pos_B, cfg.prob_pos_B])
 
 
    return _finalise(B * signs, cfg)
 
 
 
 
def modular_appropriate_interactions(
        cfg: Config,
        rng: np.random.Generator,
        S: np.ndarray = None,
        inappropriate: bool = False
    ) -> np.ndarray:
    B = appropriate_interactions(cfg, rng, S = S, inappropriate = inappropriate)
 
    module_of = module_assignment(cfg.N, cfg.n_modules)
    same_module = module_of[:, None] == module_of[None, :]
 
    B = np.where(same_module, B, B * cfg.inter)
 
 
    return _finalise(B, cfg)
 
 
 
 
def module_assignment(N: int, n_modules: int) -> np.ndarray:
 
    return np.repeat(np.arange(n_modules), module_sizes(N, n_modules))
 
 
 
 
def module_sizes(N: int, n_modules: int) -> np.ndarray:
    r = N // n_modules
    sizes = np.full(n_modules, r, dtype = int)
    sizes[: N - r * n_modules] += 1
 
 
    return sizes
 
 
 
 
def adjust_interaction_magnitude(B: np.ndarray, cfg: Config, Y: float = None, inplace: bool = False) -> np.ndarray:
    Y = cfg.Y if Y is None else Y
 
    B = masked_matrix(B, cfg.mask, inplace = inplace)
    r = np.linalg.norm(B, ord = "fro")
 
 
    return B * (Y / r) if r > 0 else np.zeros_like(B)
 
