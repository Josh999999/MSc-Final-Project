"""External Imports (Libraries and APIs)"""
import numpy as np
 
 
"""Local Imports"""
from Config import Config
 
 
 
 
DTYPE = np.float64
 
 
 
def _cos(a: np.ndarray, b: np.ndarray, norm: bool = False) -> float:
    a = np.asarray(a, dtype = DTYPE).ravel()
    b = np.asarray(b, dtype = DTYPE).ravel()  
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
 
 
    if norm_a == 0 or norm_b == 0:
 
        return 0.0
 
 
    F = a @ b / (norm_a * norm_b)
 
 
    if norm:
        F = (F + 1) / 2
 
 
    return F
 
 
 
 
def tanh_sigma(x: np.ndarray) -> np.ndarray:
 
    return np.tanh(x)
 
 
 
 
def linear_sigma(x: np.ndarray) -> np.ndarray:
 
    return x
 
 
 
 
def sigmoid_sigma(x):
 
    return 1.0 / (1.0 + np.exp(-np.clip(x, -700, 700)))
 
 
 
 
SIGMOIDS = {"tanh": tanh_sigma, "linear": linear_sigma}
 
def resolve_sigmoid(sigmoid):
 
 
    if callable(sigmoid):
 
        return sigmoid
 
 
    return SIGMOIDS[str(sigmoid).strip().lower()]
 
 
 
 
def diag_mask(cfg: Config) -> np.ndarray:
    mask = np.ones((cfg.N, cfg.N), dtype = bool)
 
    np.fill_diagonal(mask, cfg.self_interaction)
 
 
    return mask
 
 
 
 
def sparse_topology(cfg: Config, rng: np.random.Generator) -> np.ndarray:
    mask = np.zeros((cfg.N, cfg.N), dtype = bool)
    all_genes = np.arange(cfg.N)
 
 
    for i in range(cfg.N):        
        others = rng.choice(all_genes[all_genes != i], size = cfg.K, replace = False)
        mask[i, others] = True
 
 
    if cfg.symmetric_mask:
 
        if cfg.mask_combine == "union":
            mask = mask | mask.T
 
        else:
            mask = mask & mask.T
 
 
    np.fill_diagonal(mask, cfg.self_interaction)
 
 
    return mask
 
 
 
 
def masked_matrix(B: np.ndarray, mask: np.ndarray, inplace: bool = True) -> np.ndarray:
    mask = np.asarray(mask).astype(bool)
 
 
    if B is None:
        B = np.zeros(mask.shape, dtype = DTYPE)
 
    else:
        B = np.asarray(B, dtype = DTYPE) if inplace else np.asarray(B, dtype = DTYPE).copy()
 
 
    return np.where(mask, B, 0.0)
 
 
 
 
def develop(G: np.ndarray, B: np.ndarray, cfg: Config, T: int) -> np.ndarray:
    sigmoid = resolve_sigmoid(cfg.sigmoid)
    G = np.asarray(G, dtype = DTYPE)
 
 
    if cfg.development == "bounded":
        P = np.zeros(cfg.N, dtype = DTYPE)
 
        for _ in range(T):
            P += cfg.t1 * (sigmoid(B @ P + G) - P)
 
 
        return P
 
 
    P = G.copy()
 
    for _ in range(T):
        P += cfg.t1 * sigmoid(B @ P) - cfg.t2 * P
 
 
    return P
 
 
def handle_develop(G: np.ndarray, B: np.ndarray, cfg: Config, induction: bool = False) -> np.ndarray:
    
    if induction:
        
        return develop(G, B, cfg, cfg.r_T)
 
    
    return develop(G, B, cfg, cfg.T)
 
 
 
 
def fitness(P: np.ndarray, S: np.ndarray, norm: bool = False) -> float:
    F = P @ S
 
 
    if norm:
        N = len(S)
        F = (F + N) / (2 * N)
 
 
    return F
 
 
 
 
def evaluate_fitness(P: np.ndarray, S: np.ndarray, cfg: Config) -> float:
 
    if cfg.fitness_type == "cosine":
 
        return _cos(P, S, norm = cfg.normalise_fitness)
 
    elif cfg.fitness_type == "standard":
 
        return fitness(P, S, cfg.normalise_fitness)
 
 
    return P @ S
 
 
 
 
def mutate_profile(G: np.ndarray, cfg: Config, rng: np.random.Generator) -> np.ndarray:
    G_mut = np.asarray(G, dtype = DTYPE).copy()
 
    idx = rng.integers(low = 0, high = cfg.N, size = cfg.n_mut_G)
    h = G_mut[idx] + rng.uniform(low = -cfg.u1, high = cfg.u1, size = cfg.n_mut_G)
    G_mut[idx] = np.clip(h, -1.0, 1.0)
 
 
    return G_mut
 
 
 
 
def hebbian_interactions(cfg: Config, S: np.ndarray = None) -> np.ndarray:
    S = cfg.targets if S is None else S
    S = np.atleast_2d(np.asarray(S, dtype = DTYPE))
 
    H = cfg.lr * (S.T @ S)
    H = masked_matrix(H, cfg.mask)
 
 
    return H * cfg.Y if cfg.Y > 0 else H
 
 
 
 
def hamming_dist(a: np.ndarray, b: np.ndarray) -> int:
 
    return int(np.sum(np.sign(np.asarray(a, dtype = DTYPE).ravel()) != np.sign(np.asarray(b, dtype = DTYPE).ravel())))
 
 
 
 
def create_mask(cfg: Config, rng: np.random.Generator) -> np.ndarray:
 
    if cfg.sparse_interactions:
 
        return sparse_topology(cfg, rng)
 
 
    return diag_mask(cfg)
 