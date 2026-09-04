"""External Imports (Libraries and APIs)"""
import numpy as np


"""Local Imports"""
from Config import Config, DEVELOPMENT_SIGMOIDS




def _cos(a: np.ndarray, b: np.ndarray, norm: bool = False) -> float:
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()  
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




_SIGMOIDS = {"tanh": tanh_sigma, "linear": linear_sigma}
assert set(_SIGMOIDS) == set(DEVELOPMENT_SIGMOIDS)


def resolve_sigmoid(sigmoid):


    if callable(sigmoid):

        return sigmoid


    return _SIGMOIDS[str(sigmoid).strip().lower()]




def diag_mask(N: int) -> np.ndarray:
    mask = np.ones((N, N), dtype = bool)
    np.fill_diagonal(mask, False)


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




def mask_indices(mask: np.ndarray) -> np.ndarray:


    return np.flatnonzero(np.asarray(mask).astype(bool).ravel())




def masked_matrix(B: np.ndarray, mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask).astype(bool)
    B = np.zeros(mask.shape, dtype = float) if B is None else np.asarray(B, dtype = float).copy()


    return np.where(mask, B, 0.0)




def develop(G: np.ndarray, B: np.ndarray, cfg: Config, T: int) -> np.ndarray:
    sigmoid = resolve_sigmoid(cfg.sigmoid)

    P = np.asarray(G, dtype = float).copy()


    for _ in range(T):
        P += cfg.t1 * sigmoid(B @ P) - cfg.t2 * P


    return P




def handle_develop(G: np.ndarray, B: np.ndarray, cfg: Config, induction: bool = False) -> np.ndarray:
    
    if induction:
        
        return develop(G, B, cfg, cfg.r_T)

    
    return develop(G, B, cfg, cfg.T)




def evaluate_fitness(P: np.ndarray, S: np.ndarray, cfg: Config) -> float:


    return _cos(P, S, norm = cfg.fitness_norm)




def random_profiles(n: int, cfg: Config, rng: np.random.Generator) -> np.ndarray:


    return rng.uniform(low = -1.0, high = 1.0, size = (n, cfg.N))




def mutate_profile(G: np.ndarray, cfg: Config, rng: np.random.Generator) -> np.ndarray:
    G_mut = np.asarray(G, dtype = float).copy()

    idx = rng.integers(low = 0, high = cfg.N, size = cfg.n_mut_G)
    h = G_mut[idx] + rng.uniform(low = -cfg.u1, high = cfg.u1, size = cfg.n_mut_G)
    G_mut[idx] = np.clip(h, -1.0, 1.0)


    return G_mut




def mutate_interactions(B: np.ndarray, cfg: Config, rng: np.random.Generator) -> np.ndarray:
    B_mut = np.asarray(B, dtype = float).copy()


    if rng.random() >= cfg.prob_mut_B:

        return B_mut


    allowed = cfg.allowed()


    if allowed.size <= 0:

        return B_mut


    flat = allowed[rng.integers(0, allowed.size, size = cfg.n_mut_B)]
    rows, cols = np.unravel_index(flat, (cfg.N, cfg.N))
    deltas = rng.uniform(low = -cfg.u2, high = cfg.u2, size = cfg.n_mut_B)


    for i, j, d in zip(rows, cols, deltas):
        B_mut[i, j] += d


        if cfg.symmetric_B and i != j:
            B_mut[j, i] += d


    return B_mut




def hebbian_interactions(cfg: Config, S: np.ndarray = None) -> np.ndarray:
    S = cfg.targets if S is None else S
    S = np.atleast_2d(np.asarray(S, dtype = float))

    H = cfg.lr * (S.T @ S)

    # No self-interaction in development or in the energy calculation, so the
    # Hebbian matrix must match or it is not comparable to an evolved B.
    np.fill_diagonal(H, 0.0)
    H = masked_matrix(H, cfg.mask)

    Y = abs(cfg.Y)


    return H / Y if Y > 0 else H




def hamming_dist(a: np.ndarray, b: np.ndarray) -> int:


    return int(np.sum(np.sign(np.asarray(a).ravel()) != np.sign(np.asarray(b).ravel())))



