"""External Imports (Libraries and APIs)"""
import numpy as np
 
 
"""Local Imports"""
from Config import Config
from GRN import evaluate_fitness, sigmoid_sigma, _cos
from Mutations import compute_mutation
 
 
 
 
def energy(P: np.ndarray, B: np.ndarray, normalise: bool = True) -> float:
    P = np.asarray(P, dtype = float)
    q = float(-0.5 * P @ (B @ P))
 
 
    if not normalise:
 
        return q
 
 
    n = float(P @ P)
 
 
    return q / n if n > 0 else 0.0
 
 
 
 
def candidate_energies(
        h: np.ndarray,
        c: float,
        P: np.ndarray = None,
        normalise: bool = True
    ) -> np.ndarray:
    
    h = np.asarray(h, dtype = float)
 
 
    if not normalise:
 
        return -c * h
 
 
    P = np.asarray(P, dtype = float)
    q = float(P @ h)
    n = float(P @ P)
 
    E0 = -0.5 * q / n if n > 0 else 0.0
    E1 = -0.5 * (q + 2.0 * c * h) / (n + 2.0 * c * P + c * c)
 
 
    return E1 - E0
 
 
 
 
def adaptive_tau(dE_pool: np.ndarray, cfg: Config) -> float:
 
 
    return max(cfg.c_tau * float(np.std(dE_pool)), cfg.tau_floor)
 
 
 
 
def energy_gate(F_try: float, F: float, w: float, dE: float, cfg: Config, rng: np.random.Generator) -> bool:
 
    if cfg.use_energy:
 
        if cfg.energy_gate == "or":
 
            return (F_try >= F) or (rng.random() <= w)
 
        elif cfg.energy_gate == "and":
            
            return (F_try >= F) and (rng.random() <= w)
        
        elif cfg.energy_gate == "harsh":
            slack = rng.uniform(0, cfg.limit_slack)

 
            return (F_try >= F) and (cfg.energy_limit - slack <= w)

        elif cfg.energy_gate == "deterministic":

            return (F_try >= F) and (dE <= 0)
 
        else:
 
            return F_try * w >= F
 
 
    return F_try >= F
 
 
 
 
def phenotype_alignment(P: np.ndarray, B: np.ndarray, cfg: Config) -> float:
    """cos(P (x) P, B) over masked entries: how strongly B encodes P."""
    P = np.asarray(P, dtype = float)
    B = np.asarray(B, dtype = float)
 
    PP = np.outer(P, P)
 
    num = float(PP[cfg.mask] @ B[cfg.mask])
    den = float(np.linalg.norm(PP[cfg.mask]) * np.linalg.norm(B[cfg.mask]))
 
 
    return num / den if den > 0 else 0.0
 
 
 
 
def plastic_search(
        B: np.ndarray,
        P: np.ndarray,
        cfg: Config,
        rng: np.random.Generator,
        S: np.ndarray = None,
    ) -> dict:
    S = cfg.target if S is None else np.asarray(S, dtype = float)
    P = np.asarray(P, dtype = float).copy()        
 
    F = evaluate_fitness(P, S, cfg)
 
    curve = [F]
    align_curve = [phenotype_alignment(P, B, cfg)]
    accepted = 0
    dE_mean, dE_std, taus = [], [], []
    accepted_energies = []
    accepted_alignments = []
 
 
    for _ in range(cfg.M):
 
        h = B @ P                                   
        
        step = cfg.c * (np.linalg.norm(P) / np.sqrt(cfg.N)) if cfg.relative_mutation else cfg.c
        step = max(step, 1e-12)
 
        # Pool statistics over every single-gene move of the nominal size.
        dE_pool = candidate_energies(h, step, P, cfg.normalise_energy)
        tau = adaptive_tau(dE_pool, cfg)
 
        dE_mean.append(float(np.mean(dE_pool)))
        dE_std.append(float(np.std(dE_pool)))
        taus.append(tau)
 
 
        P_try, _ = compute_mutation(P, cfg, rng)
 
        dE = energy(P_try, B, cfg.normalise_energy) - energy(P, B, cfg.normalise_energy)
 
        # Alignment of the CANDIDATE: measuring P here records the pre-move
        # state, so the curve lags and align_end misses the last accepted move.
        alignment = phenotype_alignment(P_try, B, cfg)
 
        F_try = evaluate_fitness(P_try, S, cfg)
        w = sigmoid_sigma(-dE / tau)
 
 
        # Apply the Fitness gate with Energy
        accept = energy_gate(F_try, F, w, dE, cfg, rng)
 
 
        if accept:
            P, F = P_try, F_try
            A = alignment
            accepted += 1
            accepted_energies.append(dE)
            accepted_alignments.append(alignment)
 
        else:
            A = align_curve[-1]         # rejected: the held phenotype is unchanged
 
 
        curve.append(F)
        align_curve.append(A)

    
    curve = np.asarray(curve, dtype = float)
    align_curve = np.asarray(align_curve, dtype = float)
    accepted_energies = np.asarray(accepted_energies, dtype = float)
    accepted_alignments = np.asarray(accepted_alignments, dtype = float)
 

    return B, P, F, curve, align_curve, accepted_energies, accepted_alignments, accepted




def plastic_search_return_wrapper(
        B: np.ndarray,
        P: np.ndarray,
        cfg: Config,
        rng: np.random.Generator,
        S: np.ndarray = None,
        limit_return: bool = False
    ) -> dict:

    B, P, F, curve, align_curve, accepted_energies, accepted_alignments, accepted = plastic_search(B, P, cfg, rng, S)


    if limit_return:
    
        return {
            "B": B,
            "P": P,
            "F": F,
            "auc": np.mean(curve) if curve.size else 0.0,                 # area under the ABSOLUTE curve
            "F_change": curve[-1] - curve[0],
            "align_change": align_curve[-1] - align_curve[0],
            "curve": curve,
        }


    return {
        "B": B,
        "P": P,
        "F": F,
        "F_change": curve[-1] - curve[0],
        "avg_accept_E": np.mean(accepted_energies) if accepted_energies.size else 0.0, 
        "std_accept_E": np.std(accepted_energies) if accepted_energies.size else 0.0, 
        "avg_accept_A": np.mean(accepted_alignments) if accepted_alignments.size else 0.0,
        "std_accept_A": np.std(accepted_alignments) if accepted_alignments.size else 0.0, 
        "auc": np.mean(curve) if curve.size else 0.0,                 # area under the ABSOLUTE curve
        "acceptance_rate": accepted / cfg.M if accepted and cfg.M else 0.0,        # collapse signature (5.6)
        "curve": curve,
        "align_curve": align_curve,
        "align_start": align_curve[0],
        "align_end": align_curve[-1],
        "align_change": align_curve[-1] - align_curve[0],
    }