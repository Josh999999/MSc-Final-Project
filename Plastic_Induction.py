"""External Imports (Libraries and APIs)"""
import numpy as np
 
 
"""Local Imports"""
from Config import Config
from GRN import DTYPE, evaluate_fitness, sigmoid_sigma
from Mutations import compute_mutation
from Interactions import adjust_interaction_magnitude
 
 
 
 
def energy(P: np.ndarray, B: np.ndarray, cfg: Config, normalise: bool = True, normalise_interactions: bool = False) -> float:
 
    # Determine the interaction matrix
    if normalise_interactions:
        B = adjust_interaction_magnitude(B, cfg = cfg, Y = 1.0, inplace = False)
 
 
    P = np.asarray(P, dtype = DTYPE)
    q = float(-0.5 * P @ (B @ P))
 
 
    if not normalise:
 
        return q
 
 
    n = float(P @ P)
 
 
    return q / n if n > 0 else 0.0
 
 
 
 
def differential_energy(P: np.ndarray, P_try: np.ndarray, B: np.ndarray, cfg: Config, normalise: bool = True, normalise_interactions: bool = False) -> float:
 
    # Determine the interaction matrix
    if normalise_interactions:
        B = adjust_interaction_magnitude(B, cfg = cfg, Y = 1.0, inplace = False)
 
        
    P = np.asarray(P, dtype = DTYPE)
    P_try = np.asarray(P_try, dtype = DTYPE)
    dP = P_try - P
    q = float(-0.5 * dP @ (B @ dP))
 
 
    if not normalise:
 
        return q
 
 
    n = float(dP @ dP)
 
 
    return q / n if n > 0 else 0.0
 
 
 
 
def candidate_energies(h: np.ndarray, c: float, P: np.ndarray = None, normalise: bool = True) -> np.ndarray:
    h = np.asarray(h, dtype = DTYPE)
 
 
    if not normalise:
 
        return np.concatenate((-c * h, c * h))
 
 
    P = np.asarray(P, dtype = DTYPE)
    q = float(P @ h)
    n = float(P @ P)
 
    E0 = -0.5 * q / n if n > 0 else 0.0
 
    d_pos = n + 2.0 * c * P + c * c
    E_pos = -0.5 * (q + 2.0 * c * h) / np.where(d_pos > 0, d_pos, 1.0)
 
    d_neg = n - 2.0 * c * P + c * c
    E_neg = -0.5 * (q - 2.0 * c * h) / np.where(d_neg > 0, d_neg, 1.0)
 
 
    return np.concatenate((E_pos - E0, E_neg - E0))
 
 
 
 
def adaptive_tau(dE_pool: np.ndarray, cfg: Config) -> float: 
 
    return max(cfg.c_tau * float(np.std(dE_pool)), cfg.tau_floor)
 
 
 
 
def phenotype_alignment(P: np.ndarray, B: np.ndarray, cfg: Config) -> float:
    P = np.asarray(P, dtype = DTYPE)
    B = np.asarray(B, dtype = DTYPE)
 
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
    S = cfg.target if S is None else np.asarray(S, dtype = DTYPE)
    P = np.asarray(P, dtype = DTYPE).copy()        
 
    F = evaluate_fitness(P, S, cfg)
    A = phenotype_alignment(P, B, cfg) 
 
    curve = [F]
    align_curve = [A]
    accepted = 0
    accepted_alignments = []
 
    tau = None
 

    for _ in range(cfg.M):      
 
        # Compute current phenotype mutation
        P_try = compute_mutation(P, cfg, rng)
 
 
        # Calculate the energy differential
        dE = 0
 
        if cfg.energy_type == "standard":
            E_try = energy(P_try, B, cfg, normalise = cfg.normalise_energy, normalise_interactions = cfg.energy_normalise_interactions)
            E = energy(P, B, cfg, normalise = cfg.normalise_energy, normalise_interactions = cfg.energy_normalise_interactions)
            dE = E_try - E
            
        elif cfg.energy_type == "differential":
            dE = differential_energy(P, P_try, B, cfg, normalise = cfg.normalise_energy, normalise_interactions = cfg.energy_normalise_interactions)
 
        
        # Alignment of the CANDIDATE: measuring P here records the pre-move state, so the curve lags and align_end misses the last accepted move.
        alignment = phenotype_alignment(P_try, B, cfg) 
        F_try = evaluate_fitness(P_try, S, cfg)
 
 
        # Apply the energy gates with fitness based selection
        if cfg.energy_gate in ["or", "and", "harsh"]:
 
            # tau depends only on P, so recompute it only when P has moved.
            if tau is None:
                h = B @ P
 
                step = cfg.c * (np.linalg.norm(P) / np.sqrt(cfg.N)) if cfg.relative_mutation else cfg.c
                step = max(step, 1e-12)
 
                # Pool statistics over every single-gene move of the nominal size.
                dE_pool = candidate_energies(h, step, P, cfg.normalise_energy)
                tau = adaptive_tau(dE_pool, cfg)
 
 
            # Calculate the energy probability
            w = sigmoid_sigma(-dE / tau)
 
 
            # Apply the relevant energy gates
            if cfg.energy_gate == "or":        
                accept = (F_try > F) or (rng.random() < w)
        
            elif cfg.energy_gate == "and":                
                accept = (F_try > F) and (rng.random() < w)
            
            elif cfg.energy_gate == "harsh":
                slack = rng.uniform(0, cfg.slack_limit) 
                accept = (F_try > F) and (cfg.energy_limit - slack < w)
    
        elif cfg.energy_gate == "deterministic":    
            accept = (F_try > F) and (dE <= 0)
        
        else:
            accept = (F_try > F) and (dE <= 0)
        
 
        # Set new values under acceptance criteria
        if accept:
            P, F = P_try, F_try
            A = alignment
            accepted += 1
            accepted_alignments.append(alignment)
 
            # P has moved, so the cached h / tau are stale.
            tau = None
 
 
        curve.append(F)
        align_curve.append(A)
 
 
    return B, P, F, curve, align_curve, accepted_alignments, accepted
 
 
 
 
def plastic_search_return_wrapper(B: np.ndarray, P: np.ndarray, cfg: Config, rng: np.random.Generator, S: np.ndarray = None, limit_return: bool = False) -> dict:
 
    B, P, F, curve, align_curve, accepted_alignments, accepted = plastic_search(B, P, cfg, rng, S)
 
 
    # Used for the induction process
    if limit_return:
    
        return {
            "B": B,
            "P": P,
            "F": float(F),
            "auc_inner": float(np.mean(curve)) if len(curve) else 0.0
        }
 
 
    # Used for analysis
    return {
        "B": B,
        "P": P,
        "F": float(F),
        "F_change": float(curve[-1] - curve[0]),
        "avg_accept_A": float(np.mean(accepted_alignments)) if len(accepted_alignments) else 0.0,
        "std_accept_A": float(np.std(accepted_alignments)) if len(accepted_alignments) else 0.0, 
        "auc_inner": float(np.mean(curve)) if len(curve) else 0.0,                 # area under the ABSOLUTE curve
        "acceptance_rate": float(accepted / cfg.M) if accepted and cfg.M else 0.0,        # collapse signature (5.6)
        "curve": curve,
        "align_curve": align_curve,
        "align_start": float(align_curve[0]),
        "align_end": float(align_curve[-1]),
        "align_change": float(align_curve[-1] - align_curve[0]),
    }