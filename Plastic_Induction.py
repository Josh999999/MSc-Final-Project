"""External Imports (Libraries and APIs)"""
import numpy as np


"""Local Imports"""
from Config import Config
from GRN import evaluate_fitness, sigmoid_sigma
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




def energy_gate(F_try: float, F: float, w: float, cfg: Config, rng: np.random.Generator) -> bool:

    if cfg.use_energy:

        if cfg.energy_gate == "or":

            return (F_try >= F) or (rng.random() < w)

        elif cfg.energy_gate == "and":
            
            return (F_try >= F) and (rng.random() < w)
        
        elif cfg.energy_gate == "harsh":

            return (F_try >= F) and (cfg.energy_limit < w)

        else:

            return F_try * w >= F


    return F_try >= F




def plastic_search(
        B: np.ndarray,
        P: np.ndarray,
        cfg: Config,
        rng: np.random.Generator,
        S: np.ndarray = None
    ) -> dict:
    S = cfg.target if S is None else np.asarray(S, dtype = float)
    P = np.asarray(P, dtype = float).copy()        

    F = evaluate_fitness(P, S, cfg)

    curve = [F]
    accepted = 0
    dE_mean, dE_std, taus = [], [], []
    accepted_fitnesses = []
    accepted_energies = []
    accepted_energies_ = []


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

        F_try = evaluate_fitness(P_try, S, cfg)
        w = sigmoid_sigma(-dE / tau)


        # Apply the Fitness gate with Energy
        accept = energy_gate(F_try, F, w, cfg, rng)


        if accept:
            P, F = P_try, F_try
            accepted += 1
            accepted_fitnesses.append(F_try)
            accepted_energies.append(dE)
            accepted_energies_.append(w)


        curve.append(F)


    return {
        "B": B,
        "P": P,
        "F": float(F),
        "avg_accept_F": float(np.mean(accepted_fitnesses)) if accepted_fitnesses else float(0),
        "avg_accept_E": float(np.mean(accepted_energies)) if accepted_energies else float(0), 
        "std_accept_E": float(np.std(accepted_energies)) if accepted_energies else float(0), 
        "avg_accept_w": float(np.mean(accepted_energies_)) if accepted_energies_ else float(0), 
        "std_accept_w": float(np.std(accepted_energies_)) if accepted_energies_ else float(0), 
        "auc": float(np.mean(curve)) if curve else float(0),                 # area under the ABSOLUTE curve
        "acceptance_rate": float(accepted / cfg.M) if accepted and cfg.M else float(0),        # collapse signature (5.6)
        "dE_mean": np.array(dE_mean),               # log these: drift is the
        "dE_std": np.array(dE_std),                 # early warning signal
        "tau": np.array(taus),
    }
