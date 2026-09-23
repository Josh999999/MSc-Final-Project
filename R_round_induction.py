"""External Imports (Libraries and APIs)"""
import numpy as np


"""Local Imports"""
from Config import Config
from GRN import DTYPE, handle_develop, evaluate_fitness
from Plastic_Induction import plastic_search_return_wrapper, phenotype_alignment
from Interactions import normalise_interactions




def r_round_induction(
        B: np.ndarray, 
        P: np.ndarray, 
        G: np.ndarray, 
        cfg: Config, 
        rng: np.random.Generator, 
        S: np.ndarray = None, 
        AUC: float = -1
    ) -> np.ndarray:    
    B = np.asarray(B, dtype = DTYPE).copy()
    S = cfg.target if S is None else np.asarray(S, dtype = DTYPE)
    P = np.asarray(P, dtype = DTYPE)
    
    F = evaluate_fitness(P, S, cfg)
    A = phenotype_alignment(P, B, cfg) 

    if AUC == -1:
        search = plastic_search_return_wrapper(B, P, cfg, rng, S, limit_return = True)
        AUC = search["auc_inner"]
    
    inner_curve = [AUC]
    outer_curve = [F]
    align_curve = [A]
    
    R = cfg.rounds
    lr = cfg.eta
    
    
    # For each round (R)
    for _ in range(0, R):
        
        # Run placticity
        search = plastic_search_return_wrapper(B, P, cfg, rng, S, limit_return = True)
        AUC = search["auc_inner"]
        P_ = search['P']

        inner_curve.append(AUC)
        
        
        # Update the matrix using a contrastive update
        dPP_ = np.outer(P_, P_) - np.outer(P, P)


        # Determine how the interactions are updated
        mask = np.asarray(cfg.mask.copy(), dtype = bool)
        dB = 0

        if cfg.induction_interactions == "inclusive":
            dB = lr * dPP_ * mask

        elif cfg.induction_interactions == "exclusive":
            dB = lr * dPP_ * ~mask

        elif cfg.induction_interactions == "all":
            dB = lr * dPP_

        else:
            dB = lr * dPP_

        B += dB
 

        # Re-Normalise the matrix (after learning)
        if cfg.normalise_interactions:
            B = normalise_interactions(B, cfg)

        
        # Redevelop the Genotype under the new interaction matrix
        if cfg.relax:
            # Relaxing the genotype is ordinary development: use T, not r_T.
            P = handle_develop(G, B, cfg, induction = False)
            F = evaluate_fitness(P, S, cfg)
            outer_curve.append(F)

        else:
            P = P_
            F = search['F']
            outer_curve.append(F)


        # Track alignment of the Phenotypes induction produces
        A = phenotype_alignment(P, B, cfg)
        align_curve.append(A)
        

    inner_curve = np.asarray(inner_curve, dtype = DTYPE)
    outer_curve = np.asarray(outer_curve, dtype = DTYPE)
    align_curve = np.asarray(align_curve, dtype = DTYPE)


    return B, P, F, inner_curve, outer_curve, align_curve




def r_round_induction_return_wrapper(
        B: np.ndarray, 
        P: np.ndarray, 
        G: np.ndarray, 
        cfg: Config, 
        rng: np.random.Generator, 
        S: np.ndarray = None, 
        limit_return: bool = False, 
        AUC: float = 0.0
    ) -> dict:
    B, P, F, inner_curve, outer_curve, align_curve = r_round_induction(B, P, G, cfg, rng, S, AUC)


    if limit_return:
    
        return {
            "B": B,
            "P": P,
            "F": float(F),
            "auc_inner": np.mean(inner_curve) if inner_curve.size else 0.0,
        }


    return {
        "B": B,
        "P": P,
        "F": F,
        "F_change_inner": inner_curve[-1] - inner_curve[0],
        "F_change_outer": outer_curve[-1] - outer_curve[0],        
        "auc_inner": np.mean(inner_curve) if inner_curve.size else 0.0,                 # area under the ABSOLUTE curve
        "auc_outer": np.mean(outer_curve) if outer_curve.size else 0.0,
        "inner_curve": inner_curve,
        "outer_curve": outer_curve,
        "align_curve": align_curve,
        "align_change": align_curve[-1] - align_curve[0],
        "F_first_inner": inner_curve[0],
        "F_final_inner": inner_curve[-1],
        "F_first_outer": outer_curve[0],
        "F_final_outer": outer_curve[-1],
    }