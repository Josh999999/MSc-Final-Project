"""External Imports (Libraries and APIs)"""
import numpy as np


"""Local Imports"""
from Config import Config
from GRN import handle_develop, evaluate_fitness
from Plastic_Induction import plastic_search_return_wrapper, phenotype_alignment
from Interactions import normalise_interactions




def r_round_induction(B: np.ndarray, P: np.ndarray, G: np.ndarray, cfg: Config, rng: np.random.Generator, S: np.ndarray = None) -> np.ndarray:    
    B = np.asarray(B, dtype = float).copy() if not cfg.induction_inplace else np.asarray(B, dtype = float)
    S = cfg.target if S is None else np.asarray(S, dtype = float)
    P = np.asarray(P, dtype = float)
    
    inner_curve = []
    outer_curve = []
    align_curve = [phenotype_alignment(P, B, cfg)]
    
    R = cfg.rounds
    lr = cfg.eta
    
    
    # For each round (R)
    for _ in range(0, R):
        
        # Run placticity
        search = plastic_search_return_wrapper(B, P, cfg, rng, S, limit_return = True)
        F_ = search['auc']
        P_ = search['P']

        inner_curve.append(F_)
        
        
        # Update the matrix using a contrastive update
        dPP_ = np.outer(P_, P_) - np.outer(P, P)
        
        B += lr * dPP_


        # Re-Normalise the matrix (after learning)
        if cfg.interactions_norm:
            B = normalise_interactions(B, cfg)

        
        # Redevelop the Genotype under the new interaction matrix
        P = handle_develop(G, B, cfg, induction = True)
        F = evaluate_fitness(P, S, cfg)
        outer_curve.append(F)


        # Track alignment of the Phenotypes induction produces
        alignment = phenotype_alignment(P, B, cfg)
        align_curve.append(alignment)
        

    inner_curve = np.asarray(inner_curve, dtype = float)
    outer_curve = np.asarray(outer_curve, dtype = float)
    align_curve = np.asarray(align_curve, dtype = float)


    return B, P, F, inner_curve, outer_curve, align_curve




def r_round_induction_return_wrapper(
        B: np.ndarray,
        P: np.ndarray,
        G: np.ndarray,
        cfg: Config,
        rng: np.random.Generator,
        S: np.ndarray = None,
        limit_return: bool = False
    ) -> dict:

    B, P, F, inner_curve, outer_curve, align_curve = r_round_induction(B, P, G, cfg, rng, S)


    if limit_return:
    
        return {
            "B": B,
            "P": P,
            "F": F,
            "auc_inner": np.mean(inner_curve) if inner_curve.size else 0.0,                 # area under the ABSOLUTE curve
            "F_change_inner": inner_curve[-1] - inner_curve[0],
            "align_change": align_curve[-1] - align_curve[0],
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
    }