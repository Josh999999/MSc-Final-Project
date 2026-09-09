"""External Imports (Libraries and APIs)"""
import numpy as np


"""Local Imports"""
from Config import Config
from GRN import handle_develop
from Plastic_Induction import plastic_search
from Interactions import normalise_interactions




def r_round_induction(B: np.ndarray, P: np.ndarray, G: np.ndarray, cfg: Config, rng: np.random.Generator, S: np.ndarray = None) -> np.ndarray:    
    B = np.asarray(B, dtype = float).copy() if not cfg.induction_inplace else np.asarray(B, dtype = float)
    S = cfg.target if S is None else np.asarray(S, dtype = float)
    P = np.asarray(P, dtype = float)
    
    curve = []
    
    R = cfg.rounds
    lr = cfg.eta
    
    
    # For each round (R)
    for _ in range(0, R):
        
        # Run placticity
        search = plastic_search(B, P, cfg, rng, S)
        F = search['auc']
        P_ = search['P']

        curve.append(F)
        
        
        # Update the matrix using a contrastive update
        dPP_ = np.outer(P_, P_) - np.outer(P, P)
        
        B += lr * dPP_


        # Re-Normalise the matrix (after learning)
        if cfg.interactions_norm:
            B = normalise_interactions(B, cfg)

        
        # Redevelop the Genotype under the new interaction matrix
        P = handle_develop(G, B, cfg, induction = True)


    curve = np.asarray(curve, dtype = float)


    return {
        "B": B,
        "P": P,
        "F": float(F),
        "F_change": float(curve[-1] - curve[0]),
        "curve": curve,
        "auc": float(curve.mean()),     
    }