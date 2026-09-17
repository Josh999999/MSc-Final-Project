"""External Imports (Libraries and APIs)"""
import numpy as np


"""Local Imports"""
from Config import Config
from Plastic_Induction import plastic_search_return_wrapper
from R_round_induction import r_round_induction_return_wrapper
from GRN import evaluate_fitness



def handle_induction(   
        B: np.ndarray,     
        P: np.ndarray,
        G: np.ndarray,
        cfg: Config,
        rng: np.random.Generator,
        S: np.ndarray = None,
        limit_return: bool = False,
        AUC: float = 0.0
    ) -> dict[any]:

    history = {}
    induction_process_ = cfg.induction_process.lower().strip()


    if induction_process_ == "plastic":

        history = plastic_search_return_wrapper(B, P, cfg, rng, S, limit_return)

    elif induction_process_ == "r-round":

        history = r_round_induction_return_wrapper(B, P, G, cfg, rng, S, limit_return, AUC)

    else:
        history = {
            "B": B,
            "P": P,
            "F": evaluate_fitness(P, S, cfg),
        }


    return history