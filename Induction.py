"""External Imports (Libraries and APIs)"""
import numpy as np


"""Local Imports"""
from Config import Config
from Plastic_Induction import plastic_search
from R_round_induction import r_round_induction
from GRN import evaluate_fitness



def handle_induction(   
        B: np.ndarray,     
        P: np.ndarray,
        G: np.ndarray,
        cfg: Config,
        rng: np.random.Generator,
        S: np.ndarray = None
    ) -> dict[any]:

    """
    Induction methods should, at the least, return a dictionary of the form
    {
        "B": B,
        "P": P,
        "F": float(F),
    }
    B - The altered interaction matrix
    P - The phenotype produced by the induction process
    F - The fitness the induction method conveys given the interaction matrix and genotype pair
    """

    history = {}
    induction_process_ = cfg.induction_process.lower().strip()


    if induction_process_ == "plastic":

        history = plastic_search(B, P, cfg, rng, S)

    elif induction_process_ == "r-round":

        history = r_round_induction(B, P, G, cfg, rng, S)

    elif induction_process_ == "hopfield":

        """!-- TODO --!"""
        history = {
            "B": B,
            "P": P,
            "F": evaluate_fitness(P, S, cfg),
        }

    else:
        history = {
            "B": B,
            "P": P,
            "F": evaluate_fitness(P, S, cfg),
        }


    return history