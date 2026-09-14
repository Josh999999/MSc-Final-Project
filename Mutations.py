"""External Imports (Libraries and APIs)"""
import numpy as np


"""Local Imports"""
from Config import Config




def single_gene_mutation(P: np.ndarray, cfg: Config, rng: np.random.Generator) -> tuple:
    P = np.asarray(P, dtype = np.float32).copy()
    
    idx = rng.integers(cfg.N)

    magnitude = float(rng.uniform(low = 0.0, high = cfg.c))
    sign = float(rng.choice([-1.0, 1.0]))


    if cfg.mutation_operation == "additive":
        P[idx] += magnitude * sign

    else:
        P[idx] *= 1.0 + magnitude * sign


    return P




def phenotype_mutation(P: np.ndarray, cfg: Config, rng: np.random.Generator) -> tuple:
    P = np.asarray(P, dtype = np.float32).copy()
    low_c = 0.1 * cfg.c 

    magnitude = rng.uniform(low = low_c, high = cfg.c, size = cfg.N)
    sign = rng.choice([-1.0, 1.0], size = cfg.N)


    if cfg.mutation_operation == "additive":
        P += magnitude * sign

    else:
        P *= 1.0 + magnitude * sign


    return P




def perturbation(P: np.ndarray, cfg: Config, rng: np.random.Generator) -> tuple:
    P = np.asarray(P, dtype = np.float32).copy()

    P += rng.uniform(low = -cfg.c, high = cfg.c, size = cfg.N)


    return P




_MUTATIONS = {"single-gene": single_gene_mutation, "phenotype": phenotype_mutation, "perturbation": perturbation}

def compute_mutation(P: np.ndarray, cfg: Config, rng: np.random.Generator) -> tuple:
    
    
    return _MUTATIONS[cfg.mutation_type.strip().lower()](P, cfg, rng)