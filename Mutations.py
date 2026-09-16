"""External Imports (Libraries and APIs)"""
import numpy as np


"""Local Imports"""
from Config import Config
from GRN import DTYPE




def single_gene_mutation(P: np.ndarray, cfg: Config, rng: np.random.Generator) -> tuple:
    P = np.asarray(P, dtype = DTYPE).copy()
    
    idx = rng.integers(cfg.N)

    magnitude = float(rng.uniform(low = 0.0, high = cfg.c))
    sign = float(rng.choice([-1.0, 1.0]))


    P[idx] += magnitude * sign


    return P




def phenotype_mutation(P: np.ndarray, cfg: Config, rng: np.random.Generator) -> tuple:
    P = np.asarray(P, dtype = DTYPE).copy()
    low_c = 0.1 * cfg.c 

    magnitude = rng.uniform(low = low_c, high = cfg.c, size = cfg.N)
    sign = rng.choice([-1.0, 1.0], size = cfg.N)

    P += magnitude * sign


    return P




MUTATION_TYPES = {"single-gene": single_gene_mutation, "phenotype": phenotype_mutation}

def compute_mutation(P: np.ndarray, cfg: Config, rng: np.random.Generator) -> tuple: 
    
    return MUTATION_TYPES[cfg.mutation_type.strip().lower()](P, cfg, rng)