"""External Imports (Libraries and APIs)"""
import numpy as np


"""Local Imports"""
from Config import Config




def _prepare(P: np.ndarray, cfg: Config) -> np.ndarray:
    P = np.asarray(P, dtype = float)


    return P if cfg.mutate_inplace else P.copy()




def single_gene_mutation(P: np.ndarray, cfg: Config, rng: np.random.Generator) -> tuple:
    P = _prepare(P, cfg)
    
    idx = rng.integers(cfg.N)
    before = float(P[idx])

    magnitude = float(rng.uniform(low = 0.0, high = cfg.c))
    sign = float(rng.choice([-1.0, 1.0]))


    if cfg.mutation_operation == "additive":
        P[idx] += magnitude * sign

    else:
        P[idx] *= 1.0 + magnitude * sign


    undo = ("single-gene", idx, before)


    return P, undo




def phenotype_mutation(P: np.ndarray, cfg: Config, rng: np.random.Generator) -> tuple:
    P = _prepare(P, cfg)
    before = P.copy()

    magnitude = rng.uniform(low = 0.0, high = cfg.c, size = cfg.N)
    sign = rng.choice([-1.0, 1.0], size = cfg.N)


    if cfg.mutation_operation == "additive":
        P += magnitude * sign

    else:
        P *= 1.0 + magnitude * sign


    undo = ("phenotype", None, before)


    return P, undo




def perturbation(P: np.ndarray, cfg: Config, rng: np.random.Generator) -> tuple:
    P = _prepare(P, cfg)
    before = P.copy()

    P += rng.uniform(low = -cfg.c, high = cfg.c, size = cfg.N)

    undo = ("perturbation", None, before)


    return P, undo




_MUTATIONS = {"single-gene": single_gene_mutation, "phenotype": phenotype_mutation, "perturbation": perturbation}

def compute_mutation(P: np.ndarray, cfg: Config, rng: np.random.Generator) -> tuple:
    
    
    return _MUTATIONS[cfg.mutation_type.strip().lower()](P, cfg, rng)




def reverse_last_mutation(P: np.ndarray, undo: tuple, cfg: Config) -> np.ndarray:
    
    if undo is None:

        return P


    kind, idx, before = undo

    P = _prepare(P, cfg)


    if kind == "single-gene":
        P[idx] = before

    else:
        P[...] = before


    return P
