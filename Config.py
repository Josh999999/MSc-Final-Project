from dataclasses import dataclass, field, replace
import numpy as np
 
from Data import S1, S2, DEFAULT_SEED
 
 
 
 
def make_rng(seed: int = DEFAULT_SEED) -> np.random.Generator:
 
    return np.random.default_rng(seed)
 
 
 
 
# !-- Global variables --!
 
ENERGY_GATES = ("or", "and", "deterministic")
 
 
 
 
@dataclass(eq = False)
class Config:
 
    # !-- Core problem definition --!
    N: int = 8                              # size of the phenotype (number of genes)
    targets: np.ndarray = field(default_factory = lambda: np.array([S1, S2], dtype = float))
    seed: int = DEFAULT_SEED
 
    # !-- Deterministic evaluation --!
    development: str = "standard"   # "standard" = the published form,
                                    # P(0) = G, P += t1*sigma(BP) - t2*P;
                                    # "bounded" = G as a persistent input,
                                    # P += t1*(sigma(BP + G) - P)
    bound_phenotype: bool = True            # keep plastic moves inside [-1, 1] too
 
    # !-- What selection scores --!
    selection_score: str = "gated"
    plastic_bonus: float = 0.1
    inherit_induced: str = "none"
    B_limit: float = 0.3
 
    deterministic_induction: bool = True
    induction_seed: int = DEFAULT_SEED
    fitness_type: str = "standard"          # Can be "cosine" or "standard"
 
 
    # !-- Topology --!
    mask: np.ndarray = None                 # None -> dense minus the diagonal
    sparse_interactions: bool = False       # Create a mask for a sparse interaction matrix
    K: int = 4                              # out-degree before symmetrisation
    symmetric_mask: bool = False
    mask_combine: str = "union"             # "union" (mean degree ~2K) or "intersection"
    self_interaction: bool = True
 
 
    # !-- Development --!
    sigmoid: str = "tanh"                   # "tanh" or "linear"
    t1: float = 1.0
    t2: float = 0.2
    T: int = 10                             # development time
    lr: float = 1.0                         # Hebbian learning rate
    
 
    # !-- Normalisation switches --!
    normalise_interactions: bool = False    # rescale B to Frobenius norm Y
    normalise_fitness: bool = True          # rescale fitness into [0, 1]
 
 
    # !-- Evolution --!
    u1: float = 0.1                         # genotype mutation size
    u2: float = 0.3                         # interaction mutation size
    prob_mut_B: float = 0.5
    n_mut_G: int = 1
    n_mut_B: int = 8
    n_generations: int = 200_000
    switch_every: int = 2000
    record: bool = True
    record_every: int = 1000
    drift_selection: bool = True            # accept ties as well as strict gains
    symmetric_interactions: bool = False    # mirror each B mutation to (j, i)
    baldwin_effect: bool = True             # PLACEHOLDER: accepted but currently inert
 
 
    # !-- Interaction matrix construction --!
    prob_pos_B: float = 0.5
    uniform: bool = False
    n_modules: int = 5
    intra: float = 1.0
    inter: float = 0.05
    flip_frac: float = 0.25
    Y: float = 1.0                           # magnitude (gain) applied to B
 
 
    # !-- Induction --!
    induction: bool = False                     # Toggle induction (inside of the evolutionary algorithm)
    induction_process: str = "r-round"          # Can be "plastic" or "r-round"
    energy_gate: str = "deterministic"          # Determines how the energy of the plasticy phenotype should impact acceptance: "or", "and" or "deterministic"
    energy_type: str = "differential"           # Can be 'standard' or 'differential'
    energy_normalise_interactions: bool = True  # Normalise the interaction matrix inside of the energy calculations
    M: int = 100                                # mutation attempts per plastic search
    c: float = 1.0                              # single-gene mutation size
    eta: float = 0.5                            # contrastive learning rate for B
    rounds: int = 10                            # R rounds of develop -> plasticity -> induct
    mutation_type: str = "phenotype"            # Can be "single-gene" or "phenotype"
    normalise_energy: bool = True               # Rayleigh quotient: direction only
    r_T: int = None                             # redevelopment time after updating
    relax: bool = False                         # Toggle relaxation after induction (development of the original genotype under the new interaction matrix produced by induction)
    induction_interactions: str = "all"         # Controls which interactions are changed during induction with regard to the mask; can be "inclusive", "exclusive" or "all"
 
 
 
 
 
    def __post_init__(self):
        self._auto_mask = self.mask is None
        self._auto_r_T  = self.r_T is None
 
        self.targets = np.atleast_2d(np.asarray(self.targets, dtype = float))
        self._rebuild_mask()
        self._rebuild_r_T()
 
 
 
 
    """!---- Derived values ----!"""
    def _rebuild_mask(self):
 
        if self._auto_mask:
            m = np.ones((self.N, self.N), dtype = bool)
            np.fill_diagonal(m, self.self_interaction)
            object.__setattr__(self, "mask", m)
 
        else:
            object.__setattr__(self, "mask", np.asarray(self.mask).astype(bool))
 
 
 
 
    def _rebuild_r_T(self):
 
        if self._auto_r_T:
            object.__setattr__(self, "r_T", self.T)
 
 
 
 
    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)
 
        # Nothing to keep in step until __post_init__ has run.
        if not hasattr(self, "_auto_mask"):
 
            return
 
 
        # An explicitly assigned mask or r_T stops being auto-derived.
        if name == "mask":
            object.__setattr__(self, "_auto_mask", value is None)
            self._rebuild_mask()
 
        elif name == "r_T":
            object.__setattr__(self, "_auto_r_T", value is None)
            self._rebuild_r_T()
 
        # Sources that derived values depend on.
        elif name in ("N", "self_interaction"):
            self._rebuild_mask()
 
        elif name == "T":
            self._rebuild_r_T()
 
        elif name == "targets":
            object.__setattr__(self, "targets",
                               np.atleast_2d(np.asarray(value, dtype = float)))
 
 
 
 
    """!---- Mutation helpers ----!"""
    def set(self, **kwargs) -> "Config":
 
        for k, v in kwargs.items():
 
            if k not in self.__dataclass_fields__:
                raise AttributeError(f"Config has no field '{k}'")
 
            setattr(self, k, v)
 
 
        return self
 
 
 
 
    @property
    def optimal_fitness(self) -> float:
 
        if self.fitness_type == "cosine":
        
            return 1.0
 
        elif self.fitness_type == "standard":
 
            return 1.0 if self.normalise_fitness else self.N
 
        else:
 
            return 1.0
 
 
 
 
    @property
    def target(self) -> np.ndarray:
 
        return self.targets[0]
 
 
 
 
    def allowed(self) -> np.ndarray:
 
        return np.flatnonzero(self.mask.ravel())
 
 
 
 
 
 
 
 
def with_mask(cfg: Config, mask: np.ndarray) -> Config:
 
    return replace(cfg, mask = mask)
 
 
 
 
def set_mask(cfg: Config, mask: np.ndarray) -> Config:
    cfg.mask = mask
 
 
    return cfg
 