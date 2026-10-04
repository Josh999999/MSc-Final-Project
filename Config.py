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
 
    # !-- Development form --!
    development: str = "standard"   # "standard" = the published form,
                                    # P(0) = G, P += t1*sigma(BP) - t2*P;
                                    # "bounded" = G as a persistent input,
                                    # P += t1*(sigma(BP + G) - P)
    bound_phenotype: bool = True            # keep plastic moves inside [-1, 1] too
 
    # !-- Deterministic evaluation --!
    deterministic_induction: bool = True    # score every genotype with the same fixed induction stream (common random numbers)
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
    normalise_fitness: bool = True          # rescale fitness into [0, 1]
 
 
    # !-- Evolution --!
    u1: float = 0.1                         # genotype mutation size
    u2: float = 0.3                         # interaction mutation size
    prob_mut_B: float = 0.5
    n_mut_G: int = 1
    n_mut_B: int = 8
    B_limit: float = 0.3                    # per-entry bound on the interaction weights under mutation
    n_generations: int = 200_000
    switch_every: int = 2000
    record: bool = True
    record_every: int = 1000
    symmetric_interactions: bool = False    # mirror each B mutation to (j, i)
 
 
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
    energy_gate: str = "deterministic"          # Determines how the energy of the plastic phenotype should impact acceptance: "or", "and" or "deterministic"
    energy_type: str = "differential"           # Can be 'standard' or 'differential'
    energy_normalise_interactions: bool = True  # Normalise the interaction matrix inside of the energy calculations
    M: int = 100                                # mutation attempts per plastic search
    c: float = 1.0                              # single-gene mutation size
    eta: float = 0.5                            # contrastive learning rate for B
    rounds: int = 10                            # R rounds of plasticity -> induct -> relax. The walk's AUC saturates within ~5 rounds (Experiment 5), and every induction evaluation costs R walks: 50 rounds made evolution 5x slower for no measurable gain
    mutation_type: str = "phenotype"            # Can be "single-gene" or "phenotype"
    normalise_energy: bool = True               # Rayleigh quotient: direction only
    r_T: int = None                             # redevelopment time after updating
    relax: bool = True                          # Relax after each round: develop the plastic phenotype P' under the updated interaction matrix before the next walk (off: the next walk starts from P' itself)
 
 
 
 
 
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
 