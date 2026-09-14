from dataclasses import dataclass, field, replace
import numpy as np

from Data import S1, S2, DEFAULT_SEED




def make_rng(seed: int = DEFAULT_SEED) -> np.random.Generator:

    return np.random.default_rng(seed)




# !-- Global variables --!

# String Option lists
DEVELOPMENT_SIGMOIDS = ("tanh", "linear")
MUTATION_TYPES = ("single-gene", "phenotype", "perturbation")
MUTATION_OPERATIONS = ("additive", "multiplicative")
INDUCTION_PROCESSES = ("plastic", "r-round")
ENERGY_GATES = ("or", "and", "harsh", "deterministic")




@dataclass(frozen = True, eq = False)
class Config:

    # !-- Core problem definition --!
    N: int = 8                              # size of the phenotype (number of genes)
    targets: np.ndarray = field(default_factory = lambda: np.array([S1, S2], dtype = float))
    seed: int = DEFAULT_SEED
    fitness_type: str = "cosine"            # Can be "cosine" or "standard"
    limit_fitness: bool = False             # Switch that limits gene representations affect on fitness to a magnitude of one


    # !-- Topology --!
    mask: np.ndarray = None                 # None -> dense minus the diagonal
    K: int = 4                              # out-degree before symmetrisation
    symmetric_mask: bool = True
    mask_combine: str = "union"             # "union" (mean degree ~2K) or "intersection"
    self_interaction: bool = False


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
    u2: float = 0.0067                      # interaction mutation size
    prob_mut_B: float = 0.067
    n_mut_G: int = 1
    n_mut_B: int = 1
    n_generations: int = 200_000
    switch_every: int = 2000
    record_trajectories: bool = True
    record_every: int = 1000
    drift_selection: bool = True            # accept ties as well as strict gains
    symmetric_B: bool = False               # mirror each B mutation to (j, i)
    baldwin_effect: bool = True             # PLACEHOLDER: accepted but currently inert


    # !-- Interaction matrix construction --!
    prob_pos_B: float = 0.5
    uniform: bool = False
    n_modules: int = 5
    intra: float = 1.0
    inter: float = 0.05
    flip_frac: float = 0.25
    small_magnitude: float = 0.1
    Y: float = 1.0                          # magnitude (gain) applied to B


    # !-- Induction --!
    induction: bool = False                     # Toggle induction (inside of the evolutionary algorithm)
    induction_process: str = "plastic"          # "plastic", "hopfield" or "r-round"
    energy_gate: str = "or"                     # Determines how the energy of the plasticy phenotype should impact acceptance: "or", "and", "harsh" or "deterministic"
    energy_limit: float = 0.5                   # Maximum amount sigmoid energy needs to achieve when `energy_gate = "harsh"` sigma(-dE/tau)
    energy_type: str = "standard"               # Can be 'standard' or 'differential'
    energy_normalise_interactions: bool = False # Normalise the interaction matrix inside of the energy calculations
    slack_limit: float = 0.01                   # Maximum amount of slack to be given in the randomised scaling of the energy limit boundry inside the plastic selection proccess
    M: int = 20                                 # mutation attempts per plastic search
    c: float = 2.0                              # single-gene mutation size
    c_tau: float = 1.0                          # tau = c_tau * std(dE) over the pool
    eta: float = 0.01                           # contrastive learning rate for B
    rounds: int = 10                            # R rounds of develop -> plasticity -> induct
    mutation_type: str = "single-gene"          # "single-gene", "phenotype" or "perturbation"
    mutation_operation: str = "additive"        # "additive" or "multiplicative"
    tau_floor: float = 1e-12                    # guard for a degenerate candidate pool
    normalise_energy: bool = True               # Rayleigh quotient: direction only
    relative_mutation: bool = False             # scale the step by |P|/sqrt(N)
    r_T: int = None                             # redevelopment time after updating
    relax: bool                                 # Toggle relaxation after induction (development of the original genotype under the new interaction matrix produced by induction)


    # !-- Output --!
    figures_output: str = ""





    def __post_init__(self):

        targets = np.atleast_2d(np.asarray(self.targets, dtype = float))
        object.__setattr__(self, "targets", targets)




        """!---- Configuring Object Values ----!"""
        if self.mask is None:
            m = np.ones((self.N, self.N), dtype = bool)
            np.fill_diagonal(m, self.self_interaction)
            object.__setattr__(self, "mask", m)

        else:
            m = np.asarray(self.mask).astype(bool)
            object.__setattr__(self, "mask", m)

        
        # Induction development time to match standard development time when not set
        if self.r_T is None:
            object.__setattr__(self, "r_T", self.T)






    """!---- Decorators for Setting Variables ----!"""
    @property
    def baseline_fitness(self) -> float:

        if self.fitness_type == "cosine":

            return 0.0 if self.normalise_fitness else -1.0

        elif self.fitness_type == "standard":

            return 0.0 if self.normalise_fitness else -self.N

        else:

            return 0.0




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




    @property
    def n_targets(self) -> int:

        return int(self.targets.shape[0])




    def allowed(self) -> np.ndarray:

        return np.flatnonzero(self.mask.ravel())








def with_mask(cfg: Config, mask: np.ndarray) -> Config:

    return replace(cfg, mask = mask)