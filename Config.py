from dataclasses import dataclass, field, replace
import numpy as np

from Data import S1, S2, DEFAULT_SEED




def make_rng(seed: int = DEFAULT_SEED) -> np.random.Generator:

    return np.random.default_rng(seed)




def spawn_rngs(seed: int, n: int) -> list:

    return list(np.random.default_rng(seed).spawn(n))




# !-- Global variables --!

# String Option lists
DEVELOPMENT_SIGMOIDS = ("tanh", "linear")
MUTATION_TYPES = ("single-gene", "phenotype", "perturbation")
MUTATION_OPERATIONS = ("additive", "multiplicative")
INDUCTION_PROCESSES = ("plastic", "hopfield", "r-round")
ENERGY_GATES = ("or", "and", "harsh", "deterministic")




@dataclass(frozen = True, eq = False)
class Config:


    # !-- Core problem definition --!
    N: int = 8                          # size of the phenotype (number of genes)
    targets: np.ndarray = field(default_factory = lambda: np.array([S1, S2], dtype = float))
    seed: int = DEFAULT_SEED


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
    interactions_norm: bool = False         # rescale B to Frobenius norm Y
    genotype_norm: bool = False             # rescale G to unit length
    fitness_norm: bool = True               # rescale fitness into [0, 1]


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
    baldwin_effect: bool = False            # PLACEHOLDER: accepted but currently inert


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
    induction: bool = False
    induction_process: str = "plastic"      # "plastic", "hopfield" or "r-round"
    induction_inplace: bool = False         # induction based learning is performed inplace
    energy_gate: str = "or"                 # Determines how the energy of the plasticy phenotype should impact acceptance: "or", "and", "harsh" or "deterministic"
    energy_limit: float = 0.5               # Maximum amount sigmoid energy needs to achieve when `energy_gate = "harsh"` sigma(-dE/tau)
    limit_slack: float = 0.01               # Maximum amount of slack to be given in the randomised scaling of the energy limit boundry inside the plastic selection proccess
    M: int = 20                             # mutation attempts per plastic search
    c: float = 0.1                          # single-gene mutation size
    c_tau: float = 1.0                      # tau = c_tau * std(dE) over the pool
    use_energy: bool = True                 # sigma(-dE/tau) inside the acceptance test
    eta: float = 0.01                       # contrastive learning rate for B
    rounds: int = 10                        # R rounds of develop -> plasticity -> induct
    mutation_type: str = "single-gene"      # "single-gene", "phenotype" or "perturbation"
    mutation_operation: str = "additive"    # "additive" or "multiplicative"
    mutate_inplace: bool = False            # mutate the caller's array rather than a copy
    tau_floor: float = 1e-12                # guard for a degenerate candidate pool
    normalise_energy: bool = True           # Rayleigh quotient: direction only
    relative_mutation: bool = False         # scale the step by |P|/sqrt(N)
    r_T: int = None                         # redevelopment time after updating


    # !-- Output --!
    figures_output: str = ""





    def __post_init__(self):

        targets = np.atleast_2d(np.asarray(self.targets, dtype = float))
        object.__setattr__(self, "targets", targets)




        """!---- Error Handling ----!"""
        if self.mask is not None:
            m = np.asarray(self.mask).astype(bool)


            if m.shape != (self.N, self.N):
                raise ValueError(f"mask is {m.shape}, expected ({self.N}, {self.N})")




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




        """!---- Error Handling ----!"""
        if targets.shape[1] != self.N:
            raise ValueError(
                f"targets have {targets.shape[1]} genes but N={self.N}. "
                "Pass targets and N together."
            )


        if self.K > self.N - 1:
            raise ValueError(
                f"K={self.K} exceeds the {self.N - 1} available partners for N={self.N}"
            )


        if self.mask_combine not in ("union", "intersection"):
            raise ValueError(
                "mask_combine must be 'union' or 'intersection'"
            )


        if self.sigmoid not in DEVELOPMENT_SIGMOIDS and not callable(self.sigmoid):
            raise ValueError(
                f"sigmoid must be callable or one of {DEVELOPMENT_SIGMOIDS}. "
                "The logistic 'sigmoid' is not odd and is not valid for development."
            )


        if self.mutation_type not in MUTATION_TYPES:
            raise ValueError(
                f"mutation_type must be one of {MUTATION_TYPES}"
            )


        if self.mutation_operation not in MUTATION_OPERATIONS:
            raise ValueError(
                f"mutation_operation must be one of {MUTATION_OPERATIONS}"
            )


        if self.mutation_operation == "multiplicative" and self.c >= 1.0:
            raise ValueError(
                f"multiplicative mutation with c={self.c} >= 1 can produce a factor of "
                "zero, which is not invertible. Use c < 1."
            )


        if self.induction_process.strip().lower() not in INDUCTION_PROCESSES:
            raise ValueError(
                "induction_process must be 'plastic', 'hopfield', 'r-round'"
            )

        
        if self.symmetric_B and not np.array_equal(self.mask, self.mask.T):
            raise ValueError(
                "symmetric_B=True requires a symmetric mask"
            )






    """!---- Decorators for Setting Variables ----!"""
    @property
    def baseline_fitness(self) -> float:

        return 0.0 if self.fitness_norm else -1.0




    @property
    def optimal_fitness(self) -> float:

        return 1.0




    @property
    def target(self) -> np.ndarray:

        return self.targets[0]




    @property
    def n_targets(self) -> int:

        return int(self.targets.shape[0])




    def allowed(self) -> np.ndarray:

        return np.flatnonzero(self.mask.ravel())




    def summary(self) -> dict:

        return {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in self.__dict__.items()
        }








def with_mask(cfg: Config, mask: np.ndarray) -> Config:

    return replace(cfg, mask = mask)