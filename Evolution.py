
"""External Imports (Libraries and APIs)"""
import numpy as np


"""Local Imports"""
from Config import Config
from Induction import handle_induction
from GRN import masked_matrix, handle_develop, evaluate_fitness, mutate_profile




def sswm_evolve(cfg: Config, rng: np.random.Generator, B: np.ndarray = None) -> dict:
    S = cfg.targets
    M = cfg.n_targets

    B = masked_matrix(B, cfg.mask)
    B_flat = B.reshape(-1)                  # view: mutated in place
    G = np.zeros(cfg.N)

    allowed = cfg.allowed()
    n_allowed = allowed.size


    if cfg.symmetric_B:
        rows, cols = np.unravel_index(allowed, (cfg.N, cfg.N))
        partner = cols * cfg.N + rows       # flat index of (j, i)


    interaction_developments = []
    recorded_gens = []
    native_fitness = []
    plastic_fitness = []
    start_curve = 0
    end_curve = 0

    ei = int(rng.integers(M))
    P = handle_develop(G, B, cfg, induction = False)



    
    """ Used when induction needs to be evaluated early (e.g. for a switch of targets) to ensure fair comparison """
    def induction_switch():
        global start_curve
        global end_curve
        global B
        global F

        B_ind = None

    
        if cfg.induction:
            history = handle_induction(B, P, G, cfg, rng, S[ei])
            F = history["auc"]
            B_ind = history["B"]

            # Log the linear progression of fitness curve
            start_curve = history['curve'][0]
            end_curve = history['curve'][-1]

        else:
            F = evaluate_fitness(P, S[ei], cfg)


        # Re-assign the interaction matrix with the induction vairant if the Baldwin effect is not active
        if cfg.induction and not cfg.baldwin_effect:
            B[...] = B_ind


        return F




    # The initial model should be allowed it's own induction proccess for the sake of fairness and reliable comparison
    # Now all incumbents have been through the induction process (if not recently) exactly once
    # Used when induction needs to be evaluated early (e.g. for a switch of targets) to ensure fair comparison
    F = induction_switch()


    # Loop through the generations
    for gen in range(cfg.n_generations):


        if M > 1 and gen > 0 and gen % cfg.switch_every == 0:
            ei = int(rng.integers(M))
            F = induction_switch()


        # Create the mutated model for comparison
        G_mut = mutate_profile(G, cfg, rng)

        # Mutate B in place, remembering the delta so it can be reverted.
        undo = None


        if n_allowed and rng.random() < cfg.prob_mut_B:
            t = rng.integers(0, n_allowed, size = cfg.n_mut_B)
            deltas = rng.uniform(low = -cfg.u2, high = cfg.u2, size = cfg.n_mut_B)
            idx = allowed[t]
            np.add.at(B_flat, idx, deltas)


            if cfg.symmetric_B:
                mirror = partner[t]
                off = mirror != idx
                np.add.at(B_flat, mirror[off], deltas[off])
                undo = (idx, mirror[off], deltas, deltas[off])

            else:
                undo = (idx, None, deltas, None)


        # Develop under the MUTATED matrix (B has just been mutated in place).
        P_mut = handle_develop(G_mut, B, cfg, induction = False)


        # Handle Induction
        F_mut = 0
        B_ind = None

        if cfg.induction:

            # Compute induction
            history = handle_induction(B, P_mut, G, cfg, rng, S[ei])
            F_mut = history["auc"]
            B_ind = history["B"]

            # Log the linear progression of fitness curve
            start_curve = history['curve'][0]
            end_curve = history['curve'][-1]

        else:
            F_mut = evaluate_fitness(P_mut, S[ei], cfg)



        # !-- Perform the selection process -- !

        # Compute acceptance boundary
        accept = (F_mut >= F) if cfg.drift_selection else (F_mut > F)

        if accept:

            # Re-assign the interaction matrix with the induction vairant if the Baldwin effect is not active
            if cfg.induction and not cfg.baldwin_effect:
                B[...] = B_ind

            G, P, F = G_mut, P_mut, F_mut      

        elif undo is not None:
            idx, mirror, deltas, mdeltas = undo
            np.add.at(B_flat, idx, -deltas)


            if mirror is not None:
                np.add.at(B_flat, mirror, -mdeltas)

        
        # Record the start and end curve
        if cfg.induction and accept:
            native_fitness.append(start_curve)
            plastic_fitness.append(end_curve)

        elif cfg.induction:
            native_fitness.append(native_fitness[-1])
            plastic_fitness.append(plastic_fitness[-1])


        # Subsampled recording of masked entries only
        if cfg.record_trajectories and (gen % cfg.record_every == 0):
            interaction_developments.append(B_flat[allowed].copy())
            recorded_gens.append(gen)


    if interaction_developments:
        trajectories = np.array(interaction_developments).T
        gens = np.array(recorded_gens)

    else:
        trajectories = np.empty((n_allowed, 0))
        gens = np.array([], dtype = int)


    return {
        "G": G,
        "B": B,
        "P": P,
        "F": F,
        "target_index": ei,
        "trajectories": trajectories,
        "recorded_gens": gens,
        "config": cfg,
        "native_fitness": native_fitness,
        "plastic_fitness": plastic_fitness
    }
