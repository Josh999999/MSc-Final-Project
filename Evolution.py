"""External Imports (Libraries and APIs)"""
import numpy as np


"""Local Imports"""
from Config import Config
from Induction import handle_induction
from GRN import DTYPE, masked_matrix, handle_develop, evaluate_fitness, mutate_profile, _cos




def sswm_evolve(cfg: Config, rng: np.random.Generator, B: np.ndarray = None) -> dict:
    S = cfg.targets
    M = cfg.n_targets

    B = np.asarray(B, dtype = DTYPE)
    B = masked_matrix(B, cfg.mask)
    B_flat = B.reshape(-1)                  # view: mutated in place
    B_ind = None

    G = np.zeros(cfg.N)

    allowed = cfg.allowed()
    n_allowed = allowed.size


    if cfg.symmetric_interactions:
        rows, cols = np.unravel_index(allowed, (cfg.N, cfg.N))
        partner = cols * cfg.N + rows       # flat index of (j, i)



    # Recording the evolutionary process
    interaction_developments = []
    recorded_gens = []
    fitness = []
    selections = []
    G_alignment = []
    B_alignment = []
    B_magnitude = []


    # Recording the induction process
    native_fitness = []
    plastic_fitness = []
    F_change_inner = []
    F_change_outer = []
    AUC_inner = []
    AUC_outer = []



    ei = int(rng.integers(M))
    P = handle_develop(G, B, cfg, induction = False)

    F = 0
    F_mut



    
    """ Used when induction needs to be evaluated early (e.g. for a switch of targets) to ensure fair comparison """
    def induction_history():    
        F_mut = history["auc_inner"]
        B_ind = history["B"]


        # Log the linear progression of fitness curve (Record the start and end curve)
        if cfg.record:
            start_curve = history['curve'][0]
            end_curve = history['F']
            F_ci = history['F_change_inner']
            F_co = history['F_change_outer']

            native_fitness.append(start_curve)
            plastic_fitness.append(end_curve)



    def induction_switch():
    
        if cfg.induction:
            history = handle_induction(B, P, G, cfg, rng, S[ei])
            F_mut = history["auc_inner"]
            B_ind = history["B"]


            # Log the linear progression of fitness curve
            if cfg.record:
                native_fitness.append(history['curve'][0])
                plastic_fitness.append(history['curve'][-1])

        else:
            F_mut = evaluate_fitness(P, S[ei], cfg)


        # Re-assign the interaction matrix with the induction vairant if the Baldwin effect is not active
        if cfg.induction and not cfg.baldwin_effect:
            B[...] = B_ind

        
        # Record Fintess
        if cfg.record:
            fitness.append(F_mut)


        return F_mut




    # The initial model should be allowed it's own induction proccess for the sake of fairness and reliable comparison
    # Now all incumbents have been through the induction process (if not recently) exactly once
    # Used when induction needs to be evaluated early (e.g. for a switch of targets) to ensure fair comparison
    F = induction_switch()    




    # Loop through the generations
    for gen in range(cfg.n_generations):


        # !-- Switching targets (For Multiple Target problems e.g. S1, S2) -- !

        if M > 1 and gen > 0 and gen % cfg.switch_every == 0:
            ei = int(rng.integers(M))

            # Used when induction needs to be evaluated early (e.g. for a switch of targets) to ensure fair comparison
            F = induction_switch()



        
        # !-- Perform the Mutation Process -- !

        # Mutate B in place, remembering the delta so it can be reverted.
        undo = None

        if n_allowed and rng.random() < cfg.prob_mut_B:
            t = rng.integers(0, n_allowed, size = cfg.n_mut_B)
            deltas = rng.uniform(low = -cfg.u2, high = cfg.u2, size = cfg.n_mut_B)
            idx = allowed[t]
            np.add.at(B_flat, idx, deltas)


            if cfg.symmetric_interactions:
                mirror = partner[t]
                off = mirror != idx
                np.add.at(B_flat, mirror[off], deltas[off])
                undo = (idx, mirror[off], deltas, deltas[off])

            else:
                undo = (idx, None, deltas, None)


        # Create the mutated model for comparison
        G_mut = mutate_profile(G, cfg, rng)

        # Develop under the MUTATED matrix (B has just been mutated in place).
        P_mut = handle_develop(G_mut, B, cfg, induction = False)


        # Record alignment of the mutations with the target
        if cfg.record:
            G_A = _cos(G, S[ei], norm = True)
            B_A = _cos(B, S[ei], norm = True)
            G_alignment.append(G_A)
            B_alignment.append(B_A)




        # !-- Perform the Induction Process -- !

        # Handle Induction
        if cfg.induction:

            # Compute induction
            history = handle_induction(B, P_mut, G, cfg, rng, S[ei])
            F_mut = history["auc_inner"]
            B_ind = history["B"]


            # Log the linear progression of fitness curve (Record the start and end curve)
            if cfg.record:
                start_curve = history['curve'][0]
                end_curve = history['F']

                native_fitness.append(start_curve)
                plastic_fitness.append(end_curve)

        else:
            F_mut = evaluate_fitness(P_mut, S[ei], cfg)




        # !-- Perform the Selection Process -- !

        # Compute acceptance boundary
        accept = (F_mut >= F) if cfg.drift_selection else (F_mut > F)

        if accept:

            # Re-assign the interaction matrix with the induction vairant if the Baldwin effect is not active
            if cfg.induction and not cfg.baldwin_effect:
                B[...] = B_ind


            G, P, F = G_mut, P_mut, F_mut


            # Record weather the mutation was selected      
            if cfg.record:
                selections.append(True)

        elif undo is not None:
            idx, mirror, deltas, mdeltas = undo
            np.add.at(B_flat, idx, -deltas)


            if mirror is not None:
                np.add.at(B_flat, mirror, -mdeltas)


            # Record weather the mutation was selected      
            if cfg.record:
                selections.append(True)



        
        # Subsampled recording of masked entries only
        if cfg.record and (gen % cfg.record_every == 0):
            interaction_developments.append(B_flat[allowed].copy())
            recorded_gens.append(gen)




    if cfg.record:
        if interaction_developments:
            trajectories = np.array(interaction_developments, dtype = DTYPE).T
            gens = np.array(recorded_gens)

        else:
            trajectories = np.empty((n_allowed, 0))
            gens = np.array([], dtype = int)


        # Record the magnitude of the current interaction matrix
        Y = np.linalg.norm(B, ord = "fro")
        B_magnitude.append(Y)




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




def sswm_evolve_return_wrapper(cfg: Config, rng: np.random.Generator, B: np.ndarray = None, limit_return: bool = False) -> dict:

    B, P, F, inner_curve, outer_curve, align_curve = sswm_evolve(cfg, rng, B)


    if limit_return:
    
        return {
            "B": B,
            "auc_inner": np.mean(inner_curve) if inner_curve.size else 0.0,
        }


    return {
        "B": B,
        "P": P,
        "F": F,
        "F_change_inner": inner_curve[-1] - inner_curve[0],
        "F_change_outer": outer_curve[-1] - outer_curve[0],        
        "auc_inner": np.mean(inner_curve) if inner_curve.size else 0.0,                 # area under the ABSOLUTE curve
        "auc_outer": np.mean(outer_curve) if outer_curve.size else 0.0,
        "inner_curve": inner_curve,
        "outer_curve": outer_curve,
        "align_curve": align_curve,
        "align_change": align_curve[-1] - align_curve[0],
    }