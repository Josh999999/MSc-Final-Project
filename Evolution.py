"""External Imports (Libraries and APIs)"""
import numpy as np
import time


"""Local Imports"""
from Config import Config
from Induction import handle_induction
from GRN import DTYPE, masked_matrix, handle_develop, evaluate_fitness, mutate_profile, fitness, _cos
from Analysis import check_convergence
from Tenets import measure_tenet1, measure_tenet2




def sswm_evolve(cfg: Config, rng: np.random.Generator, B: np.ndarray = None) -> dict:

    # Start code timing at the very start of the code for consistency
    start_time = time.perf_counter()


    S = cfg.targets
    M = cfg.n_targets

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
    fitnesses = []
    selections = []
    G_alignment = []
    B_alignment = []
    B_magnitude = []
    converged = []

    # Recording the induction process
    native_fitness = []
    plastic_fitness = []
    F_change_inner = []
    F_change_outer = []
    AUC_inner = []
    AUC_outer = []    

    # Single value results
    conv_first_gen = -1
    conv_stick_gen = -1
    conv_fgs = False
    conv_final_gen = False
    conv_first_time = -1
    conv_stick_time = -1

    final_fitness_uncapped = -1
    final_fitness_capped = -1



    ei = int(rng.integers(M))
    P = handle_develop(G, B, cfg, induction = False)

    F = 0
    F_mut = 0



    
    """ Used when induction needs to be evaluated early (e.g. for a switch of targets) to ensure fair comparison """
    def induction_history(history: dict) -> tuple[float, np.ndarray]:    
        F_mut = history["auc_inner"]
        B_ind = history["B"]


        # Log the linear progression of fitnesses curve (Record the start and end curve)
        if cfg.record:
            start_curve = history['curve'][0]
            end_curve = history['F']
            F_ci = history['F_change_inner']
            F_co = history['F_change_outer']
            auc_inner = history['auc_inner']
            auc_outer = history['auc_outer']

            native_fitness.append(start_curve)
            plastic_fitness.append(end_curve)
            F_change_inner.append(F_ci)
            F_change_outer.append(F_co)
            AUC_inner.append(auc_inner)
            AUC_outer.append(auc_outer)


        return F_mut, B_ind



    def induction_switch() -> float:
    
        if cfg.induction:
            history = handle_induction(B, P, G, cfg, rng, S[ei])       
            F_mut, B_ind = induction_history(history)

        else:
            F_mut = evaluate_fitness(P, S[ei], cfg)


        # Re-assign the interaction matrix with the induction vairant if the Baldwin effect is not active
        if cfg.induction and not cfg.baldwin_effect:
            B[...] = B_ind

        
        # Record Fintess
        if cfg.record:
            fitnesses.append(F_mut)


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
            G_A = measure_tenet2(G, cfg, S[ei])
            B_A = measure_tenet1(B, cfg, S[ei])
            G_alignment.append(G_A)
            B_alignment.append(B_A)




        # !-- Perform the Induction Process -- !

        # Handle Induction
        if cfg.induction:

            # Compute induction
            history = handle_induction(B, P_mut, G, cfg, rng, S[ei])            
            F_mut, B_ind = induction_history(history)

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
                selections.append(False)

        else:

            # Record weather the mutation was selected      
            if cfg.record:
                selections.append(False)



        
        # Subsampled recording of masked entries only
        if cfg.record:

            if (gen % cfg.record_every == 0):
                interaction_developments.append(B_flat[allowed].copy())
                recorded_gens.append(gen)


            # Record the magnitude of the current interaction matrix
            Y = np.linalg.norm(B, ord = "fro")
            B_magnitude.append(Y)

            # Check if the model has converged
            is_converged = check_convergence(F, cfg, l = 0.2)
            converged.append(is_converged)

            # Record the fitness
            fitnesses.append(F)




            # !-- Single Value Calculations -- !

            # Record convergence results
            if is_converged and conv_first_gen == -1:
                conv_first_gen = gen
                conv_stick_gen = gen

                end_time = time.perf_counter()
                conv_first_time = end_time - start_time

            elif is_converged and conv_stick_gen == -1:
                conv_stick_gen = gen

                end_time = time.perf_counter()
                conv_stick_time = end_time - start_time

            elif not is_converged:
                conv_stick_gen = -1
                conv_stick_time = -1





    if cfg.record:
        if interaction_developments:
            trajectories = np.array(interaction_developments, dtype = DTYPE).T
            gens = np.array(recorded_gens)

        else:
            trajectories = np.empty((n_allowed, 0))
            gens = np.array([], dtype = int)
        

        # Final analysis of the convergence
        if conv_first_gen == conv_stick_gen:
            conv_fgs = True


        if conv_stick_time > 0:
            conv_final_gen = True


        # Final analysis of the fitness
        final_fitness_uncapped = fitness(P, S[ei], limit = False, norm = cfg.normalise_fitness) if cfg.fitness_type == "standard" else _cos(P, S, norm = cfg.normalise_fitness)
        final_fitness_capped = fitness(P, S[ei], limit = True, norm = cfg.normalise_fitness) if cfg.fitness_type == "standard" else _cos(P, S, norm = cfg.normalise_fitness)






    if not cfg.record:
        
        return {
            "B": B,
            "G": G,
            "P": P,
            "F": F,
        }


    return {
        "B": B,
        "G": G,
        "P": P,
        "F": F,
        "target_index": ei,
        "trajectories": trajectories,
        "recorded_gens": gens,
        "fitnesses": fitnesses,
        "selections": selections,
        "G_alignment": G_alignment,
        "B_alignment": B_alignment,
        "B_magnitude": B_magnitude,
        "converged": converged, 
        "native_fitness": native_fitness,
        "plastic_fitness": plastic_fitness,
        "F_change_inner": F_change_inner,
        "F_change_outer": F_change_outer,
        "AUC_inner": AUC_inner,
        "AUC_outer": AUC_outer,
        "conv_first_gen": conv_first_gen,
        "conv_stick_gen": conv_stick_gen,
        "conv_fgs": conv_fgs,
        "conv_final_gen": conv_final_gen,
        "conv_first_time": conv_first_time,
        "conv_stick_time": conv_stick_time,
        "final_fitness_uncapped": final_fitness_uncapped,
        "final_fitness_capped": final_fitness_capped
    }