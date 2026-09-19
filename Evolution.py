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
    switches = []
    transfer_fitness = []        
    transfer_loss = []        
    G_alignment = []
    B_alignment = []
    B_magnitude = []
    converged = []
 
    # Recording the induction process
    native_fitness = []
    plastic_fitness = []
    F_change_inner = []
    F_change_outer = []
    F_first_inner = []
    F_final_inner = []
    F_first_outer = []
    F_final_outer = []
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
    Final_AUC = 0
 
 
 
    
    """ Used when induction needs to be evaluated early (e.g. for a switch of targets) to ensure fair comparison """
    def induction_history(cfg: Config, history: dict, record: bool = True, record_now: bool = True) -> tuple[float, np.ndarray]:    
        F_mut = history["auc_inner"]
        B_ind = history["B"]
        Final_AUC = 0
 
 
        # `record` is False when this is the incumbent being re-scored (start of
        # run, or after a target switch).  Those are not generations, so logging
        # them here would add extra entries and push every induction series out
        # of step with the generation index.
        # `record`     : False when re-scoring the incumbent (not a generation)
        # `record_now` : False on generations between subsample points
        if cfg.record and record and record_now:
 
            if cfg.induction_process == "r-round":
                start_curve = history['outer_curve'][0]
                end_curve = history['F']
                F_ci = history['F_change_inner']
                F_co = history['F_change_outer']
                auc_inner = history['auc_inner']
                auc_outer = history['auc_outer']
                f_first_in = history['F_first_inner']
                f_final_in = history['F_final_inner']
                f_first_out = history['F_first_outer']
                f_final_out = history['F_final_outer']
 
                native_fitness.append(start_curve)
                plastic_fitness.append(end_curve)
                F_change_inner.append(F_ci)
                F_change_outer.append(F_co)
                AUC_inner.append(auc_inner)
                AUC_outer.append(auc_outer)
                F_first_inner.append(f_first_in)
                F_final_inner.append(f_final_in)
                F_first_outer.append(f_first_out)
                F_final_outer.append(f_final_out)
 
                Final_AUC = history['inner_curve'][-1]
 
 
        return F_mut, B_ind, Final_AUC
 
 
 
    def induction_switch(B: np.ndarray, P: np.ndarray, G: np.ndarray, S: np.ndarray, cfg: Config, rng: np.random.Generator, AUC: float) -> tuple[float]:
        Final_AUC = 0
    
        if cfg.induction:
            history = handle_induction(B, P, G, cfg, rng, S, limit_return = not cfg.record, AUC = AUC)       
            F_mut, B_ind, Final_AUC = induction_history(cfg, history, record = False)
 
        else:
            F_mut = evaluate_fitness(P, S, cfg)
 
 
        # Re-assign the interaction matrix with the induction vairant if the Baldwin effect is not active
        if cfg.induction and not cfg.baldwin_effect:
            B[...] = B_ind
 
        
 
 
 
        return F_mut, Final_AUC
 
 
 
 
    # The initial model should be allowed it's own induction proccess for the sake of fairness and reliable comparison
    # Now all incumbents have been through the induction process (if not recently) exactly once
    # Used when induction needs to be evaluated early (e.g. for a switch of targets) to ensure fair comparison
    F, Final_AUC = induction_switch(B, P, G, S[ei], cfg, rng, AUC = F)    
 
 
 
 
    # Loop through the generations
    for gen in range(cfg.n_generations):
 
 
        # !-- Switching targets (For Multiple Target problems e.g. S1, S2) -- !
        switched = bool(M > 1 and gen > 0 and gen % cfg.switch_every == 0)
 
        if switched:
            F_before = F                     
            ei = int(rng.integers(M))
 
            F, Final_AUC = induction_switch(B, P, G, S[ei], cfg, rng, AUC = F)
 
 
 
        
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
 
 
        record_now = bool(cfg.record and ((gen % cfg.record_every == 0) or (gen == cfg.n_generations - 1) or switched))
 
        if cfg.record:
            switches.append(switched)
 
 
        if record_now:
            transfer_fitness.append(F if switched else np.nan)
            transfer_loss.append(F_before - F if switched else np.nan)
 
            G_alignment.append(measure_tenet2(G, cfg, S[ei]))
            B_alignment.append(measure_tenet1(B, cfg, S[ei]))
 
 
 
 
        # !-- Perform the Induction Process -- !
 
        # Handle Induction
        if cfg.induction:
 
            # Compute induction
            history = handle_induction(B, P_mut, G, cfg, rng, S[ei], limit_return = not cfg.record, AUC = Final_AUC)        
            F_mut, B_ind, Final_AUC = induction_history(cfg, history, record_now = record_now)
 
        else:
            F_mut = evaluate_fitness(P_mut, S[ei], cfg)
 
 
 
 
        # !-- Perform the Selection Process -- !
 
        # Compute acceptance boundary
        accept = (F_mut >= F) if cfg.drift_selection else (F_mut > F)
        selected = False
 
        if accept:
 
            # Re-assign the interaction matrix with the induction vairant if the Baldwin effect is not active
            if cfg.induction and not cfg.baldwin_effect:
                B[...] = B_ind
 
 
            G, P, F = G_mut, P_mut, F_mut
 
 
            # Record weather the mutation was selected      
            selected = True
 
        elif undo is not None:
            idx, mirror, deltas, mdeltas = undo
            np.add.at(B_flat, idx, -deltas)
 
 
            if mirror is not None:
                np.add.at(B_flat, mirror, -mdeltas)
 
 
        # Per-generation event flags, always at full resolution.
        if cfg.record:
            selections.append(selected)
 
            is_converged = check_convergence(F, cfg, l = 0.2)
            converged.append(is_converged)
 
 
        # Subsampled continuous measures.
        if record_now:
            interaction_developments.append(B_flat[allowed].copy())
            recorded_gens.append(gen)
 
            # Magnitude of the current interaction matrix
            B_magnitude.append(np.linalg.norm(B, ord = "fro"))
 
            # Fitness the generation ended on
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
        if not conv_first_gen == 1 and conv_first_gen == conv_stick_gen:
            conv_fgs = True
 
 
        if conv_stick_time > 0:
            conv_final_gen = True
 
 
        # Final analysis of the fitness
        final_fitness_uncapped = fitness(P, S[ei], limit = False, norm = cfg.normalise_fitness) if cfg.fitness_type == "standard" else _cos(P, S[ei], norm = cfg.normalise_fitness)
        final_fitness_capped = fitness(P, S[ei], limit = True, norm = cfg.normalise_fitness) if cfg.fitness_type == "standard" else _cos(P, S[ei], norm = cfg.normalise_fitness)
 
 
 
 
 
 
    # Convert all arrays to numpy arrays before returning them
    trajectories = np.array(trajectories, dtype = DTYPE)
    gens = np.array(gens, dtype = int)
    fitnesses = np.array(fitnesses, dtype = DTYPE)
    selections = np.array(selections, dtype = bool)
    switches = np.array(switches, dtype = bool)
    transfer_fitness = np.array(transfer_fitness, dtype = DTYPE)
    transfer_loss = np.array(transfer_loss, dtype = DTYPE)
    G_alignment = np.array(G_alignment, dtype = DTYPE)
    B_alignment = np.array(B_alignment, dtype = DTYPE)
    B_magnitude = np.array(B_magnitude, dtype = DTYPE)
    converged = np.array(converged, dtype = bool)
    native_fitness = np.array(native_fitness, dtype = DTYPE)
    plastic_fitness = np.array(plastic_fitness, dtype = DTYPE)
    F_change_inner = np.array(F_change_inner, dtype = DTYPE)
    F_change_outer = np.array(F_change_outer, dtype = DTYPE)
    AUC_inner = np.array(AUC_inner, dtype = DTYPE)
    AUC_outer = np.array(AUC_outer, dtype = DTYPE)
 
 
    if not cfg.record:
        
        return {
            "B": B,
            "G": G,
            "P": P,
            "F": F,
        }
 
 
    return (
        {
            "B": B,
            "G": G,
            "P": P,
            "F": F,
            "target_index": ei,
            "trajectories": trajectories,
            "recorded_gens": gens,
            "fitnesses": fitnesses,
            "selections": selections,
            "switches": switches,
            "transfer_fitness": transfer_fitness,
            "transfer_loss": transfer_loss,
            "G_alignment": G_alignment,
            "B_alignment": B_alignment,
            "B_magnitude": B_magnitude,
            "converged": converged, 
            "native_fitness": native_fitness,
            "plastic_fitness": plastic_fitness,
            "F_change_inner": F_change_inner,
            "F_change_outer": F_change_outer,
            "AUC_inner": AUC_inner,
            "AUC_outer": AUC_outer
        },
        {
            "F": F,
            "conv_first_gen": conv_first_gen,
            "conv_stick_gen": conv_stick_gen,
            "conv_fgs": conv_fgs,
            "conv_final_gen": conv_final_gen,
            "conv_first_time": conv_first_time,
            "conv_stick_time": conv_stick_time,
            "final_fitness_uncapped": final_fitness_uncapped,
            "final_fitness_capped": final_fitness_capped,        
            "F_first_inner": F_first_inner,
            "F_final_inner": F_final_inner,
            "F_first_outer": F_first_outer,
            "F_final_outer": F_final_outer,
        }
    )
 
 


def compare_results(induction_results: dict, standard_results: dict) -> dict:
    comparisons = dict()
    n = len(induction_results["recorded_gens"])
    m = len(induction_results["selections"])

    # Here we compare the induction results with the results of the standard evolutionary algorithm
    n = len(induction_results["fitnesses"])
    comparisons["greater_fitness_curve"] = np.asarray(induction_results["fitnesses"] > standard_results["fitnesses"], dtype = int).sum() / n
    comparisons["equal_fitness_curve"] = np.asarray(induction_results["fitnesses"] == standard_results["fitnesses"], dtype = int).sum() / n
    comparisons["less_fitness_curve"] = np.asarray(induction_results["fitnesses"] < standard_results["fitnesses"], dtype = int).sum() / n
    
    n = len(induction_results["selections"])
    comparisons["only_selection_curve"] = np.asarray(induction_results["selections"] and not standard_results["selections"], dtype = int).sum() / n
    comparisons["both_selection_curve"] = np.asarray(induction_results["selections"] and standard_results["selections"], dtype = int).sum() / n
    comparisons["other_selection_curve"] = np.asarray(not induction_results["selections"] and standard_results["selections"], dtype = int).sum() / n
    comparisons["none_selection_curve"] = np.asarray(not induction_results["selections"] and not standard_results["selections"], dtype = int).sum() / n

    n = len(induction_results["converged"])
    comparisons["only_converged_curve"] = np.asarray(induction_results["converged"] and not standard_results["converged"], dtype = int).sum() / n
    comparisons["both_converged_curve"] = np.asarray(induction_results["converged"] and standard_results["converged"], dtype = int).sum() / n
    comparisons["other_converged_curve"] = np.asarray(not induction_results["converged"] and standard_results["converged"], dtype = int).sum() / n
    comparisons["none_converged_curve"] = np.asarray(not induction_results["converged"] and not standard_results["converged"], dtype = int).sum() / n



 
def compare_results_str(induction_results: dict, standard_results: dict) -> dict:
    comparisons = dict()
 
    # Here we compare the induction results with the results of the standard evolutionary algorithm
    comparisons["first_initial_convergence"] = int(induction_results["conv_first_gen"] < standard_results["conv_first_gen"])
    comparisons["same_initial_convergence"] = int(induction_results["conv_first_gen"] == standard_results["conv_first_gen"])
    comparisons["last_initial_convergence"] = int(induction_results["conv_first_gen"] > standard_results["conv_first_gen"])
    
    comparisons["first_stick_convergence"] = int(induction_results["conv_stick_gen"] < standard_results["conv_stick_gen"])
    comparisons["same_stick_convergence"] = int(induction_results["conv_stick_gen"] == standard_results["conv_stick_gen"])
    comparisons["last_stick_convergence"] = int(induction_results["conv_stick_gen"] > standard_results["conconv_stick_gen_first_gen"])
 
    comparisons["only_first_convergence_stuck"] = int(induction_results["conv_fgs"] and not standard_results["conv_fgs"])
    comparisons["both_first_convergence_stuck"] = int(induction_results["conv_fgs"] and standard_results["conv_fgs"])
    comparisons["other_first_convergence_stuck"] = int(not induction_results["conv_fgs"] and standard_results["conv_fgs"])
    comparisons["none_first_convergence_stuck"] = int(not induction_results["conv_fgs"] and not standard_results["conv_fgs"])

    comparisons["greater_fitness"] = int(induction_results["F"] > standard_results["F"])
    comparisons["equal_fitness"] = int(induction_results["F"] == standard_results["F"])
    comparisons["less_fitness"] = int(induction_results["F"] < standard_results["F"])



 
def average_dictionaries(dictionaries: list[dict[np.ndarray | DTYPE]]) -> dict[np.ndarray]:
    keys = list(dictionaries[0].keys())
    n = len(keys)
    A = [0.0] * n
    average_items = np.asarray(A, dtype = DTYPE)
 
    for item in dictionaries:
        average_items += np.asarray(item.values(), dtype = DTYPE)
 
    average_items /= n
 
    average_dictionary = dict(zip(keys, average_items))
 
 
    return average_dictionary




def average_dictionaries_str(dictionaries: list[dict[np.ndarray | DTYPE]]) -> dict[np.ndarray]:
    keys = list(dictionaries[0].keys())
    n = len(keys)
    A = [0.0] * n
    average_items = np.asarray(A, dtype = DTYPE)
 
    for item in dictionaries:
        average_items += np.asarray(item.values(), dtype = DTYPE)
 
    average_items = str(average_items) + f" / {n}"
 
    average_dictionary = dict(zip(keys, average_items))
 
 
    return average_dictionary