"""External Imports (Libraries and APIs)"""
import os
import itertools
import numpy as np
 
 
"""Local Imports"""
from Config import Config, make_rng, set_mask
from Data import S1, S2, DEFAULT_SEED
from Plotting import plot_trajectories, plot_binary_strip, plot_transfer, show_interaction_heatmap, show_phenotypes
from Evolution import sswm_evolve
from GRN import diag_mask, handle_develop, evaluate_fitness, masked_matrix, _cos, hebbian_interactions
from Analysis import average_results, compare_arms, comparison_table, arm_summary, arm_summary_table
 
 
 
 
 
N_GENERATIONS = 20_000
SWITCH_EVERY  = 1_500
RECORD_EVERY  = 200
SEEDS         = 5
 
ARMS          = ("control", "induction")
 
 
 
 
 
def standard_environment():
    """The original task: two hand-made targets."""
 
    return np.array([S1, S2]), None
 
 
 
 
def modular_environment(N: int, K: int, seed: int):
    rng = make_rng(seed)
    size = N // K
    mods = [rng.choice([-1.0, 1.0], size = size) for _ in range(K)]
    targets = np.array([np.concatenate([s * m for s, m in zip(signs, mods)])
                        for signs in itertools.product([-1.0, 1.0], repeat = K)])
 
    # the "right" memory: block-diagonal within-module Hebbian
    ideal = np.zeros((N, N))
 
    for k, m in enumerate(mods):
        ideal[k * size:(k + 1) * size, k * size:(k + 1) * size] = np.outer(m, m)
 
 
    return targets, ideal
 
 
 
 
def make_config(targets: np.ndarray, induction: bool, seed: int) -> Config:
    cfg = Config(
        N = targets.shape[1],
        targets = targets,
        induction = induction,
        n_generations = N_GENERATIONS,
        switch_every = SWITCH_EVERY,
        record_every = RECORD_EVERY,
        seed = DEFAULT_SEED + seed,
        normalise_interactions = False,
    )
    set_mask(cfg, diag_mask(cfg))
 
 
    return cfg
 
 
 
 
def summarise(results: dict, cfg: Config, ideal: np.ndarray) -> dict:
    P = handle_develop(results["G"], results["B"], cfg, induction = False)
    tl = np.asarray(results["transfer_loss"], float); tl = tl[~np.isnan(tl)]
    B = np.asarray(results["B"])
 
    out = {
        "final_native_fitness": float(evaluate_fitness(P, cfg.targets[results["target_index"]], cfg)),
        "mean_fitness":         float(np.mean(results["fitnesses"])),
        "mean_switch_loss":     float(tl.mean()) if tl.size else float("nan"),
        "final_B_magnitude":    float(np.linalg.norm(B)),
        "acceptance_rate":      float(np.mean(results["selections"])),
    }
 
 
    if ideal is not None:
        out["modularity"] = float(_cos(B[cfg.mask], masked_matrix(ideal, cfg.mask)[cfg.mask]))
 
 
    return out
 
 
 
 
def plot_run(results: dict, cfg: Config, folder: str, label: str, rng: np.random.Generator):
    """The per-run diagnostic plots, exactly as configured in the original experiment."""
    trajectories = results["trajectories"]
    recorded_gens = np.asarray(results["recorded_gens"])
    fitnesses = results["fitnesses"]
    selections = results["selections"]
    switches = np.asarray(results["switches"], dtype = bool)
    transfer_fitness = results["transfer_fitness"]
    transfer_loss = results["transfer_loss"]
    G_alignment = results["G_alignment"]
    B_alignment = results["B_alignment"]
    B_magnitude = results["B_magnitude"]
    converged = results["converged"]
    # induction series are empty for the control arm and absent from an averaged control
    native_fitness = results.get("native_fitness", [])
    plastic_fitness = results.get("plastic_fitness", [])
    F_change_inner = results.get("F_change_inner", [])
    F_change_outer = results.get("F_change_outer", [])
    AUC_inner = results.get("AUC_inner", [])
    AUC_outer = results.get("AUC_outer", [])
    B = np.asarray(results["B"])
 
    OUTPUT_FOLDER = folder
    os.makedirs(OUTPUT_FOLDER, exist_ok = True)
 
    # Seed-averaged runs carry the binary series as FRACTIONS of seeds (0..1),
    # which the strip plots cannot draw; those two plots become line plots.
    sel = np.asarray(selections, float)
    averaged = not np.all(np.isin(sel, (0.0, 1.0)))
 
    PLOT_NAME = "interaction_trajectories.png"
    SAVELOC = os.path.join(OUTPUT_FOLDER, PLOT_NAME)
 
    plot_trajectories(
        x = recorded_gens,
        trajectories = trajectories,
        x_label = "Recorded Generation",
        y_label = "Interaction size",
        title = "Trajectories of the Weights inside the Interaction Matrix",        
        save_loc = SAVELOC, 
        figsize = (6, 4)
    )      
 
    print(f"         wrote {PLOT_NAME} to {SAVELOC}")
 
 
 
    # Plot the Fitness of the GRN over evolution (generations)
    PLOT_NAME = "fitness.png"
    SAVELOC = os.path.join(OUTPUT_FOLDER, PLOT_NAME)
 
    plot_trajectories(
        x = recorded_gens,
        trajectories = [fitnesses],
        marker = "x",
        x_label = "Generation",
        y_label = "Fitness (AUC)",
        title = "Fitness of the chosen GRN over Generations",        
        subtitle = "Fitness is the AUC of the selected GRN from the Induction Process",
        grid = True,
        save_loc = SAVELOC, 
        figsize = (12, 6),
        switch_plot = True,
        switch_lines = True,
        x_switches = switches
    )   
 
    print(f"         wrote {PLOT_NAME} to {SAVELOC}") 
 
 
 
    # Plot the Fitness lost for the current GRN when the Target Switches
    PLOT_NAME = "switch_loss.png"
    SAVELOC = os.path.join(OUTPUT_FOLDER, PLOT_NAME)
 
    plot_transfer(
        x = recorded_gens,
        transfer_fitness = transfer_fitness, 
        fitnesses = fitnesses, 
        transfer_loss = transfer_loss, 
        switches = switches,  
        save_loc = SAVELOC
    )
 
 
 
    # Plot generations where the mutated GRN is selected
    PLOT_NAME = "selections.png"
    SAVELOC = os.path.join(OUTPUT_FOLDER, PLOT_NAME)
 
    if averaged:
        plot_trajectories(x = np.arange(len(selections)), trajectories = [np.asarray(selections, float)],
                          x_label = "Generation", y_label = "fraction of seeds accepting", title = "Accepted mutations",
                          subtitle = label, grid = True, save_loc = SAVELOC, figsize = (12, 4),
                          switch_lines = True, x_switches = switches)
        print(f"         wrote {PLOT_NAME} to {SAVELOC}")
 
    else:
        SAVELOC = os.path.join(OUTPUT_FOLDER, PLOT_NAME)
 
        plot_binary_strip(
            values = selections,
            true_colour = "black",
            false_colour = "white",
            true_label = "Selected",
            false_label = "Not Selected",
            x_label = "Generation",
            title = "Selected Mutations over Generations",        
            subtitle = "Tracking Generations where the Mutated GRN was selected over the original",
            legend = True,
            save_loc = SAVELOC, 
            figsize = (16, 10)
        )   
 
        print(f"         wrote {PLOT_NAME} to {SAVELOC}")   
 
 
    
    # Plot the alignment of the elements (G and B) of the selected GRN with the target (T) over evolution (generations)
    PLOT_NAME = "grn_alignment.png"
    SAVELOC = os.path.join(OUTPUT_FOLDER, PLOT_NAME)
 
    plot_trajectories(
        x = recorded_gens,
        trajectories = [G_alignment, B_alignment],
        keys = ["Genotype Alignment", "Interaction Alignment"], 
        linestyle = ["-.", "--"], 
        marker = "x", 
        grid = True,
        x_label = "Generation",
        y_label = "Alignment (cosine alignmet with S))",
        title = "Genotype (G) and Interaction (B) Alignment with the Target (S)",        
        subtitle = "Tracks the Alignment of the Genotype (G) and Interaction (B) selected",
        save_loc = SAVELOC, 
        figsize = (14, 8),
        switch_plot = True,
        switch_lines = True,
        x_switches = switches
    )      
 
    print(f"         wrote {PLOT_NAME} to {SAVELOC}")   
 
 
    
    # Plot the Magnitude of the Interaction Matrix from the Selected GRN over evolution (generations)
    PLOT_NAME = "interaction_magnitude.png"
    SAVELOC = os.path.join(OUTPUT_FOLDER, PLOT_NAME)
 
    plot_trajectories(
        x = recorded_gens,
        trajectories = [B_magnitude],
        marker = "x",
        grid = True,
        x_label = "Generation",
        y_label = "Interaction Matrix (B) Magnitude (Y)",
        title = "Magnitude of the Selected Interaction Matrix (B) over Generations",        
        subtitle = "Interaction Matrix (B) is from the Selected GRN",
        save_loc = SAVELOC, 
        figsize = (12, 6)
    )   
 
    print(f"         wrote {PLOT_NAME} to {SAVELOC}")  
 
 
    
    # Plot generations where the Selected GRN produces a Phenotype (P) that has Converged to the Target (S)
    PLOT_NAME = "convergence.png"
    SAVELOC = os.path.join(OUTPUT_FOLDER, PLOT_NAME)
 
    if averaged:
        plot_trajectories(x = np.arange(len(converged)), trajectories = [np.asarray(converged, float)],
                          x_label = "Generation", y_label = "fraction of seeds converged", title = "Convergence",
                          subtitle = label, grid = True, save_loc = SAVELOC, figsize = (12, 4),
                          switch_lines = True, x_switches = switches)
        print(f"         wrote {PLOT_NAME} to {SAVELOC}")
 
    else:
        SAVELOC = os.path.join(OUTPUT_FOLDER, PLOT_NAME)
 
        plot_binary_strip(
            values = converged,
            true_colour = "green",
            false_colour = "red",
            true_label = "Converged",
            false_label = "Not Converged",
            x_label = "Generation",
            title = "Convergence of GRN over Generations",        
            subtitle = "Tracking Generations where the Selected GRN produces a Phenotype (P) that has Converged to the Target (S)",
            legend = True,
            save_loc = SAVELOC, 
            figsize = (16, 10),
            switch_plot = True,
            x_switches = switches
        )   
 
        print(f"         wrote {PLOT_NAME} to {SAVELOC}")   



    # Interaction matrix derived from Hebb's rule.
    PLOT_NAME = "interaction_heatmap.png"
    SAVELOC = os.path.join(OUTPUT_FOLDER, PLOT_NAME)    
    show_interaction_heatmap(
        B = B_heb, 
        title = "Hebbian interaction matrix", 
        save_loc = SAVELOC
    )
 
    print(f"         wrote {PLOT_NAME} to {SAVELOC}")  



    # The matrix of evolved regulatory interactions.
    PLOT_NAME = "hebbian_heatmap.png"
    SAVELOC = os.path.join(OUTPUT_FOLDER, PLOT_NAME)    
    B_heb = hebbian_interactions(cfg = cfg, S = cfg.targets)

    show_interaction_heatmap(
        B = B, 
        title = "Evolved interaction matrix B", 
        save_loc = SAVELOC
    )
 
    print(f"         wrote {PLOT_NAME} to {SAVELOC}")  


    # (E) 30 independent adult phenotypes from random G -> target pattern or its complement.
    show_phenotypes(
        B = B, 
        R = 30,
        cfg = cfg,
        rng = rng,
        title=  "Adult phenotypes from random G", 
        save_loc = f"{folder}/Figure1E.png"
    )

 
 
 
    # Plot the Comparative Fitnesses of the Initial Phenotype and Final/Induction Phenotype inside the Induction process
    if cfg.induction:
        PLOT_NAME = "induction_fitnesses.png"
        SAVELOC = os.path.join(OUTPUT_FOLDER, PLOT_NAME)
 
        plot_trajectories(
            x = recorded_gens,
            trajectories = [native_fitness, plastic_fitness],
            keys = ["Initial Phenotype Fitness", "Induction Phenotype Fitness"], 
            linestyle = ["-.", "--"], 
            marker = "x",
            x_label = "Generation",
            y_label = "Fitness",
            grid = True,
            title = "Comparative Fitnesses of the Initial Phenotype and Final/Induction Phenotype inside the Induction process",        
            subtitle = "Fitess is the Alignment of Phenotype (P) with the Target S not AUC",
            save_loc = SAVELOC, 
            moving_average = True,
            smooth = 0.1,
            figsize =(14, 8),
            switch_plot = True,
            switch_lines = True,
            x_switches = switches
        )      
 
        print(f"         wrote {PLOT_NAME} to {SAVELOC}")
 
 
        
 
        PLOT_NAME = "induction_fitness_change.png"
        SAVELOC = os.path.join(OUTPUT_FOLDER, PLOT_NAME)
 
        plot_trajectories(
            x = recorded_gens,
            trajectories = [F_change_inner, F_change_outer],
            keys = ["Change in Phenotype Fitness", "Change in Plastic AUC"], 
            linestyle = ["-.", "--"], 
            marker = ".",
            x_label = "Generation",
            y_label = "Change in Fitness",
            grid = True,
            title = "Comparative change in Fitness of the two Induction curves (Tracking Phenotype Fitness and Plastic AUC over R round respectively) over Generations",        
            subtitle = "Change in Fitness is the difference between Fitness at the start and end of the curve",
            save_loc = SAVELOC, 
            moving_average = True,
            smooth = 0.1,
            figsize =(14, 8),
            switch_plot = True,
            switch_lines = True,
            x_switches = switches
        )      
 
        print(f"         wrote {PLOT_NAME} to {SAVELOC}")
 
 
 
 
        PLOT_NAME = "induction_AUCs.png"
        SAVELOC = os.path.join(OUTPUT_FOLDER, PLOT_NAME)
 
        plot_trajectories(
            x = recorded_gens,
            trajectories = [AUC_inner, AUC_outer],
            keys = ["AUC of the Phenotype Fitness Curve", "AUC of the Plastic AUC Curve"], 
            linestyle = ["-.", "--"], 
            marker = "x",
            x_label = "Generation",
            y_label = "AUC",
            grid = True,
            title = "Comparative AUC of two Induction curves (Tracking Phenotype Fitness and Plastic AUC over R round respectively) over Generations",        
            subtitle = "AUC of the Plastic AUC curve is used as Fitness inside the Evolutionary process",
            save_loc = SAVELOC, 
            moving_average = True,
            smooth = 0.1,
            figsize =(14, 8),
            switch_plot = True,
            switch_lines = True,
            x_switches = switches
        )      
 
        print(f"         wrote {PLOT_NAME} to {SAVELOC}")
 
    
 
 
 
 
 
 
SUMMARY_KEYS = ("final_native_fitness", "mean_fitness", "mean_switch_loss",
                "modularity", "final_B_magnitude", "acceptance_rate")
 
 
 
 
def run_environment(env: str, targets: np.ndarray, ideal, figures_output: str,
                    n_generations: int, switch_every: int, record_every: int, seeds: int,
                    arms = ARMS):
    """
    Run every arm for `seeds` seeds on one environment.  Writes:
        <figures_output>/<arm>/seed_N/      per-seed diagnostic plots
        <figures_output>/<arm>/average/     the same plots on the seed-averaged series
        <figures_output>/<arm>/summary.png  per-seed values, mean and std for that arm
        <figures_output>/comparison.png     induction vs control on the averaged results
    """
    global N_GENERATIONS, SWITCH_EVERY, RECORD_EVERY
    N_GENERATIONS, SWITCH_EVERY, RECORD_EVERY = n_generations, switch_every, record_every
 
    os.makedirs(figures_output, exist_ok = True)
    averaged = {}
 
 
    for arm in arms:
        per_seed = []
 
 
        for seed in range(seeds):
            cfg = make_config(targets, induction = (arm == "induction"), seed = seed)
            results, _ = sswm_evolve(cfg, make_rng(cfg.seed), np.zeros((cfg.N, cfg.N)))
            results.update(summarise(results, cfg, ideal))
            per_seed.append(results)
            rng = make_rng(seed)
 
            folder = os.path.join(figures_output, arm, f"seed_{seed}")
            plot_run(results, cfg, folder, label = f"{env} / {arm} / seed {seed}", rng = rng)
 
 
        # ---- per-arm summary table: every seed, mean, std ----
        summary = arm_summary(per_seed, SUMMARY_KEYS)
        arm_summary_table(summary, arm, seeds,
                          save_loc = os.path.join(figures_output, arm, "summary.png"),
                          title = f"{env} environment, {n_generations} generations, switch every {switch_every}")
 
 
        # ---- seed-averaged results and plots for this arm ----
        avg = average_results(per_seed)
        avg["recorded_gens"] = per_seed[0]["recorded_gens"]       # shared x axis
        avg["switches"] = per_seed[0]["switches"]                  # same schedule every seed
 
 
        if "trajectories" not in avg:                              # 2-D, not averaged: show seed 0's
            avg["trajectories"] = per_seed[0]["trajectories"]
 
 
        averaged[arm] = avg
        
        rng = make_rng(DEFAULT_SEED)
 
        plot_run(avg, cfg, os.path.join(figures_output, arm, "average"),
                 label = f"{env} / {arm} / mean of {seeds} seeds", rng = rng)
 
 
    # ---- compare the arms on the averaged results ----
    if "induction" in averaged and "control" in averaged:
        rows = compare_arms(averaged["induction"], averaged["control"])
        keep = [k for k in SUMMARY_KEYS + ("fitnesses", "B_alignment", "G_alignment") if k in rows]
        comparison_table({k: rows[k] for k in keep},
                         save_loc = os.path.join(figures_output, "comparison.png"),
                         title = f"{env}: induction vs control, mean of {seeds} seeds, "
                                 f"{n_generations} generations, switch every {switch_every}")
        print(f"  wrote comparison.png to {figures_output}")
 
 
    return averaged
 