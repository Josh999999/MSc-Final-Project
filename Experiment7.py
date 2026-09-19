"""External Imports (Libraries and APIs)"""
import os
import numpy as np
 
 
"""Local Imports"""
from Config import Config, make_rng, set_mask
from Data import S1, S2, DEFAULT_SEED
from Plotting import plot_trajectories, plot_binary_strip, plot_transfer
from Evolution import sswm_evolve
from GRN import diag_mask

 
 
 
FIGURES_OUTPUT = "Experiment7"
N_GENERATIONS = 50_000
SWITCH_EVERY  = 2_000
RECORD_EVERY  = 200
SEEDS         = 5




if __name__ == "__main__": 
    
    # Two targets: switches, and everything plotted about transfer, need M > 1.
    cfg = Config(
        N = len(S1),
        targets = np.array([S1, S2]),
        induction = True,
        n_generations = N_GENERATIONS,
        switch_every = SWITCH_EVERY,
        record_every = RECORD_EVERY,
    )
 
    os.makedirs(FIGURES_OUTPUT, exist_ok = True)

    OUTPUT_FOLDER = "./"
    PLOT_NAME = "Plot.png"
 
 
    # One figure per (gate, measure): panels across the magnitude sweep, each
    # panel a heatmap over Tenet 1 (x) by Tenet 2 (y).
    for seed in range(0, SEEDS):
        
        print(f"run{seed}")

        OUTPUT_FOLDER = os.path.join(FIGURES_OUTPUT, f"seed_{seed}")
        os.makedirs(OUTPUT_FOLDER, exist_ok = True)

        print(f"         writing to: {OUTPUT_FOLDER}")


        # Make new random generator for the run
        rng = make_rng(seed = DEFAULT_SEED + seed)

        # Initialise the interaction matrix (Zeros)
        B = np.zeros(shape = (cfg.N, cfg.N))

        # Create mask for the matrix (no sparsity)
        mask = diag_mask(cfg)
        set_mask(cfg, mask)

        # Run the evolutionary algorithm
        results, _ = sswm_evolve(cfg, rng, B)


        # Extract arrays from the results
        trajectories = results["trajectories"]
        recorded_gens = results["recorded_gens"]
        fitnesses = results["fitnesses"]
        selections = results["selections"]
        switches = results["switches"]
        transfer_fitness = results["transfer_fitness"]        
        transfer_loss = results["transfer_loss"]        
        G_alignment = results["G_alignment"]
        B_alignment = results["B_alignment"]
        B_magnitude = results["B_magnitude"]
        converged = results["converged"] 
        native_fitness = results["native_fitness"]
        plastic_fitness = results["plastic_fitness"]
        F_change_inner = results["F_change_inner"]
        F_change_outer = results["F_change_outer"]
        AUC_inner = results["AUC_inner"]
        AUC_outer = results["AUC_outer"]




        # Plot the development of the interaction trajectories
        PLOT_NAME = "interaction_trajectories.png"
        SAVELOC = os.path.join(OUTPUT_FOLDER, PLOT_NAME)

        plot_trajectories(
            trajectories = trajectories,
            x_label = "Recorded Generation",
            y_label = "Interaction size",
            title = "Trajectories of the Weights inside the Interaction Matrix",        
            saveloc = SAVELOC, 
            figsize = (6, 4)
        )      

        print(f"         wrote {PLOT_NAME} to {SAVELOC}")



        # Plot the Fitness of the GRN over evolution (generations)
        PLOT_NAME = "fitness.png"
        SAVELOC = os.path.join(OUTPUT_FOLDER, PLOT_NAME)

        plot_trajectories(
            trajectories = [fitnesses],
            marker = "x",
            x_label = "Generation",
            y_label = "Fitness (AUC)",
            title = "Fitness of the chosen GRN over Generations",        
            subtitle = "Fitness is the AUC of the selected GRN from the Induction Process",
            grid = True,
            saveloc = SAVELOC, 
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
            transfer_fitness = transfer_fitness, 
            fitnesses = fitnesses, 
            transfer_loss = transfer_loss, 
            switches = switches,  
            saveloc = SAVELOC
        )



        # Plot generations where the mutated GRN is selected
        PLOT_NAME = "selections.png"
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
            saveloc = SAVELOC, 
            figsize = (16, 10)
        )   

        print(f"         wrote {PLOT_NAME} to {SAVELOC}")   


        
        # Plot the alignment of the elements (G and B) of the selected GRN with the target (T) over evolution (generations)
        PLOT_NAME = "grn_alignment.png"
        SAVELOC = os.path.join(OUTPUT_FOLDER, PLOT_NAME)

        plot_trajectories(
            trajectories = [G_alignment, B_alignment],
            keys = ["Genotype Alignment", "Interaction Alignment"], 
            linestyle = ["-.", "--"], 
            marker = "x", 
            grid = True,
            x_label = "Generation",
            y_label = "Alignment (cosine alignmet with S))",
            title = "Genotype (G) and Interaction (B) Alignment with the Target (S)",        
            subtitle = "Tracks the Alignment of the Genotype (G) and Interaction (B) selected",
            saveloc = SAVELOC, 
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
            trajectories = [B_magnitude],
            marker = "x",
            grid = True,
            x_label = "Generation",
            y_label = "Interaction Matrix (B) Magnitude (Y)",
            title = "Magnitude of the Selected Interaction Matrix (B) over Generations",        
            subtitle = "Interaction Matrix (B) is from the Selected GRN",
            saveloc = SAVELOC, 
            figsize = (12, 6)
        )   

        print(f"         wrote {PLOT_NAME} to {SAVELOC}")  


        
        # Plot generations where the Selected GRN produces a Phenotype (P) that has Converged to the Target (S)
        PLOT_NAME = "convergence.png"
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
            saveloc = SAVELOC, 
            figsize = (16, 10),
            switch_plot = True,
            x_switches = switches
        )   

        print(f"         wrote {PLOT_NAME} to {SAVELOC}")   



        # Plot the Comparative Fitnesses of the Initial Phenotype and Final/Induction Phenotype inside the Induction process
        PLOT_NAME = "induction_fitnesses.png"
        SAVELOC = os.path.join(OUTPUT_FOLDER, PLOT_NAME)

        plot_trajectories(
            trajectories = [native_fitness, plastic_fitness],
            keys = ["Initial Phenotype Fitness", "Induction Phenotype Fitness"], 
            linestyle = ["-.", "--"], 
            marker = "x",
            x_label = "Generation",
            y_label = "Fitness",
            grid = True,
            title = "Comparative Fitnesses of the Initial Phenotype and Final/Induction Phenotype inside the Induction process",        
            subtitle = "Fitess is the Alignment of Phenotype (P) with the Target S not AUC",
            saveloc = SAVELOC, 
            moving_average = True,
            smooth = 0.1,
            figsize =(14, 8),
            switch_plot = True,
            switch_lines = True,
            x_switches = switches
        )      

        print(f"         wrote {PLOT_NAME} to {SAVELOC}")


        
        # Plot the Comparative change in Fitness of the two Induction curves (Tracking Phenotype Fitness and Plastic AUC over R round respectively) over Generations
        PLOT_NAME = "induction_fitness_change.png"
        SAVELOC = os.path.join(OUTPUT_FOLDER, PLOT_NAME)

        plot_trajectories(
            trajectories = [F_change_inner, F_change_outer],
            keys = ["Change in Phenotype Fitness", "Change in Plastic AUC"], 
            linestyle = ["-.", "--"], 
            marker = ".",
            x_label = "Generation",
            y_label = "Change in Fitness",
            grid = True,
            title = "Comparative change in Fitness of the two Induction curves (Tracking Phenotype Fitness and Plastic AUC over R round respectively) over Generations",        
            subtitle = "Change in Fitness is the difference between Fitness at the start and end of the curve",
            saveloc = SAVELOC, 
            moving_average = True,
            smooth = 0.1,
            figsize =(14, 8),
            switch_plot = True,
            switch_lines = True,
            x_switches = switches
        )      

        print(f"         wrote {PLOT_NAME} to {SAVELOC}")



        # Plot the Comparative AUC of two Induction curves (Tracking Phenotype Fitness and Plastic AUC over R round respectively) over Generations
        PLOT_NAME = "induction_AUCs.png"
        SAVELOC = os.path.join(OUTPUT_FOLDER, PLOT_NAME)

        plot_trajectories(
            trajectories = [AUC_inner, AUC_outer],
            keys = ["AUC of the Phenotype Fitness Curve", "AUC of the Plastic AUC Curve"], 
            linestyle = ["-.", "--"], 
            marker = "x",
            x_label = "Generation",
            y_label = "AUC",
            grid = True,
            title = "Comparative AUC of two Induction curves (Tracking Phenotype Fitness and Plastic AUC over R round respectively) over Generations",        
            subtitle = "AUC of the Plastic AUC curve is used as Fitness inside the Evolutionary process",
            saveloc = SAVELOC, 
            moving_average = True,
            smooth = 0.1,
            figsize =(14, 8),
            switch_plot = True,
            switch_lines = True,
            x_switches = switches
        )      

        print(f"         wrote {PLOT_NAME} to {SAVELOC}")

        
    print(f"         wrote to {FIGURES_OUTPUT}")