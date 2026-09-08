"""External Imports (Libraries and APIs)"""
from dataclasses import replace
import os


"""Local Imports"""
from Config import Config, make_rng
from Data import S1
from Tenets import fitness_surface
from Plotting import plot_fitness_surface




if __name__ == "__main__":

    # One config, constructed and validated up front. No ordering hazard:
    # targets, N and the mask are checked against each other in __post_init__.
    base = Config(
        N = len(S1),
        targets = S1,
        induction = True,
        induction_process = "r-round",
        figures_output = "Experiment4"
    )

    os.makedirs(base.figures_output, exist_ok = True)


    for measure in ['auc', 'F_change']:
        surfaces = [] # Reset the surface to make a new one for each measurement

        for Y in [0.5, 1.0, 2.0, 6.0]:
            cfg = replace(base, Y = Y)
            rng = make_rng(cfg.seed)        # same stream per panel, so panels are comparable

            result = fitness_surface(cfg, rng, n_seeds = 8, induction_measure = measure)
            surfaces.append(result)


        # Handle the output folder (for each measure)
        OUTPUT = os.path.join(cfg.figures_output or ".", f"tenet_{measure}_surface.png")

        plot_fitness_surface(surfaces, saveloc = OUTPUT)