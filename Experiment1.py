"""External Imports (Libraries and APIs)"""
import os


"""Local Imports"""
from Config import Config, make_rng
from Data import S1
from Tenets import fitness_surface
from Plotting import plot_fitness_surface




FIGURES_OUTPUT = "Experiment1"




if __name__ == "__main__":

    # One config, constructed and validated up front. No ordering hazard: targets, N and the mask are checked against each other in __post_init__.
    cfg = Config(
        N = len(S1),
        targets = S1,
    )


    # (key, axis label, colour scale).  energy and diff_energy are SIGNED, so they need a diverging scale centred on zero; fitness is non-negative.
    N_SEEDS = 8

    MEASURES = (
        ("fitness",     "Fitness",                        "sequential"),
        ("energy",      "Energy  -0.5 P.B.P",             "diverging"),
        ("diff_energy", "Differential energy (G -> P)",   "diverging"),
    )

    for measure, label, scale in MEASURES:
        surfaces = []


        for Y in [0.5, 1.0, 2.0, 6.0]:
            cfg = cfg.set(Y = Y)
            rng = make_rng(cfg.seed)

            result = fitness_surface(
                cfg, 
                rng, 
                n_seeds = N_SEEDS,
                amplitude = 1.0,
                grid = 21,
                measure = measure
            )
            surfaces.append(result)


        # Handle the output folder
        OUTPUT = os.path.join(FIGURES_OUTPUT or ".", f"tenet_{measure}_surface.png")
        os.makedirs(FIGURES_OUTPUT, exist_ok = True)

        plot_fitness_surface(
            surfaces,
            save_loc = OUTPUT,
            label_measure = label,
            scale = scale,
            n_seeds = N_SEEDS
        )


    print(f"running: Experiment1")
    print(f"  wrote {OUTPUT}")