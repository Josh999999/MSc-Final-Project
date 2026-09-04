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
        figures_output = "Experiment2"
    )

    surfaces = []


    for Y in [0.5, 1.0, 2.0, 6.0]:
        cfg = replace(base, Y = Y)
        rng = make_rng(cfg.seed)        # same stream per panel, so panels are comparable

        result = fitness_surface(cfg, rng, n_seeds = 8)
        surfaces.append(result)

        print(
            f"Y={Y:>4}  tenet1 err {result['max_tenet1_error']:.2e}  "
            f"tenet2 err {result['max_tenet2_error']:.2e}  "
            f"fitness range [{result['Z'].min():+.2f}, {result['Z'].max():+.2f}]"
        )


    # Handle the output folder
    OUTPUT = os.path.join(cfg.figures_output or ".", "tenet_surface_induction.png")
    os.makedirs(cfg.figures_output, exist_ok = True)

    plot_fitness_surface(surfaces, saveloc = OUTPUT)
    print("wrote tenet_surface_induction.png")
