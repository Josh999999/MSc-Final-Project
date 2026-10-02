"""External Imports (Libraries and APIs)"""
import os
 
 
"""Local Imports"""
from Config import Config, make_rng, ENERGY_GATES
from Data import S1
from Tenets import plastic_measure_surface
from Plotting import plot_measure_surfaces

 
 
 
FIGURES_OUTPUT = "Experiment4"




if __name__ == "__main__":
 
    MEASURES = (
        ("auc_inner",             "Plastic AUC (mean fitness over the search)",         "sequential"),
        ("auc_outer",             "Relaxed AUC (mean fitness over the search)",         "sequential"),
        ("F_change_inner",        "Plastic Fitness change over the search",             "diverging"),
        ("F_change_outer",        "Relaxed Fitness change over the search",             "diverging"),
        ("align_change",          "Change in cos(P (x) P, B)",                          "diverging"),
    )
    
    cfg = Config(
        N = len(S1),
        targets = S1,
        induction = True,
        induction_process = "r-round",
        energy_gate = "deterministic"
    )
 
    os.makedirs(FIGURES_OUTPUT, exist_ok = True)
 
 
    # One figure per (gate, measure): panels across the magnitude sweep, each panel a heatmap over Tenet 1 (x) by Tenet 2 (y). 
    for key, label, scale in MEASURES:

        surfaces = []


        for Y in [0.5, 1.0, 2.0, 6.0]:

            # Same stream per panel so the panels are comparable.
            result = plastic_measure_surface(
                cfg, make_rng(cfg.seed),
                measure = key, n_seeds = 8, grid = 15
            )
            surfaces.append(result)


        OUTPUT = os.path.join(FIGURES_OUTPUT,
                                f"tenet_deterministic_{key}.png")

        plot_measure_surfaces(
            surfaces,
            save_loc = OUTPUT,
            label_measure = label,
            scale = scale,
            subtitle = "energy gate: deterministic",
            colour_scale = "percentile"
        )

        span = [f"{s['Z'].min():+.4f}..{s['Z'].max():+.4f}" for s in surfaces]
        print(f"deterministic | {key:<16} ranges per Y: {'  '.join(span)}")
        print(f"         wrote {OUTPUT}")
