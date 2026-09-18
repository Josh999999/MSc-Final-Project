"""External Imports (Libraries and APIs)"""
import os
 
 
"""Local Imports"""
from Config import Config, make_rng, ENERGY_GATES
from Data import S1
from Tenets import plastic_measure_surface
from Plotting import plot_measure_surfaces

 
 
 
# Where this experiment writes its figures.  Kept out of Config: it is a
# property of the SCRIPT, not of the model being configured.
FIGURES_OUTPUT = "Experiment3"




if __name__ == "__main__":
 
    # What the plastic search reports, and how each should be coloured.
    #   diverging : signed quantity, centred on zero
    #   sequential: non-negative quantity
    MEASURES = (
        # diverging only where the measure is genuinely SIGNED; a zero-centred
        # scale on a non-negative measure wastes half the colormap.
        ("auc_inner",             "Plastic AUC (mean fitness over the search)", "sequential"),
        ("F_change",        "Fitness change over the search",             "diverging"),
        ("acceptance_rate", "Acceptance rate",                            "sequential"),
        ("align_change",    "Change in cos(P (x) P, B)",                  "diverging"),
        ("avg_accept_A",    "Mean alignment of accepted phenotypes",      "diverging"),
    )
 
    cfg = Config(
        N = len(S1),
        targets = S1,
        induction = True,
        induction_process = "plastic",
        T = 10,
        normalise_interactions = True,
        normalise_energy = False,
        energy_type = "differential",
        M = 100,
        fitness_type = "standard",
        mutation_type = "phenotype"
    )
 
    os.makedirs(FIGURES_OUTPUT, exist_ok = True)
 
 
    # One figure per (gate, measure): panels across the magnitude sweep, each
    # panel a heatmap over Tenet 1 (x) by Tenet 2 (y).
    for gate in ENERGY_GATES:
 
        for key, label, scale in MEASURES:
 
            surfaces = []
 
 
            for Y in [0.5, 1.0, 2.0, 6.0]:
                cfg = cfg.set(Y = Y, energy_gate = gate)
 
                # Same stream per panel so the panels are comparable.
                result = plastic_measure_surface(
                    cfg, make_rng(cfg.seed),
                    measure = key, n_seeds = 8, grid = 15
                )
                surfaces.append(result)
 
  
            OUTPUT = os.path.join(FIGURES_OUTPUT,
                                  f"tenet_{gate}_{key}.png")
 
            plot_measure_surfaces(
                surfaces,
                saveloc = OUTPUT,
                label_measure = label,
                scale = scale,
                subtitle = f"energy gate: {gate}",
                colour_scale = "symlog"
            )
 
            span = [f"{s['Z'].min():+.4f}..{s['Z'].max():+.4f}" for s in surfaces]
            print(f"{gate:>6} | {key:<16} ranges per Y: {'  '.join(span)}")
            print(f"         wrote {OUTPUT}")
 