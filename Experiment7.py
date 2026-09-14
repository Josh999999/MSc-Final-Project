"""External Imports (Libraries and APIs)"""
from dataclasses import replace
import os
 
 
"""Local Imports"""
from Config import Config, make_rng, ENERGY_GATES
from Data import S1
from Tenets import plastic_measure_surface
from Plotting import plot_measure_surfaces
from Interactions import appropriate_interactions
from Plastic_Induction import plastic_search
from GRN import develop

 
 
 
if __name__ == "__main__":
 
    # What the plastic search reports, and how each should be coloured.
    #   diverging : signed quantity, centred on zero
    #   sequential: non-negative quantity
    MEASURES = (
        ("auc",             "Plastic AUC (mean fitness over the search)", "diverging"),
        ("F_change",        "Fitness change over the search",             "diverging"),
        ("acceptance_rate", "Acceptance rate",                            "diverging"),
        ("align_change",    "Change in cos(P (x) P, B)",                  "diverging"),
        ("avg_accept_A",    "Mean alignment of accepted phenotypes",      "diverging"),
    )
 
    base = Config(
        N = len(S1),
        targets = S1,
        induction = True,
        induction_process = "plastic",
        figures_output = "Experiment7",
        T = 6,
        normalise_interactions = True,
        normalise_energy = False,
        energy_type = "differential",
        M = 100,
        fitness_type = "cosine",
        mutation_type = "phenotype",
        mutation_operation = "additive",
        limit_fitness = True,
        normalise_fitness = True,
        Y = 1.0,
        energy_gate = "deterministic",
    )

    rng = make_rng(base.seed)
 

    B = appropriate_interactions(base, rng, S1, inappropriate = False, normalise = base.normalise_interactions)

    G = -S1.copy()

    P = develop(G, B, base, base.T)

    print('\n', G, '\n')
    print('\n', P, '\n')

    plastic_search(B, P, base, rng, S1)


    surfaces = []

    for Y in [0.25, 0.5, 1.0, 2.0, 6.0]:

        base = replace(base, Y = Y)

        # Same stream per panel so the panels are comparable.
        result = plastic_measure_surface(
            base, make_rng(base.seed),
            measure = 'auc', n_seeds = 8, grid = 15
        )
        surfaces.append(result)


    OUTPUT = os.path.join(base.figures_output, f"fitness_surface.png")

    os.makedirs(base.figures_output, exist_ok = True)

    plot_measure_surfaces(
        surfaces,
        saveloc = OUTPUT,
        label_measure = 'AUC',
        scale = 'sequential',
        subtitle = f"energy gate: Deterministic",
        colour_scale = "percentile"
    )