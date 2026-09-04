"""External Imports (Libraries and APIs)"""
from dataclasses import replace
import numpy as np
import os  
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


"""Local Imports"""
from Config import Config, make_rng, ENERGY_GATES
from Data import S1
from Interactions import appropriate_interactions
from Plastic_Induction import plastic_search
from GRN import handle_develop




def create_plastic_search_table(column_tites: np.ndarray, row_results: np.ndarray, interaction_type: str, save_loc: str):
    interaction_type = interaction_type.strip().lower().capitalize()

    fig, ax = plt.subplots(figsize=(6, 2))

    tbl = ax.table(
        cellText=[[f"{v:.2f}" if isinstance(v, float) else v for v in r] for r in row_results],
        colLabels = column_tites,
        loc="center", 
        cellLoc="center"
    )

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.4)
    ax.set_title(f"Effect of energy gates in the Platic search under a {interaction_type}", pad=20, fontweight="bold")

    ax.axis("off")
    fig.tight_layout()
    fig.savefig(save_loc, dpi=150, bbox_inches="tight")








if __name__ == "__main__":

    # One config, constructed and validated up front. No ordering hazard:
    # targets, N and the mask are checked against each other in __post_init__.
    base = Config(
        N = len(S1),
        targets = S1,
        induction = True,
        interactions_norm = True,
        figures_output = "Experiment3"
    )

    COLUMN_TITLES = (
        "Magnitude (Y)",
        "Energy Gate",
        "Acceptance rate",
        "AUC",
        "Accepted Phenotype avg. Fitness" ,
        "Last Phenotype Fitness",
        "Accepted Phenotype avg. Energy",
        "Accepted Phenotype std. of Energy",
        "Accepted Phenotype avg. Energy Acceptance Probability",
        "Accepted Phenotype std. of Energy Acceptance Probability",
    )


    # Generate the starting profile (Constant used for all interaction matricies)
    G = np.random.uniform(low = -1.0, high = 1.0, size = base.N)
    



    """Test the Plastic search for a range of different initialised interaction matricies"""

    # Replicable functionality for running the experiment
    def _experiment3():

        # Save the search data
        search_data = []

        # Develop the Phenotype as the base for the plastic search
        P = handle_develop(G, B, base, induction = False)


        # Run the Plastic search for a sweep of magnitudes and energy gate protocols
        for Y in [0.5, 1.0, 2.0, 6.0]:
            cfg = replace(base, Y = Y)
            rng = make_rng(cfg.seed)        # same stream per panel, so panels are comparable


            for gate in ENERGY_GATES:

                # Generates three new rows in the table for each magntiude Y
                row_data = []

                # Reset the Energy gate
                cfg = replace(cfg, energy_gate = gate)

                # Perform the plastic search
                history = plastic_search(B = B.copy() * Y, P = P, cfg = cfg, rng = rng)


                # Save the search data for the current row
                row_data.append(Y)
                row_data.append(gate)
                row_data.append(history["acceptance_rate"])
                row_data.append(history["auc"])
                row_data.append(history["avg_accept_F"])
                row_data.append(history["F"])
                row_data.append(history["avg_accept_E"])
                row_data.append(history["std_accept_E"])
                row_data.append(history["avg_accept_w"])
                row_data.append(history["std_accept_w"])


                # Collect the row data
                search_data.append(row_data) 


        # Display the results of the experiment and analysis in a table
        create_plastic_search_table(COLUMN_TITLES, search_data, interaction_type, OUTPUT)




    # !-- Appropriate Interactions --!

    # Test the energy gate on the Plastic search function (using appropriate interactions)

    # Set-up
    rng = make_rng(base.seed)
    B = appropriate_interactions(cfg = base, rng = rng, S = base.target, inappropriate = False, normalise = base.interactions_norm)
    interaction_type = "appropriate interactions"

    # Handle the output folder
    OUTPUT = os.path.join(base.figures_output or ".", f"plastic_search_table_{interaction_type}.png")
    os.makedirs(base.figures_output, exist_ok = True)

    # Run the experiment
    _experiment3()