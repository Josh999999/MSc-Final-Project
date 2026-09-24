"""External Imports (Libraries and APIs)"""
import os  
import numpy as np
 
 
"""Local Imports"""
from Config import Config, make_rng, with_mask, ENERGY_GATES
from Data import S1
from Interactions import (appropriate_interactions, noisy_appropriate_interactions,
                          random_interactions, modular_interactions,
                          modular_appropriate_interactions)
from GRN import sparse_topology, diag_mask
from Plastic_Induction import plastic_search_return_wrapper
from GRN import handle_develop
from Plotting import create_search_table




FIGURES_OUTPUT = "Experiment9"




if __name__ == "__main__":
        # One config, constructed and validated up front. No ordering hazard: targets, N and the mask are checked against each other in __post_init__.
    cfg = Config(
        N = len(S1),
        targets = S1,
        induction = True,
    )
    
 
 
 
    # Replicable functionality for running the experiment
    def _experiment2(cfg: Config, n_seeds: int = 8):
 
        # Save the search data
        search_data = []
 
 
        # Run the Plastic search for a sweep of magnitudes and energy gate protocols
        for Y in [0.5, 1.0, 2.0, 6.0]:
            BY = B * Y
            
            # Develop the Phenotype as the cfg for the plastic search
            P = handle_develop(G, BY, cfg, induction = False)
 
            cfg = cfg.set(Y = Y)
 
 
            for gate in ENERGY_GATES:
 
                # Generates three new rows in the table for each magntiude Y
                row_data = [Y, gate]
                row_measurements = np.asarray([0] * (len(COLUMN_TITLES) - 2), dtype = float)

                # Reset the Energy gate
                cfg.energy_gate = gate


                for i in range(0, n_seeds):
 
                    # Perform the plastic search - This search doesn't alter B
                    # Fresh stream per gate, so the gates are compared on identical draws.
                    history = plastic_search_return_wrapper(B = BY, P = P, cfg = cfg, rng = make_rng(cfg.seed + i), limit_return = False)    
    
                    # Save the search data for the current row
                    data = []
                    data.append(history['acceptance_rate']) # Convert to a percentage in a string
                    data.append(history["auc_inner"])
                    data.append(history["F_change"])
                    data.append(history["align_change"])
                    data.append(history["avg_accept_A"])
                    data.append(history["std_accept_A"])

                    row_measurements += np.asarray(data, dtype = float)


                # Handle Inserting the measurements into the row as data

                # Take the mean of measurements accumulated across seeds
                row_measurements = row_measurements / n_seeds

                # Configure the acceptance rate for percentage display
                acceptance_rate = row_measurements[0]
                row_measurements = list(row_measurements)
                row_measurements[0] = f"{acceptance_rate * 100:.1f}%"

                row_data = row_data + row_measurements

 
                # Collect the row data
                search_data.append(row_data) 
 
 
        # Display the results of the experiment and analysis in a table
        create_search_table(
            column_tites = COLUMN_TITLES, 
            row_results = search_data, 
            save_loc = OUTPUT, 
            title = f"Effect of energy gates in the plastic search under {interaction_type}"
        )
 
 
 
 
    # !-- Sweep the interaction builders --! 
    def _dense(build):
 
        def wrapped(cfg, rng):
            cfg = with_mask(cfg, diag_mask(cfg))

 
            return build(cfg, rng), cfg
 
 
        return wrapped 

 
 
 
    def _sparse(build):
 
        def wrapped(cfg, rng):
            mask = sparse_topology(cfg, rng)
            sparse_cfg = with_mask(cfg, mask)
 
 
            return build(sparse_cfg, rng), sparse_cfg
 
 
        return wrapped
 

 
 
    # Appropriate / inappropriate crossed with dense, modular and sparse topologies
    BUILDERS = (
        ("appropriate interactions",
            _dense(lambda cfg, rng: appropriate_interactions(cfg = cfg, rng = rng, S = cfg.target,
                                                             inappropriate = False,
                                                             normalise = cfg.normalise_interactions))),
        ("inappropriate interactions",
            _dense(lambda cfg, rng: appropriate_interactions(cfg = cfg, rng = rng, S = cfg.target,
                                                             inappropriate = True,
                                                             normalise = cfg.normalise_interactions))),
        ("noisy appropriate interactions",
            _dense(lambda cfg, rng: noisy_appropriate_interactions(cfg = cfg, rng = rng, S = cfg.target,
                                                                   inappropriate = False,
                                                                   normalise = cfg.normalise_interactions))),
        ("noisy inappropriate interactions",
            _dense(lambda cfg, rng: noisy_appropriate_interactions(cfg = cfg, rng = rng, S = cfg.target,
                                                                   inappropriate = True,
                                                                   normalise = cfg.normalise_interactions))),
        ("random interactions",
            _dense(lambda cfg, rng: random_interactions(cfg = cfg, rng = rng,
                                                        normalise = cfg.normalise_interactions))),
 
        ("modular random interactions",
            _dense(lambda cfg, rng: modular_interactions(cfg = cfg, rng = rng,
                                                         normalise = cfg.normalise_interactions))),
        ("modular appropriate interactions",
            _dense(lambda cfg, rng: modular_appropriate_interactions(cfg = cfg, rng = rng, S = cfg.target,
                                                                     inappropriate = False,
                                                                     normalise = cfg.normalise_interactions))),
        ("modular inappropriate interactions",
            _dense(lambda cfg, rng: modular_appropriate_interactions(cfg = cfg, rng = rng, S = cfg.target,
                                                                     inappropriate = True,
                                                                     normalise = cfg.normalise_interactions))),
 
        ("sparse random interactions",
            _sparse(lambda cfg, rng: random_interactions(cfg = cfg, rng = rng,
                                                         normalise = cfg.normalise_interactions))),
        ("sparse appropriate interactions",
            _sparse(lambda cfg, rng: appropriate_interactions(cfg = cfg, rng = rng, S = cfg.target,
                                                              inappropriate = False,
                                                              normalise = cfg.normalise_interactions))),
        ("sparse inappropriate interactions",
            _sparse(lambda cfg, rng: appropriate_interactions(cfg = cfg, rng = rng, S = cfg.target,
                                                              inappropriate = True,
                                                              normalise = cfg.normalise_interactions))),
    )
 
    os.makedirs(FIGURES_OUTPUT or ".", exist_ok = True)
 
 
    for interaction_type, build in BUILDERS:
 
        # Same seed for every builder so the comparison is like for like.
        B, run_cfg = build(cfg, make_rng(cfg.seed))
 
        OUTPUT = os.path.join(FIGURES_OUTPUT or ".",
                              f"plastic_search_table_{interaction_type.replace(' ', '_')}.png")
 
        print(f"running: {interaction_type}")
        _experiment2(run_cfg)
        print(f"  wrote {OUTPUT}")
 