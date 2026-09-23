"""External Imports (Libraries and APIs)"""
import os  
 
 
"""Local Imports"""
from Config import Config, make_rng, with_mask, ENERGY_GATES
from Data import S1
from Interactions import (appropriate_interactions,
                          random_interactions,
                          noisy_appropriate_interactions,
                          modular_interactions,
                          modular_appropriate_interactions
                          )
from GRN import sparse_topology, masked_matrix, diag_mask
from Interactions import adjust_interaction_magnitude
from Plastic_Induction import plastic_search_return_wrapper
from GRN import handle_develop
from Plotting import create_search_table
 
 
 
 
 
FIGURES_OUTPUT = "Experiment5"




if __name__ == "__main__":
 
    # One config, constructed and validated up front. No ordering hazard: targets, N and the mask are checked against each other in __post_init__.
    cfg = Config(
        N = len(S1),
        targets = S1,
        induction = True,
    )
 
    MUTATIONS = [25, 50, 100, 250, 400]
 
    # Generate the starting profile (Constant used for all interaction matricies)
    G = make_rng(cfg.seed).uniform(low = -1, high = 1, size = cfg.N)


    # Create the masks
    masks = []
    SPARSITIES = [1, 2, 4, 6, cfg.N - 1]
 
    for K in SPARSITIES:
        cfg.set(K = K)         
        mask = sparse_topology(cfg, make_rng(cfg.seed))
        masks.append(mask)
    
 
 

    # Replicable functionality for running the experiment
    def _experiment5(cfg: Config, measurement: str = "align_change", gate: str = "or", n_seeds: int = 8):

        # Create the column titles using the name of the measurement
        measurement_str = measurement.replace('_', ' ').strip().lower().capitalize()

        MUTATION_TITLES = tuple(f"{measurement_str}\nM = {M}" for M in MUTATIONS)
 
        COLUMN_TITLES = (
            "Sparsity (K)",
            "Density",
        ) + MUTATION_TITLES

    
        # Save the search data
        search_data = []

    
        # Set the Energy gate
        cfg = cfg.set(energy_gate = gate)
 
 
        for mask, K in zip(masks, SPARSITIES):

            # Config is a frozen dataclass: build a new one per cell.
            cfg.set(mask = mask, K = K)
 
            # Re-mask B to this topology
            BK = masked_matrix(B, mask)
 
            if cfg.normalise_interactions:
                BK = adjust_interaction_magnitude(BK, cfg, Y = 1.0)
 
             
            # Run the Plastic search for a sweep of energy gate protocols
            P = handle_develop(G, BK, cfg, induction = False) # Develop the Phenotype as the cfg for the plastic search


            # Generate new row in the table
            density = float(mask.sum()) / float(cfg.N * (cfg.N - 1)) if cfg.self_interaction else float(mask.sum()) / float(cfg.N ** 2)
            row_data = [K, density]


            # Save the search data for the current row for the alignment under different mutation rates
            for M in MUTATIONS:

                # Alter the mutation rate
                cfg.M = M


                # Run the experiment multiple times for robustness
                a = 0

                for i in range(0, n_seeds):

                    # Perform the plastic search - This search doesn't alter B; Fresh stream per gate, so the gates are compared on identical draws.
                    history = plastic_search_return_wrapper(B = BK, P = P, cfg = cfg, rng = make_rng(cfg.seed + i), limit_return = False)
                    a += history[measurement]


                # Append the measurement for the mutation rate (M)
                data = a / n_seeds
                row_data.append(data)


            # Collect the row data
            search_data.append(row_data) 

    
        # Display the results of the experiment and analysis in a table
        create_search_table(
            column_tites = COLUMN_TITLES, 
            row_results = search_data, 
            save_loc = OUTPUT, 
            title = f"Joint Effect of Sparsity and Mutation Rate in the plastic search under {interaction_type} ({gate} Gate)")

 
 
 
 
    def _dense(build):
 
        def wrapped(cfg, rng):            
            cfg = with_mask(cfg, diag_mask(cfg))
            
 
            return build(cfg, rng)
 
 
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
    )
 
    os.makedirs(FIGURES_OUTPUT or ".", exist_ok = True)
 
 
    for interaction_type, build in BUILDERS:

        for gate in ENERGY_GATES: 
            # Same seed for every builder so the comparison is like for like.
            B = build(cfg, make_rng(cfg.seed))
    
            OUTPUT = os.path.join(FIGURES_OUTPUT or ".", f"plastic_search_table_{interaction_type.replace(' ', '_')}_{gate}Gate.png")
    
            print(f"running: {interaction_type}")
            _experiment5(cfg, measurement = "F_change", gate = gate)
            print(f"  wrote {OUTPUT}")
    