"""External Imports (Libraries and APIs)"""
from dataclasses import replace
import os  
 
 
"""Local Imports"""
from Config import Config, make_rng, with_mask, ENERGY_GATES
from Data import S1
from Interactions import (appropriate_interactions,
                          random_interactions)
from GRN import sparse_topology, masked_matrix, diag_mask
from Interactions import adjust_interaction_magnitude
from Plastic_Induction import plastic_search_return_wrapper
from GRN import handle_develop
from Plotting import create_search_table
 
 
 
 
 
 
 
if __name__ == "__main__":
 
    # One config, constructed and validated up front. No ordering hazard:
    # targets, N and the mask are checked against each other in __post_init__.
    base = Config(
        N = len(S1),
        targets = S1,
        induction = True,
        interactions_norm = True,
        figures_output = "Experiment6"
    )
 
    MUTATIONS = [25, 50, 100, 250, 400]
 
    # Generate the starting profile (Constant used for all interaction matricies)
    G = make_rng(base.seed).uniform(low = -1, high = 1, size = base.N)
 
 
    # Create the masks
    masks = []
    # K is the out-degree BEFORE symmetrisation, so it cannot exceed N-1.
    # K = N-1 is fully dense; small K leaves few active interactions.
    SPARSITIES = [1, 2, 4, 6, base.N - 1]
 
    for K in SPARSITIES:
        base_K = replace(base, K = K)
        mask = sparse_topology(base_K, make_rng(base.seed))
        masks.append(mask)
    
 
 
 
    """Test the Plastic search for a range of different initialised interaction matricies"""
 
    # Replicable functionality for running the experiment
    def _experiment6(base: Config, measurement: str = "align_change", gate: str = "or", n_seeds: int = 8):

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
        cfg = replace(base, energy_gate = gate)
 
 
        for mask, K in zip(masks, SPARSITIES):

            # Config is a frozen dataclass: build a new one per cell.
            cfg = replace(with_mask(cfg, mask), K = K)
 
            # Re-mask B to this topology
            BK = masked_matrix(B, mask)
 
            if cfg.interactions_norm:
                BK = adjust_interaction_magnitude(BK, cfg, Y = 1.0)
 
             
            # Run the Plastic search for a sweep of energy gate protocols
            P = handle_develop(G, BK, cfg, induction = False) # Develop the Phenotype as the base for the plastic search


            # Generate new row in the table
            density = float(mask.sum()) / float(cfg.N * (cfg.N - 1)) if cfg.self_interaction else float(mask.sum()) / float(cfg.N ** 2)
            row_data = [K, density]


            # Save the search data for the current row for the alignment under different mutation rates
            for M in MUTATIONS:

                # Alter the mutation rate
                cfg = replace(cfg, M = M)


                # Run the experiment multiple times for robustness
                a = 0

                for i in range(0, n_seeds):

                    # Perform the plastic search - This search doesn't alter B; Fresh stream per gate, so the gates are compared on identical draws.
                    history = plastic_search_return_wrapper(B = BK, P = P, cfg = cfg, rng = make_rng(cfg.seed + i), limit_return = True)
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

 
 
 
    # !-- Sweep the interaction builders --!
    #
    # The builder controls how well B already encodes the target, i.e. Tenet 1.
    # Appropriate interactions make development converge onto the target, so the
    # search starts at F = 1 and has no headroom; the noisy and random builders
    # leave room for the gates to differ.
 
    # Each builder returns (B, cfg).  Most keep the default dense mask, but the
    # sparse case must change the CONFIG as well, because the mask governs which
    # entries the search, the energy and the alignment measure all look at.
 
    def _dense(build):
        """Builder that simply returns the matrix (mask is set later)."""
 
        def wrapped(cfg, rng):
            
            if cfg.self_interaction:
                cfg = with_mask(cfg, diag_mask(cfg.N))
            
 
            return build(cfg, rng)
 
 
        return wrapped
 
 
 
 
    # Appropriate / inappropriate crossed with dense, modular and sparse
    # topologies: the topology sets the structure, the appropriateness sets
    # whether B points toward the target or away from it.
    BUILDERS = (
        ("appropriate interactions",
            _dense(lambda cfg, rng: appropriate_interactions(cfg = cfg, rng = rng, S = cfg.target,
                                                             inappropriate = False,
                                                             normalise = cfg.interactions_norm))),
        ("inappropriate interactions",
            _dense(lambda cfg, rng: appropriate_interactions(cfg = cfg, rng = rng, S = cfg.target,
                                                             inappropriate = True,
                                                             normalise = cfg.interactions_norm))),
        ("random interactions",
            _dense(lambda cfg, rng: random_interactions(cfg = cfg, rng = rng,
                                                        normalise = cfg.interactions_norm))),
    )
 
    os.makedirs(base.figures_output or ".", exist_ok = True)
 
 
    for interaction_type, build in BUILDERS:

        for gate in ENERGY_GATES: 
            # Same seed for every builder so the comparison is like for like.
            B = build(base, make_rng(base.seed))
    
            OUTPUT = os.path.join(base.figures_output or ".", f"plastic_search_table_{interaction_type.replace(' ', '_')}_{gate}Gate.png")
    
            print(f"running: {interaction_type}")
            _experiment6(base, measurement = "F_change", gate = gate)
            print(f"  wrote {OUTPUT}")
    