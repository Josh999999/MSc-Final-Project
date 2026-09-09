"""External Imports (Libraries and APIs)"""
from dataclasses import replace
import numpy as np
import os  
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
 
 
"""Local Imports"""
from Config import Config, make_rng, with_mask, ENERGY_GATES
from Data import S1
from Interactions import (appropriate_interactions, noisy_appropriate_interactions,
                          random_interactions, modular_interactions,
                          modular_appropriate_interactions)
from GRN import sparse_topology, diag_mask
from Plastic_Induction import plastic_search
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
        figures_output = "Experiment3"
    )
 
    COLUMN_TITLES = (
        "Magnitude\n(Y)",
        "Energy\ngate",
        "Acceptance\nrate",
        "AUC",
        "F\nchange",
        "Accepted\navg. dE",
        "Accepted\nstd. dE",
        "Accepted\navg. w",
        "Accepted\nstd. w",
        "Align\nchange",
        "Accepted\navg. align",
        "Accepted\nstd. align",
    )
 
    # Generate the starting profile (Constant used for all interaction matricies)
    rng = make_rng(base.seed)
    G = rng.uniform(low = -1, high = 1, size = base.N)
    
 
 
 
    """Test the Plastic search for a range of different initialised interaction matricies"""
 
    # Replicable functionality for running the experiment
    def _experiment3(base):
 
        # Save the search data
        search_data = []
 
 
        # Run the Plastic search for a sweep of magnitudes and energy gate protocols
        for Y in [0.5, 1.0, 2.0, 6.0]:
            BY = B * Y
            
            # Develop the Phenotype as the base for the plastic search
            P = handle_develop(G, BY, base, induction = False)
 
            cfg = replace(base, Y = Y)
 
 
            for gate in ENERGY_GATES:
 
                # Generates three new rows in the table for each magntiude Y
                row_data = []
 
                # Reset the Energy gate
                cfg = replace(cfg, energy_gate = gate)
 
                # Perform the plastic search - This search doesn't alter B
                # Fresh stream per gate, so the gates are compared on identical draws.
                history = plastic_search(B = BY, P = P, cfg = cfg, rng = make_rng(cfg.seed), limit_return = False)
 
 
                # Save the search data for the current row
                row_data.append(Y)
                row_data.append(gate)
                row_data.append(f"{history['acceptance_rate'] * 100:.1f}%") # Convert to a percentage in a string
                row_data.append(history["auc"])
                row_data.append(history["F_change"])
                row_data.append(history["avg_accept_E"])
                row_data.append(history["std_accept_E"])
                row_data.append(history["avg_accept_w"])
                row_data.append(history["std_accept_w"])
                row_data.append(history["align_change"])
                row_data.append(history["avg_accept_A"])
                row_data.append(history["std_accept_A"])
 
 
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
    #
    # The builder controls how well B already encodes the target, i.e. Tenet 1.
    # Appropriate interactions make development converge onto the target, so the
    # search starts at F = 1 and has no headroom; the noisy and random builders
    # leave room for the gates to differ.
 
    # Each builder returns (B, cfg).  Most keep the default dense mask, but the
    # sparse case must change the CONFIG as well, because the mask governs which
    # entries the search, the energy and the alignment measure all look at.
 
    def _dense(build):
        """Builder that keeps the default (dense) mask."""
 
        def wrapped(cfg, rng):

            if cfg.self_interaction:
                cfg = with_mask(cfg, diag_mask(cfg.N))

 
            return build(cfg, rng), cfg
 
 
        return wrapped
 
 
 
 
    def _sparse(build):
        """
        Builder on a SPARSE topology.  sparse_topology returns a mask, not a
        matrix, and the mask governs which entries the search, the energy and
        the alignment measure all look at -- so the CONFIG has to change too,
        not just B.
        """
 
        def wrapped(cfg, rng):
            mask = sparse_topology(cfg, rng)
            sparse_cfg = with_mask(cfg, mask)
 
 
            return build(sparse_cfg, rng), sparse_cfg
 
 
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
        ("noisy appropriate interactions",
            _dense(lambda cfg, rng: noisy_appropriate_interactions(cfg = cfg, rng = rng, S = cfg.target,
                                                                   inappropriate = False,
                                                                   normalise = cfg.interactions_norm))),
        ("noisy inappropriate interactions",
            _dense(lambda cfg, rng: noisy_appropriate_interactions(cfg = cfg, rng = rng, S = cfg.target,
                                                                   inappropriate = True,
                                                                   normalise = cfg.interactions_norm))),
        ("random interactions",
            _dense(lambda cfg, rng: random_interactions(cfg = cfg, rng = rng,
                                                        normalise = cfg.interactions_norm))),
 
        ("modular random interactions",
            _dense(lambda cfg, rng: modular_interactions(cfg = cfg, rng = rng,
                                                         normalise = cfg.interactions_norm))),
        ("modular appropriate interactions",
            _dense(lambda cfg, rng: modular_appropriate_interactions(cfg = cfg, rng = rng, S = cfg.target,
                                                                     inappropriate = False,
                                                                     normalise = cfg.interactions_norm))),
        ("modular inappropriate interactions",
            _dense(lambda cfg, rng: modular_appropriate_interactions(cfg = cfg, rng = rng, S = cfg.target,
                                                                     inappropriate = True,
                                                                     normalise = cfg.interactions_norm))),
 
        ("sparse random interactions",
            _sparse(lambda cfg, rng: random_interactions(cfg = cfg, rng = rng,
                                                         normalise = cfg.interactions_norm))),
        ("sparse appropriate interactions",
            _sparse(lambda cfg, rng: appropriate_interactions(cfg = cfg, rng = rng, S = cfg.target,
                                                              inappropriate = False,
                                                              normalise = cfg.interactions_norm))),
        ("sparse inappropriate interactions",
            _sparse(lambda cfg, rng: appropriate_interactions(cfg = cfg, rng = rng, S = cfg.target,
                                                              inappropriate = True,
                                                              normalise = cfg.interactions_norm))),
    )
 
    os.makedirs(base.figures_output or ".", exist_ok = True)
 
 
    for interaction_type, build in BUILDERS:
 
        # Same seed for every builder so the comparison is like for like.
        B, run_cfg = build(base, make_rng(base.seed))
 
        OUTPUT = os.path.join(base.figures_output or ".",
                              f"plastic_search_table_{interaction_type.replace(' ', '_')}.png")
 
        print(f"running: {interaction_type}")
        _experiment3(run_cfg)
        print(f"  wrote {OUTPUT}")
 