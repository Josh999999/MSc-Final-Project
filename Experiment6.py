"""External Imports (Libraries and APIs)"""
import os
import numpy as np
 
 
"""Local Imports"""
from Config import Config, make_rng, with_mask
from Data import S1
from Interactions import (appropriate_interactions, noisy_appropriate_interactions,
                          random_interactions, modular_interactions,
                          modular_appropriate_interactions)
from GRN import sparse_topology, diag_mask, handle_develop
from R_round_induction import r_round_induction_return_wrapper
from Plotting import plot_round_curves, plot_matrices, create_search_table, plot_counts
from Tenets import ideal_hebbian
 
 
 
 
FIGURES_OUTPUT = "Experiment6"
 
SEEDS  = 5      # independent runs per interaction type
ROUNDS = 5      # R: rounds of induction per run
 
 
 
 
if __name__ == "__main__":
 
    # Standard induction settings: the Config defaults, with R fixed at ROUNDS.
    cfg = Config(
        N = len(S1),
        targets = S1,
        induction = True,
        induction_process = "r-round",
        rounds = ROUNDS,
    )
 
 
 
 
    def _experiment6(cfg: Config, B: np.ndarray, interaction_type: str, n_seeds: int = SEEDS) -> list:
        curves, induced, keys = [], [], []
        round_aucs = []                       # AUC of each round's walk, per seed
 
 
        for seed in range(n_seeds):
            rng = make_rng(cfg.seed + seed)
 
            # A fresh genotype per seed; B is the same starting matrix for all of them, so the seeds differ only in genotype and plastic draws.
            G = rng.uniform(low = -1.0, high = 1.0, size = cfg.N)
            P = handle_develop(G, B, cfg, induction = False)
 
            # !-- Induction: R rounds of plasticity + contrastive update --!
            history = r_round_induction_return_wrapper(
                B = B.copy(), P = P, G = G, cfg = cfg, rng = rng,
                S = cfg.target, limit_return = False, AUC = -1
            )
 
            B_induced = np.asarray(history["B"], dtype = float)

 
            # !-- The plastic curve INSIDE each round --!
            curves.append([np.asarray(c, dtype = float) for c in history["round_curves"]])
            round_aucs.append([float(np.mean(c)) for c in history["round_curves"]])
            induced.append(B_induced)
            keys.append(f"seed {seed}")
 
            starts = [round(float(c[0]), 3) for c in history["round_curves"]]
 
 
        safe = interaction_type.replace(" ", "_")

 
        # !-- The plastic walk within each round, one panel per seed --!
        plot_round_curves(
            round_curves = curves,
            keys = keys,
            separate = "grid",
            saveloc = os.path.join(FIGURES_OUTPUT, f"plastic_curve_{safe}.png"),
            title = f"Plastic search within each of the {ROUNDS} rounds of induction",
            subtitle = interaction_type,
            x_label = "Plastic step within the round",
            y_label = "Fitness",
        )


        H = ideal_hebbian(cfg)

 
        # !-- The induced matrices, with the ideal and the starting matrix --!
        plot_matrices(
            matrices = [H, B] + induced,
            keys = ["ideal Hebbian S(x)S", "initial B"] + keys,
            saveloc = os.path.join(FIGURES_OUTPUT, f"induced_matrix_{safe}.png"),
            title = f"Interaction matrix produced by {ROUNDS} rounds of induction",
            subtitle = f"{interaction_type}, shared colour scale",
            colour_label = "interaction weight",
        )

 
        # !-- What induction changed: induced - initial, per seed --!
        plot_matrices(
            matrices = [B_ind - B for B_ind in induced],
            keys = keys,
            saveloc = os.path.join(FIGURES_OUTPUT, f"induced_change_{safe}.png"),
            title = f"Change made by {ROUNDS} rounds of induction (induced \u2212 initial)",
            subtitle = f"{interaction_type}, shared colour scale",
            colour_label = "change in interaction weight",
            reference = H,
            reference_key = "ideal Hebbian S(x)S",
        )
 
 
        # !-- The row for the cross-builder AUC summary --!
        aucs = np.asarray(round_aucs, dtype = float)       
        first, last = aucs[:, 0], aucs[:, -1]
        diff = last - first
        improved = int(np.sum(diff > 0))
 
 
        return [
            interaction_type,
            float(first.mean()),
            float(last.mean()),
            float(diff.mean()),
            float(diff.std()),
            f"{improved} / {n_seeds}"
        ], improved
 
 
 
 
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
 
 
    summary_rows = []
    improved_counts = []
 
 
    for interaction_type, build in BUILDERS:
 
        # Same seed for every builder so the comparison is like for like.
        B, run_cfg = build(cfg, make_rng(cfg.seed))
 
        print(f"running: {interaction_type}")
        row, improved = _experiment6(run_cfg, B, interaction_type)
        summary_rows.append(row)
        improved_counts.append((interaction_type, improved))
        print(f"  wrote plastic_curve_ and induced_matrix_ for {interaction_type}")
 
 
    # !-- A final row: the average of each column across every interaction type --!
    # The seed counts are summed rather than averaged, since "improved" is a
    # count of runs; everything else is a mean over the builders.
    numeric = np.asarray([row[1:5] for row in summary_rows], dtype = float)
    total_improved = sum(n for _, n in improved_counts)
 
    summary_rows.append([
        f"average of {len(summary_rows)} types",
        *[float(v) for v in numeric.mean(axis = 0)],
        f"{total_improved} / {len(improved_counts) * SEEDS}",
    ])
 
 
    # !-- Across every interaction type: did the rounds improve the walk? --!
    create_search_table(
        column_tites = ["Interaction type",
                        f"AUC round 1",
                        f"AUC round {ROUNDS}",
                        f"Mean difference\n(round {ROUNDS} \u2212 round 1)",
                        "Std",
                        f"Seeds improved\n(round {ROUNDS} > round 1)"],
        row_results = summary_rows,
        save_loc = os.path.join(FIGURES_OUTPUT, "round_auc_table.png"),
        title = f"Plastic AUC across {ROUNDS} rounds of induction, by starting interaction matrix "
                f"(mean \u00b1 sd over {SEEDS} seeds)",
        sig = 4,
    )
 
    print(f"  wrote {os.path.join(FIGURES_OUTPUT, 'round_auc_table.png')}")
 
 
    # !-- How many seeds improved, per interaction type --!
    plot_counts(
        labels = [name for name, _ in improved_counts],
        counts = [n for _, n in improved_counts],
        total = SEEDS,
        saveloc = os.path.join(FIGURES_OUTPUT, "round_auc_improved.png"),
        title = f"Seeds whose round {ROUNDS} plastic AUC exceeded round 1",
        subtitle = f"out of {SEEDS} seeds per interaction type",
        y_label = "seeds improved",
    )
 
    print(f"  wrote {os.path.join(FIGURES_OUTPUT, 'round_auc_improved.png')}")
 