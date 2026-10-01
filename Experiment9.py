"""External Imports (Libraries and APIs)"""
import os
import numpy as np
 
 
"""Local Imports"""
from Config import Config, make_rng, set_mask
from Evolution import sswm_evolve
from GRN import diag_mask, handle_develop, evaluate_fitness, masked_matrix, _cos, sparse_topology
from Evolution_Experiment import modular_environment
from Plotting import create_search_table, _fmt
 
 
 
 
FIGURES_OUTPUT = "Experiment9"
 
N_GENERATIONS = 30_000
SWITCH_EVERY  = 1_500
RECORD_EVERY  = 200
SEEDS         = 10
 
# The environment every configuration is tested on.
ENV_N, ENV_K, ENV_SEED = 8, 4, 5
 
 
 
 
# !--- The Grid ---!
# Each entry: a name, and the Config overrides for the INDUCTION arm.  The
# control arm is the same Config with induction = False (plus the same
# topology overrides, so sparse rows compare like with like).  Rows mirror
# FINDINGS.md section 3.
GRID = [
    ("baseline",               {}),
 
    # induction process
    ("process: plastic",       {"induction_process": "plastic"}),
    ("process: r-round",       {"induction_process": "r-round"}),
 
    # energy gate
    ("gate: or",               {"energy_gate": "or"}),
    ("gate: and",              {"energy_gate": "and"}),
    ("gate: deterministic",    {"energy_gate": "deterministic"}),
 
    # energy type
    ("energy: standard",       {"energy_type": "standard"}),
    ("energy: differential",   {"energy_type": "differential"}),
 
    # plastic mutation operator
    ("mutation: single-gene",  {"mutation_type": "single-gene", "energy_type": "standard"}),
    ("mutation: phenotype",    {"mutation_type": "phenotype"}),
 
    # selection score
    ("score: auc",             {"selection_score": "auc"}),
    ("score: native",          {"selection_score": "native"}),
    ("score: native_bonus",    {"selection_score": "native_bonus"}),
    ("score: gated",           {"selection_score": "gated"}),
 
    # inheritance
    ("inherit: none",          {"inherit_induced": "none"}),
    ("inherit: all",           {"inherit_induced": "all"}),
    ("inherit: exclusive",     {"inherit_induced": "exclusive", "induction_interactions": "exclusive", "_sparse": 4}),
 
    # development form
    ("development: decay",     {"development": "decay"}),
    ("development: bounded",   {"development": "bounded"}),
 
    # relaxation target
    ("relax: genotype",        {"relax": True}),
    ("relax: phenotype",       {"relax": False}),
 
    # topology
    ("topology: sparse K=4",   {"_sparse": 4}),
 
    # B mutation rate
    ("B rate: prob 0.2",       {"prob_mut_B": 0.2}),
    ("B rate: prob 1.0",       {"prob_mut_B": 1.0}),
]
 
 
 
 
def make_config(targets, induction, seed, overrides):
    ov = dict(overrides)
    sparse = ov.pop("_sparse", None)
 
    cfg = Config(
        N = targets.shape[1], 
        targets = targets, 
        induction = induction,
        n_generations = N_GENERATIONS, 
        switch_every = SWITCH_EVERY, 
        record_every = RECORD_EVERY,
        seed = 100 + seed, 
        normalise_interactions = False, 
        **ov
    )
 
 
    if sparse is None:
        set_mask(cfg, diag_mask(cfg))
 
    else:
        cfg.K = sparse
        cfg.mask = sparse_topology(cfg, make_rng(ENV_SEED + 1))
 
 
    return cfg
 
 
 
 
def run_diff(name, control_overrides, overrides, seed, targets, ideal) -> np.ndarray:
 
    def run_config(overrides, induction) -> np.ndarray:
        cfg = make_config(targets, induction, seed = seed, overrides = overrides)
 
        results, _ = sswm_evolve(cfg, make_rng(cfg.seed), np.zeros((cfg.N, cfg.N)))
 
        P = handle_develop(results["G"], results["B"], cfg, induction = False)
 
        fit = np.asarray(results["fitnesses"], float)
        tl = np.asarray(results["transfer_loss"], float); tl = tl[~np.isnan(tl)]
        B = np.asarray(results["B"])
 
 
        results = np.asarray([
            float(evaluate_fitness(P, cfg.targets[results["target_index"]], cfg)),
            float(fit.mean()),
            float(tl.mean()) if tl.size else float("nan"),
            float(_cos(B[cfg.mask], masked_matrix(ideal, cfg.mask)[cfg.mask])),
            float(np.linalg.norm(B)),
            float(np.mean(results["selections"])),
        ], dtype = np.float64)
 
 
        return results
 
 
    control_results = run_config(control_overrides, induction = False)
    induction_results = run_config(overrides, induction = True)
    diff = induction_results - control_results
 
 
    # list + ndarray is element-wise ADDITION, not concatenation
    return [name, seed, *diff]
 
 
 
METRICS = (
    "Native Fitness",
    "Mean Fitness",
    "Switch Loss",
    "Modularity",
    "Interaction Magnitude",
    "Acceptance Rate",
)
 
 
 
 
def summarise_seeds(seed_rows: list) -> list:
    """
    Collapse a configuration's per-seed rows into one row of "mean ± sd".
 
    The numeric columns start after the name and seed, and they arrive as
    np.float64, so the test has to be isinstance(..., np.floating) rather than
    type(...) == float.
    """
    values = np.asarray([row[2:] for row in seed_rows], dtype = np.float64)
    mean, std = np.nanmean(values, axis = 0), np.nanstd(values, axis = 0)
 
 
    return [seed_rows[0][0], f"mean of {len(seed_rows)}"] + \
           [f"{_fmt(m, 4)} \u00b1 {_fmt(s, 4)}" for m, s in zip(mean, std)]
 
 
 
 
def _experiment9(title: str, img_name: str, only_avgs: bool = False):
    targets, ideal = modular_environment(ENV_N, ENV_K, ENV_SEED)
    rows = []                                   # local: the two tables are independent
 
 
    for name, overrides in GRID:
        control_overrides = {k: v for k, v in overrides.items() if k == "_sparse"}
        seed_rows = []
 
 
        for seed in range(SEEDS):
            seed_rows.append(run_diff(name, control_overrides, overrides, seed, targets, ideal))
            print(f"        {name:<24} seed {seed}", flush = True)
 
 
        if not only_avgs:
            rows.extend(seed_rows)                     # extend, not append(a, b)
 
 
        rows.append(summarise_seeds(seed_rows))
 
 
    os.makedirs(FIGURES_OUTPUT, exist_ok = True)
    path = os.path.join(FIGURES_OUTPUT, img_name)
 
    create_search_table(
        column_tites = ["Configuration", "Seed"] + [f"{m}\n(Induction Improvement)" for m in METRICS],
        row_results = rows,
        save_loc = path,
        sig = 6,
        title = title
    )
 
    print(f"        wrote {path}")
 
 
 
 
if __name__ == "__main__":
 
    TITLE = (f"Effect of each induction configuration, paired against its control\n"
             f"(induction \u2212 control; {N_GENERATIONS} generations, switch every {SWITCH_EVERY}, "
             f"{SEEDS} seeds)")
 
    # every seed shown, with a mean +- sd row per configuration
    _experiment9(title = TITLE, img_name = "configurations_table.png", only_avgs = False)
 
    # one mean +- sd row per configuration
    _experiment9(title = TITLE, img_name = "configurations_avgs_table.png", only_avgs = True)
 