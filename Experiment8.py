"""External Imports (Libraries and APIs)"""
import os
import sys
import json
import glob
import numpy as np
 
 
"""Local Imports"""
from Config import Config, make_rng, set_mask
from Evolution import sswm_evolve
from GRN import diag_mask, handle_develop, evaluate_fitness, masked_matrix, _cos, sparse_topology
from Evolution_Experiment import modular_environment
from Plotting import create_search_table, _fmt
 
 
 
 
FIGURES_OUTPUT = "Experiment8"
 
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
 
    # development form ("standard" is the published form; there is no "decay" option)
    ("development: standard",  {"development": "standard"}),
    ("development: bounded",   {"development": "bounded"}),
 
    # relaxation after each round, on or off
    ("relax: on",              {"relax": True}),
    ("relax: off",             {"relax": False}),
 
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
 
 
 
 
SCRATCH = os.path.join(FIGURES_OUTPUT, "scratch")
N_TASKS = len(GRID) * SEEDS
 
 
def task_to_row_seed(task: int) -> tuple:
    """Array task id -> (grid row, seed). Tasks are numbered row-major."""
    if not 0 <= task < N_TASKS:
        raise SystemExit(f"task {task} out of range 0..{N_TASKS - 1}")
 
    return divmod(task, SEEDS)
 
 
def scratch_path(row: int, seed: int) -> str:
 
    return os.path.join(SCRATCH, f"{row:02d}_{seed:02d}.json")
 
 
def run_task(task: int) -> None:
    """One (configuration, seed) pair: both arms, written to a scratch file."""
    row, seed = task_to_row_seed(task)
    name, overrides = GRID[row]
    control_overrides = {k: v for k, v in overrides.items() if k == "_sparse"}
    targets, ideal = modular_environment(ENV_N, ENV_K, ENV_SEED)
 
    result = run_diff(name, control_overrides, overrides, seed, targets, ideal)
 
    os.makedirs(SCRATCH, exist_ok = True)
    with open(scratch_path(row, seed), "w") as f:
        json.dump({"row": row, "seed": seed, "name": name, "diff": [float(v) for v in result[2:]]}, f)
 
    print(f"        {name:<24} seed {seed}  -> {scratch_path(row, seed)}", flush = True)
 
 
def summarise() -> None:
    """Build both tables from the scratch files. Nothing is re-run."""
    rows_all, rows_avg, missing = [], [], []
 
    for row, (name, _) in enumerate(GRID):
        seed_rows = []
 
        for seed in range(SEEDS):
            path = scratch_path(row, seed)
 
            if not os.path.exists(path):
                missing.append(f"{row}_{seed} ({name}, seed {seed})")
                continue
 
            with open(path) as f:
                rec = json.load(f)
 
            seed_rows.append([name, seed, *np.asarray(rec["diff"], dtype = np.float64)])
 
        if not seed_rows:
            continue
 
        rows_all.extend(seed_rows)
        rows_all.append(summarise_seeds(seed_rows))
        rows_avg.append(summarise_seeds(seed_rows))
 
    if missing:
        per_row = {}
        for m in missing:
            per_row.setdefault(m.split(" (")[1].rsplit(", seed", 1)[0], 0)
            per_row[m.split(" (")[1].rsplit(", seed", 1)[0]] += 1
 
        print(f"WARNING: {len(missing)} of {N_TASKS} tasks have no scratch file; those rows are averaged "
              f"over the seeds that finished. Missing seeds per configuration:", flush = True)
 
        for name, k in per_row.items():
            print(f"        {name:<24} {k}/{SEEDS}", flush = True)
 
    n_done = N_TASKS - len(missing)
    title = (f"Effect of each induction configuration, paired against its control\n"
             f"(induction \u2212 control; {N_GENERATIONS} generations, switch every {SWITCH_EVERY}, "
             f"{SEEDS} seeds, {n_done}/{N_TASKS} runs)")
    columns = ["Configuration", "Seed"] + [f"{m}\n(Induction Improvement)" for m in METRICS]
    os.makedirs(FIGURES_OUTPUT, exist_ok = True)
 
    for img_name, rows in (("configurations_table.png", rows_all), ("configurations_avgs_table.png", rows_avg)):
        path = os.path.join(FIGURES_OUTPUT, img_name)
        create_search_table(column_tites = columns, row_results = rows, save_loc = path, sig = 6, title = title)
        print(f"        wrote {path}", flush = True)
 
 
if __name__ == "__main__":
    args = sys.argv[1:]
 
    if args == ["--count"]:
        print(N_TASKS)
 
    elif args == ["--summarise"]:
        summarise()
 
    elif len(args) == 1 and args[0].lstrip("-").isdigit():
        run_task(int(args[0]))
 
    elif not args:
        for task in range(N_TASKS):
            run_task(task)
 
        summarise()
 
    else:
        raise SystemExit("usage: Experiment8.py [--count | --summarise | <task id>]")