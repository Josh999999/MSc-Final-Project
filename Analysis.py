"""External Imports (Libraries and APIs)"""
import numpy as np


"""Local Imports"""
from Config import Config
from Plotting import create_search_table




def check_convergence(F: float, cfg: Config, l: float) -> bool:
    """
    True once F is within a tolerance `l` of the optimum.

    `l` is a TOLERANCE, so the bound is (1 - l) * optimal_fitness: with l = 0.2 and an optimum of 1.0 the run counts as converged at F >= 0.8. 

    Only a lower bound is applied: exceeding the nominal optimum (possible when normalise_fitness is off) still counts as converged.
    """
    optimal_F_lb = (1.0 - l) * cfg.optimal_fitness


    return bool(F >= optimal_F_lb)




def average_results(results: list) -> dict:
    """
    Average a list of result dicts (one per seed) key by key.

    Scalars average to a scalar; equal-length series average element-wise into a series of the same length.  
    """
    out = {"n_seeds": len(results)}


    for key in results[0]:
        vals = [r[key] for r in results if key in r]

        try:
            arr = np.asarray(vals, dtype = float)

        except (TypeError, ValueError):
            continue                        # strings, dicts, ragged series


        if arr.ndim == 0 or arr.size == 0:
            continue


        out[key] = arr.mean(axis = 0)
        out[key + "_std"] = arr.std(axis = 0)


    return out




def compare_arms(a: dict, b: dict, name_a: str = "induction", name_b: str = "control") -> dict:
    """
    Compare two averaged result dicts key by key.
    """
    rows = {}


    for key in a:

        if key == "n_seeds" or key.endswith("_std") or key not in b:
            continue


        va, vb = np.asarray(a[key], dtype = float), np.asarray(b[key], dtype = float)


        if va.shape != vb.shape or va.ndim > 1:
            continue                        # matrices / trajectory grids are not comparable as one number


        if va.ndim == 0:
            rows[key] = {name_a: float(va), name_b: float(vb), "diff": float(va - vb)}

        else:
            rows[key] = {name_a: float(va.mean()), name_b: float(vb.mean()),
                         "diff": float((va - vb).mean()),
                         "diff_final": float(va[-1] - vb[-1])}


    return rows




def comparison_table(rows: dict, save_loc: str, title: str, sig: int = 6, name_a: str = "induction", name_b: str = "control",) -> str:
    """Render compare_arms output as an aligned text table."""
    columns = [f"{'metric':<22}", f"{name_a:>14}", f"{name_b:>14}", f"{'diff':>12}", f"{'diff (final)':>14}"]


    rows = []
    for key, r in rows.items():
        fin = f"{r['diff_final']:>+14.4f}" if "diff_final" in r else f"{'':>14}"
        rows.append(f"{key:<22}", f"{r[name_a]:>14.4f}", f"{r[name_b]:>14.4f}", f"{r['diff']:>+12.4f}{fin}")


    create_search_table(
        column_tites = columns,
        row_results = rows,
        save_loc = save_loc,
        title = title,
        sig = sig
    )




def arm_summary(per_seed: list, keys: tuple) -> dict:
    """
    Per-arm summary across seeds: for each scalar key, the value from every eed plus mean and standard deviation.  
    
    This is the table to quote for a single arm, before comparing arms.
    """
    out = {}


    for k in keys:
        vals = np.array([float(r[k]) for r in per_seed if k in r], dtype = float)


        if vals.size:
            out[k] = {"per_seed": vals, "mean": float(vals.mean()), "std": float(vals.std())}


    return out




def arm_summary_table(summary: dict, arm: str, n_seeds: int, save_loc: str, title: str, sig: int = 6) -> str:
    """Render arm_summary output as an aligned text table."""
    seed_cols = [f"{f'seed {i}':>11}" for i in range(n_seeds)]
    columns = [f"{'metric':<22}", *seed_cols, f"{'mean':>11}", f"{'std':>10}"]

 
    rows = []
    for key, r in summary.items():
        rows.append([key, *r["per_seed"], r['mean'], r['std']])
 
 
    create_search_table(
        column_tites = columns,
        row_results = rows,
        save_loc = save_loc,
        title = f"{title} ({arm} arm, {n_seeds} seeds)",
        sig = sig
    )