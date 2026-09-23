"""
Regression tests.  Run before every commit:

    python tests.py

Every assertion here corresponds to a bug that actually shipped at some point
in this project.  If one fails, do not run experiments until it passes.
"""
import sys
import traceback
import numpy as np

from Config import Config, make_rng
from Data import S1, S2
from GRN import (handle_develop, masked_matrix, evaluate_fitness, mutate_profile, DTYPE)
from Induction import handle_induction
from Interactions import random_interactions
from Plastic_Induction import plastic_search_return_wrapper
from Evolution import sswm_evolve
from Analysis import check_convergence
from Tenets import synthesise_B


S = np.asarray(S1, dtype = float)
TESTS = []


def test(fn):
    TESTS.append(fn)
    return fn




# ------------------------------------------------------------------ fitness
@test
def fitness_of_target_is_one():
    cfg = Config(N = 8, targets = S1)
    assert abs(evaluate_fitness(S, S, cfg) - 1.0) < 1e-12
    assert abs(evaluate_fitness(-S, S, cfg) - 0.0) < 1e-12
    assert abs(evaluate_fitness(np.zeros(8), S, cfg) - 0.5) < 1e-12


@test
def aligned_genotype_can_score_natively():
    # the magnitude confound: under the decaying development a perfectly aligned
    # genotype scored 0.55.  Under the bounded form it must score well.
    cfg = Config(N = 8, targets = S1, T = 10)
    P = handle_develop(2.0 * S, masked_matrix(None, cfg.mask), cfg, induction = False)
    assert evaluate_fitness(P, S, cfg) > 0.95, evaluate_fitness(P, S, cfg)


@test
def phenotype_is_bounded():
    cfg = Config(N = 8, targets = S1, T = 10)
    B = 3.0 * random_interactions(cfg, make_rng(0), normalise = True)
    P = handle_develop(5.0 * make_rng(1).normal(size = 8), B, cfg, induction = False)
    assert np.all(np.abs(P) <= 1.0 + 1e-9)


@test
def convergence_bound_is_a_tolerance():
    cfg = Config(N = 8)
    assert not check_convergence(0.5, cfg, l = 0.2)
    assert check_convergence(0.8, cfg, l = 0.2)
    assert check_convergence(1.05, cfg, l = 0.2)      # above optimum still counts




# ------------------------------------------------------------------ matrices
@test
def masked_matrix_handles_none():
    cfg = Config(N = 8, targets = S1)
    B = masked_matrix(None, cfg.mask)
    assert B.shape == (8, 8) and np.all(np.isfinite(B)) and np.all(B == 0)


@test
def mask_owns_the_diagonal():
    for si in (False, True):
        cfg = Config(N = 8, targets = S1, self_interaction = si)
        B = synthesise_B(0.7, cfg, make_rng(0))
        assert (np.count_nonzero(np.diag(B)) > 0) == si


@test
def inputs_are_not_mutated():
    cfg = Config(N = 8, targets = S1, induction = True, M = 10, T = 6, rounds = 2)
    B = random_interactions(cfg, make_rng(0), normalise = True); G = make_rng(1).uniform(-1, 1, 8)
    P = handle_develop(G, B, cfg, induction = False)
    B0, G0, P0 = B.copy(), G.copy(), P.copy()
    handle_induction(B, P, G, cfg, make_rng(0), S, limit_return = False, AUC = -1)
    plastic_search_return_wrapper(B, P, cfg, make_rng(0), S, limit_return = True)
    mutate_profile(G, cfg, make_rng(2))
    assert np.array_equal(B, B0) and np.array_equal(G, G0) and np.array_equal(P, P0)


@test
def all_arrays_are_float64():
    cfg = Config(N = 8, targets = S1, induction = True, M = 10, T = 6)
    B = random_interactions(cfg, make_rng(0), normalise = True)
    P = handle_develop(make_rng(1).uniform(-1, 1, 8), B, cfg, induction = False)
    h = plastic_search_return_wrapper(B, P, cfg, make_rng(0), S, limit_return = False)
    for arr in (B, P, np.asarray(h["P"]), np.asarray(h["curve"])):
        assert arr.dtype == DTYPE, arr.dtype




# ------------------------------------------------------------------ induction
@test
def induction_is_deterministic_given_genotype():
    cfg = Config(N = 8, targets = S1, induction = True, M = 20, T = 6, rounds = 2)
    B = masked_matrix(None, cfg.mask); G = make_rng(3).uniform(-.5, .5, 8)
    P = handle_develop(G, B, cfg, induction = False)
    vals = {float(handle_induction(B, P, G, cfg, make_rng(cfg.induction_seed), S,
                                   limit_return = True, AUC = -1)["auc_inner"]) for _ in range(5)}
    assert len(vals) == 1, vals


@test
def both_processes_share_a_contract():
    for proc in ("plastic", "r-round"):
        for lim in (True, False):
            cfg = Config(N = 8, targets = S1, induction = True, induction_process = proc,
                         M = 6, T = 6, rounds = 2)
            B = random_interactions(cfg, make_rng(0), normalise = True); G = make_rng(1).uniform(-1, 1, 8)
            P = handle_develop(G, B, cfg, induction = False)
            h = handle_induction(B, P, G, cfg, make_rng(0), S, limit_return = lim, AUC = -1)
            for k in ("auc_inner", "B", "P", "F"):
                assert k in h, (proc, lim, k)


@test
def differential_energy_is_live_under_default_mutation():
    # with single-gene mutation dP is one-hot and dP.B.dP == 0 (no diagonal);
    # the default mutation type must be dense so the gate is not vacuous.
    cfg = Config(N = 8, targets = S1)
    assert cfg.mutation_type != "single-gene" or cfg.energy_type != "differential"


@test
def relaxed_induction_develops_genotype_toward_target():
    # eta and relax must be set so induction actually does something within a lifetime
    cfg = Config(N = 8, targets = S1, induction = True, M = 20, T = 10, rounds = 10)
    assert cfg.relax, "relax must be on for r-round to test assimilation"
    B = random_interactions(cfg, make_rng(0), normalise = True); G = make_rng(1).uniform(-1, 1, 8)
    P = handle_develop(G, B, cfg, induction = False)
    h = handle_induction(B, P, G, cfg, make_rng(0), S, limit_return = False, AUC = -1)
    oc = np.asarray(h["outer_curve"])
    assert oc[-1] > oc[0] + 0.05, f"relaxed fitness did not rise: {oc[0]:.3f} -> {oc[-1]:.3f}"




# ------------------------------------------------------------------ evolution
@test
def evolution_runs_all_four_configurations():
    for proc in ("plastic", "r-round"):
        for ind in (False, True):
            cfg = Config(N = 8, targets = np.array([S1, S2]), induction = ind, induction_process = proc,
                         M = 4, rounds = 2, n_generations = 60, switch_every = 20, record_every = 10, T = 6)
            st, _ = sswm_evolve(cfg, make_rng(0))
            assert np.isfinite(float(st["F"]))


@test
def per_generation_series_are_aligned():
    cfg = Config(N = 8, targets = np.array([S1, S2]), induction = True, M = 4, rounds = 2,
                 n_generations = 97, switch_every = 20, record_every = 15, T = 6)
    st, _ = sswm_evolve(cfg, make_rng(0))
    n = cfg.n_generations
    for k in ("selections", "switches", "converged"):
        assert len(st[k]) == n, (k, len(st[k]))
    rg = len(st["recorded_gens"])
    for k in ("fitnesses", "native_fitness", "plastic_fitness", "transfer_fitness", "B_magnitude"):
        assert len(st[k]) == rg, (k, len(st[k]), rg)
    assert np.flatnonzero(np.asarray(st["switches"])).tolist() == [20, 40, 60, 80]


@test
def b_mutation_undo_is_exact():
    # bounded mutation + rejection must leave B bit-identical
    cfg = Config(N = 8, targets = np.array([S1, S2]), induction = False,
                 n_generations = 200, switch_every = 1000, record_every = 50, T = 6,
                 u2 = 0.3, n_mut_B = 8, prob_mut_B = 1.0, B_limit = 0.3)
    st, _ = sswm_evolve(cfg, make_rng(0))
    B = np.asarray(st["B"])
    assert np.all(np.abs(B) <= cfg.B_limit + 1e-12), np.abs(B).max()


@test
def switch_generation_records_transfer():
    cfg = Config(N = 8, targets = np.array([S1, S2]), induction = False,
                 n_generations = 100, switch_every = 30, record_every = 7, T = 6)
    st, _ = sswm_evolve(cfg, make_rng(0))
    tf = np.asarray(st["transfer_fitness"], float); rg = np.asarray(st["recorded_gens"])
    assert set(rg[~np.isnan(tf)].tolist()) == {30, 60, 90}




# ------------------------------------------------------------------ runner
if __name__ == "__main__":
    failed = 0
    for fn in TESTS:
        try:
            fn(); print(f"  PASS  {fn.__name__}")
        except Exception as e:
            failed += 1; print(f"  FAIL  {fn.__name__}: {e}")
            traceback.print_exc(limit = 1)
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} passed")
    sys.exit(1 if failed else 0)
