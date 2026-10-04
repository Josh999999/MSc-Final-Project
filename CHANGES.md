# Changes to the codebase — 4 October 2026 (revised after the design-space pruning)

Audit of the 23 files uploaded on 4 October against what we settled in the
dissertation work. Every change below was made to *your* files; the suite
passes 18/18 on them, Experiments 1, 2 and 5 were run in full here, and
Experiments 3, 4, 6, 7 and 8 were run end to end at reduced size to exercise
every code path (including the two that used to crash).

## Fixes that stop crashes

| File | Problem | Fix |
|---|---|---|
| `Evolution_Experiment.py` | `plot_run` used `B_heb` before defining it → `NameError` after the first seed of Experiments 6/7. The two heatmap filenames were also swapped. | `B_heb` defined first; Hebbian matrix saved as `hebbian_heatmap.png`, evolved B as `interaction_heatmap.png`. |
| `Analysis.py` | `comparison_table` rebound `rows = []` then called `.items()` on the list, and `append`ed four arguments → crash at the very end of Experiments 6/7. | Builds `table_rows` as lists of values for `create_search_table`. |
| `Plotting.py` | `random_profiles` referenced an undefined `Global` when `N` was `None`. | Dead branch removed. |

## Fixes from the dissertation work

| File | Change |
|---|---|
| `Plastic_Induction.py` | The starting phenotype is clipped into the walk's range `[-1, 1]^N` when `bound_phenotype` is on (the fix behind the normalised Table 5.2). Leftover `tau = None` removed. |
| `R_round_induction.py` | `relax_from` honoured: `"phenotype"` continues each round from P′, `"genotype"` redevelops G. New `assimilation_curve` / `assimilation_change` in the full return (fitness the genotype develops to under each round's matrix). `placticity` typo. |
| `Config.py` | `relax = True` (was `False`); new field `relax_from = "phenotype"`; **`rounds = 10` (was 50 — see below)**; `plasticy` comment fixed. |
| `Evolution.py` | On gated-off recorded generations the induction series carry the last induction evaluation forward instead of being padded with native fitness (which is on the development scale, up to 2.7, and made `AUC_inner` unreadable). Three comment typos. |
| `tests.py` | `relaxed_induction_develops_genotype_toward_target` → `induction_raises_the_plastic_auc_across_rounds` (asserts the inner AUC rises across rounds and stays ≤ 1; fails at η = 0.01 and under genotype relaxation). New `plastic_search_fitness_stays_on_the_walks_scale` (fails on the unclipped walk). 18 tests. |
| `Tenets.py` | `plastic_measure_surface` defaults `grid = 21`, `n_seeds = 8`. |
| `Experiment3.py`, `Experiment4.py` | `grid = 15` → `21`. |
| `Experiment5.py` | inner function renamed `_experiment5` (was `_experiment6`). |
| `Experiment8.py` | writes to `Experiment8/` (was `Experiment9/`); usage text. |
| `Experiment2.py` | two comment typos. |
| `Data.py`, `Induction.py`, `Experiment5.py` | CRLF line endings normalised to LF. |

## Design space pruned (your list, 13:54)

| Removed | What the code does now |
|---|---|
| `relax_from` | `relax` is a plain switch: on = develop the plastic phenotype P′ under the updated matrix before the next walk; off = the next walk starts from P′ itself. The `assimilation_curve` diagnostic (what G alone develops to under each round's matrix) is kept in the r-round return — it is a measurement, not a relaxation mode; say if you want it gone too. |
| "gated" scoring | Induction runs on **every** generation when `induction` is on; the challenger and the incumbent are therefore scored by the same judge and no re-scoring is needed. `selection_score` keeps `"auc"`, `"native"`, `"native_bonus"` (default, the formula gated used). The carry-forward padding is gone with it — every recorded generation has a real induction value. |
| `normalise_interactions` | Builders no longer take `normalise=`; the post-learning renormalisation in `R_round_induction` is gone; `Interactions.normalise_interactions` removed. (Superseded below: `_finalise` now always rescales to Frobenius norm `Y`.) `energy_normalise_interactions` (normalising B *inside the energy calculation*) is a different switch and is kept at its default `True`, because the gate thresholds in Table 3.2/5.2 depend on it. |
| `drift_selection` | Acceptance is fixed at `F_mut >= F` (ties fix), as in the published model and Table 5.1. |
| `baldwin_effect` | Removed; whether the induced matrix is inherited is governed by `inherit_induced` alone (`"none"` = scoring only). |
| `induction_interactions` | The contrastive update is applied to the whole matrix (the old `"all"` default). |
| Experiment 8 grid | Rows that depended on removed options are gone (`score: gated`, `inherit: exclusive`, `relax from: genotype`); `relax from: phenotype` is now `relax: on`. 22 configurations × 10 seeds = 220 tasks (then 170 after the scoring change below). The experiment itself is kept — §4.2 and §5.9 refer to the design grid; delete `Experiment8.py` and `job_experiment8*.sh` if you decide to drop it. |

Tests still 18/18 (the builders' `normalise=True` in the tests became an explicit `adjust_interaction_magnitude(..., Y = 1.0)`, which produces the identical matrices, so no threshold moved). Table 5.2 re-verified unchanged. `Config` now has **54** parameters (§4.2 updated).

## Selection score (your message, 15:00)

Under induction a genotype's fitness is now **the mean plastic AUC over the
rounds**, and nothing else:

- `selection_score`, `plastic_bonus` and `inherit_induced` are removed from
  `Config`. `Evolution.induction_history` returns `history["auc_inner"]`; the
  induced matrix is used for scoring only and is never written back into B.
- `B_limit` stays (it is mutation regularisation, not scoring) and has moved
  to the Evolution block of `Config`; `deterministic_induction`,
  `induction_seed` and `fitness_type` sit under a "Deterministic evaluation"
  heading.
- Because the two arms now select on different quantities (native fitness in
  the control arm, plastic AUC under induction), `Evolution` records the
  **incumbent's native fitness** on every recorded generation in both arms
  (`incumbent_native_fitness`), and the Experiment 6/7 `mean_fitness` row
  uses it, so the arms are compared on one scale. The fitness plot's labels
  now say which score is being plotted; the two induction plots had their
  inner/outer legend keys swapped — fixed.
- Experiment 8 grid: the score and inheritance rows are gone; 17
  configurations × 10 seeds = **170 tasks** (array script resized).
- `Config` now has **51** parameters (§4.2 updated).

Heads-up from the 200-generation smoke run (standard environment, 2 seeds):
with B = 0 at the start, the differential energy is identically zero, so the
deterministic gate accepts every fitness-improving move and the walk alone
reaches F ≈ 1 from any genotype. The induction arm's selection score sat at
0.986 from generation 0 while the incumbent's native fitness stayed at 0.50
(the control arm's reached 2.16). That is the "selection on the genotype is
weaker under induction" effect the abstract describes, now in its pure form:
until drift gives B enough structure for the gate to bite, the AUC carries no
information about (G, B). Worth knowing before you read Experiment 6/7.

## Interaction builders always normalise; no inheritance (your message, 15:16)

- `Interactions._finalise` now ends every builder with a rescaling to
  Frobenius norm `cfg.Y`. A builder fixes the sign pattern and relative
  weights; `Y` alone sets the magnitude, which is what `Y` already meant for
  the synthesised matrices of the tenet surfaces. Consequence: **Table 5.2 and
  §5.6 were regenerated** (the appropriate structures still start at or within
  0.03 of the target; the inappropriate ones now start at ≈ 0.5 because a unit-
  norm inappropriate matrix shrinks the phenotype; the four findings stand and
  finding 4 is sharper — the deterministic gate closes 0–30 % of the headroom
  on the inappropriate family against 70–87 % on the random family). Experiment
  5's round table will also change on re-run; Experiments 1, 3, 4, 6, 7, 8 are
  unaffected (they do not use the builders).
- Interaction matrices are never inherited: there is no switch; the induced
  matrix is used for the lifetime's score only (done in the previous round,
  confirmed here).

## `rounds`: 50 → 10

Your `Config.py` had `rounds = 50`. Every induction evaluation runs R walks of
M = 100 steps, so R sets the cost of everything downstream. Measured here:

| | rounds = 10 | rounds = 50 |
|---|---|---|
| evolution with induction, N = 8 | 0.11 s/generation | 0.50 s/generation |
| evolution with induction, N = 16 | 0.12 s/generation | 0.57 s/generation |
| Experiment 4 (21×21 grid, 8 seeds, 4 magnitudes) | ≈ 2 h | ≈ 10 h |
| Experiment 6 / 7 (5 seeds × 2 arms × 20k generations) | ≈ 3.5–4 h | ≈ 14–16 h |
| Experiment 8, one (configuration, seed) task, 30k generations | ≈ 1 h | ≈ 5 h |
| Experiment 8 total, 170 tasks at 20 concurrent | ≈ 9 h wall | ≈ 45 h wall |

Experiment 5's table shows the plastic AUC saturating within about five rounds
for every structure, and all the analysis in Chapter 5 used R = 10, so the
default is now 10. If you want 50 back it is one line, and the job scripts'
time limits already cover it except Experiment 8 (raise `--time` to 08:00:00
→ 10:00:00 there).

## Experiment 5 note

The "seeds improved" column counts `round 5 > round 1` strictly, so a seed
whose round-1 walk already reaches 1.000 (appropriate matrices) cannot count
as improved. That is why appropriate shows 3/5 while its AUC is 1.000. Either
count `>=` with a note, or report "reached the optimum" separately. The
regenerated table gives 47/55 improved overall (the abstract still says 31/55
from the old run).

## Job scripts (in this folder, numbered to match your experiments)

`job_experiment1.sh` … `job_experiment7.sh`, `job_experiment8.sh` (array of
170 tasks) + `job_experiment8_summarise.sh`, `submit_all.sh`, `run_local.sh`,
`_setup.sh`, `setup_env.sh`. Each: correct `--job-name`, `-c 1`, `--mem=2G`,
`python tests.py` gate, git commit logged, `mail-type=END,FAIL`.

    mkdir -p out
    bash submit_all.sh 1 2 5        # minutes
    bash submit_all.sh 3 4          # ~40 min and ~2 h
    bash submit_all.sh 6 7          # ~4 h each
    bash submit_all.sh 8            # array + chained summarise

## Numbers for the dissertation (already changed in the .tex sources)

- `Config` has **51** parameters (§4.2 said 60).
- Table 4.1 line counts regenerated from these files (non-blank lines).
- §4.1 experiment numbering: 1–5 static, 6–7 evolution, 8 design grid.
- §4.4 and Table A now agree: AUC of a fixed genotype varies with σ ≈ 0.022
  across fresh streams; one genotype mutation moves native fitness by ≈ 0.003;
  about seven to one (measured on the default configuration, 50 draws each).
- Table A: 18 rows, matching the suite; the appendix screenshot still needs
  re-taking (18/18) and "seventeen" → "eighteen".
