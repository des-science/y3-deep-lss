# bench_v10 — the CROSS-PROBE CONSOLIDATION round

**Status (2026-08-17): configs written, nothing launched.**

**Goal.** Pick **one** DeepSphere architecture that is the default for lensing, clustering *and*
combined — robust, and no more complicated than the evidence requires. Every previous round ran on
combined only, so no lever in the whole lever table has ever been verified on a single probe.

This is the master README for the round. The per-probe directories
(`../../lensing/bench_v10/`, `../../clustering/bench_v10/`) hold the arms and only the
probe-specific notes.

## What bench_v7 → bench_v9 settled, and why that determines this round

Paired FoM from `deep_lss.apps.tuning.run_comparison`, final checkpoints, v18/default, reference
`bench_v7_full` on each probe. Seed floor 1.5%.

| run | combined | lensing | clustering |
|---|---|---|---|
| `bench_v7_full` (GCNN, `mean`) | 1.000 | 1.000 | 1.000 |
| `bench_v7_simple` | 0.961 | 0.974 | 0.922 |
| `bench_v7_transformer` | **1.123** | 0.989 = | 0.996 = |
| `bench_v8_mean_std` | **1.053** | — | — |
| `bench_v8_long` (2× budget) | 1.023 | — | — |
| `bench_v8_k20` | 1.020 | — | — |
| `bench_v9_unet_multiscale` | 1.031 | — | — |
| `bench_v8_unet_k8` | 0.933 | — | — |
| `bench_v9_moments` | 0.928 | — | — |
| `bench_v9_unet_k20` | 0.870 | — | — |
| `bench_v8_fuse_act` / `poolsplit` / `injection_conv` | 0.905 / 0.887 / 0.836 | — | — |

Four conclusions, each of which removes something from this round:

1. **The transformer's lead is combined-only.** It is a dead heat on both single probes (0.989,
   0.996) and wins 12.3% on combined — and pays for it with 2.3× the source-clustering bias of any
   GCNN (1.65σ vs 0.72σ). So "one architecture for everything" means the GCNN, and the only real gap
   is cross-probe fusion.
2. **The trunk is finished.** Six arms across two rounds added machinery to it — injection convs,
   a split pool, a seam nonlinearity, two U-net schedules — and **every one lost**. `bench_v7/full`
   (base 64, pool 3 / conv 2, ConvNeXt + DropPath 0.1 + attention every 2, trunk 512 @ nside 16)
   is the architecture. No trunk arm in this round.
3. **The readout is the only lever that ever paid, and it has a peak, not a slope.**
   `flatten → mean` +32.1%, `mean → mean_std` +5.3%, `mean_std → moments` **−12.2%**. So the round
   carries `mean_std` and nothing further up the ladder.
4. **The budget is nearly saturated on combined.** `long` doubled the wall for +2.3%, barely over the
   seed floor. That is the prior going into the 1×/2× question below.

## The arms

Three architectures. Every one is `bench_v7/full.yaml` plus declared knobs; nothing new is invented
here.

| arm | knobs vs `bench_v7/full.yaml` | combined | lensing | clustering |
|---|---|---|---|---|
| **A** `mean_std` | `map_pool` mean → mean_std | `bench_v8_mean_std` **(already run, 2×)** | 1× + 2× | 1× + 2× |
| **B** `mean_std_k20` | A, **+** `n_neighbors` 60 → 20 | new, 2× | 1× + 2× | 1× + 2× |
| **C** `unet_multiscale` | U-net schedule + k20 + `map_pool_multiscale` | `bench_v9_unet_multiscale` **(already run, 2×)** | 1× only | 1× only |

**16 new jobs**: combined 2, lensing 7, clustering 7. A-on-combined and C-on-combined already exist
and are not re-run.

### A — `mean_std`: the champion, transplanted

The programme champion at 1.053, and the *simplest* thing that wins: one extra reduction over the
pixel axis, no new depth, no spatial operator, negligible FLOPs. Nothing about it is
combined-specific — it reduces the ~448 nside-16 footprint tokens of the 512-channel trunk, and the
trunk is pinned to 512 on all three probes. This arm is the round's null hypothesis: if it transfers,
it is the answer.

### B — `mean_std_k20`: the same recipe on a sparser graph

The only *other* knob in the programme that has never lost, and it is a **simplification**: 5.03 it/s
against 3.32 on combined, a **1.51× cheaper step** for the same result. Note the honest reading of
its +2.0% — `bench_v8_k20` reached 406 500 steps against the anchor's 250 000 (1.63×) and
`bench_v8_long` measured +2.3% for a straight 2× budget, so **k20's entire gain is consistent with
the extra steps alone, with the sparser kernel neutral**. That is what this arm tests at the champion
readout. Given the cost saving, a *wash* against A is a win for k=20.

k=20 is also `deep_lss`'s own model-class default (`base_model.py:53`), silently overridden to 60 by
every config in this repo.

### C — `unet_multiscale`: settle the largest unexplained effect, cheaply

Largest single-knob gain since flatten → mean (0.870 → 1.031, +18.5% within its lineage) — but it
**never beat A in absolute terms** on the same mocks and wall. The deflationary reading fits every
number: the U-net trunk is half-width at every level except the last, and multi-scale simply returns
the 1984 channels the schedule threw away. A width patch, not a mechanism.

Single probes discriminate in a way combined cannot: **there is no injection seam**, so the taps are
one probe on one resolution ladder — the first clean test of "moments of a convolved field vs scale",
which is the starlet/scattering family this readout is the learned analogue of. Lensing (most
non-Gaussian, five stages) and clustering (57′-smoothed, four stages) predict *opposite* orderings
under the two readings.

**1× only, deliberately.** It is a diagnostic, not a candidate for the default: it is by far the most
complicated arm and it has already failed to beat A once. One job per probe settles it.

## The 1× vs 2× axis

`_1x` and `_2x` are **byte-identical except for `wall_budget_seconds`** (39 600 vs 79 200) — verified
by diff, and `job_budget_seconds` is 39 600 in both, so 2× is exactly two jobs of the 1×.

They must be separate runs, not one chain read at two checkpoints. Under `n_steps: auto` the cosine
spans the **whole** budget, so a 2× chain's mid-chain checkpoint is at cosine-midpoint with a live
learning rate — it is not a 1× result and must never be quoted as one. The gain lives in the anneal
tail.

Prior: `bench_v8_long` says 2× is worth **+2.3%** on combined, barely over the 1.5% seed floor. If
that holds on the single probes, the single-probe convention stays at one job and this round has
halved the cost of every future single-probe experiment. If it does *not* hold — plausible, since the
single probes have only ever run ~130 k steps against combined's 260 k — then the bench_v7
single-probe numbers were budget-limited and every cross-probe conclusion drawn from them needs
re-reading.

There is a free cross-check inside the matrix: **B@1× ≈ A@2× in step count** (1.51× rate against 2×
wall), at half the compute. If they land together, steps are the binding constraint; if B@1× wins,
the wide graph was costing quality as well as time.

## Scoring

**Rank with the `compare-runs` skill, per probe, against that probe's `bench_v7_full`.** Never across
probes — the FoM scale differs. Never by eye.

**Robustness (Q2) is a separate observable and is what "robust" in the goal means.** Posterior bias on
the contamination mocks, in σ of that run's own fiducial posterior. Gate on
`source_clustering_{gatti,in_place}`, **not** on the worst-overall number: `dmo` biases every
architecture by +0.58 to +0.97σ regardless of compression (a baryon-marginalization property) and
always wins a max, while source clustering spreads 0.0 → 1.66σ and is the axis that discriminates.

**Lensing is the probe whose Q2 decides the default**, since source clustering acts there. The
pattern across bench_v8/v9 is that `sc_gatti` S8 bias rises monotonically with FoM ratio
(−0.26 → +0.73σ): more informative compressions are more exposed. An arm that wins Q1 and loses Q2 is
not the answer.

DES FoM is unsigned and ranks nothing.

## The confound this round inherits, unchanged

The Cls embedding is pinned at 512 while the map readout widens: A is 1024 (2:1), C is 1984 on
lensing / 1920 on clustering (~3.8:1). Across bench_v8/v9 the FoM falls **monotonically** as that
ratio grows — 2:1 → 1.053, 3.9:1 → 1.031, 4:1 → 0.928 — so for C a **loss is uninterpretable**
(balance or readout) and only a **win** is unambiguous. A and B share the same 2:1 split, so the A-vs-B
contrast is clean.

Not corrected here: pinning the width with `map_feature_dim` inserts the projection the pooled-readout
lineage exists to avoid (the 59 M-param flatten projection is what its removal bought). The control
that separates them is `mean_wide`, still uncut and still deferred.

## Pre-launch checks

1. **Each arm built its own knob.** Diff the `ResNetMapsPlusCLSNetwork:` line across arms. C must
   print `map_pool_multiscale=True (5 scale taps)` on **lensing** and `(4 scale taps)` on
   **clustering** — the counts differ because the probes enter at different nside.
2. **The budget engaged.** `Wall-clock training budget: <total> s total, <spent> s already spent…` —
   `already spent` must be 0 on a 1× run and on job 1 of a 2× chain, and **non-zero** on job 2.
3. **Fresh lineages everywhere.** Every arm changes the readout width, and B and C also change
   `n_neighbors`, which alters the Laplacian with **no weight-shape change** — a k=60 checkpoint
   loads into a k=20 network with no error and meaningless filters, and `expect_partial()` will not
   raise. `RUN_NUM=1`, own `MODEL_DIR`, every time.
4. **Commit the tree before launch.** Follower jobs re-import the working tree hours later; an
   uncommitted readout change between jobs silently changes the architecture mid-chain.

## Launching

Via the `submit` skill — do not write a new submission script. `MAX_RUNS=1` for the 1× arms,
`MAX_RUNS=2` for the 2× ones; `MODEL_DIR` mirrors the config basename with a `bench_v10_` prefix.
