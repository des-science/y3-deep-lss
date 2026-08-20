# bench_v6 — DeepSphere/GCNN combined-probe HPO (round 3), redesigned

**Date staged:** 2026-07-26 (configs rewritten; **not submitted** — standing hold: no full training run launches without explicit go-ahead)
**Configs:** `y3-deep-lss/configs/deepsphere/combined/bench_v6/*.yaml` (+ `_deferred/`)
**Submission driver:** `y3-deep-lss/submissions/clariden/training.sh`
**Predecessor:** `dev/notes/bench_v5_deepsphere_combined_2026-07-24.md`
**Re-analysis this round is built on:** `dev/notes/bench_v5_paired_reanalysis_2026-07-26.md` — **read that first**
**Data:** v17/baseline (standard-NLA, bta-free), scales `8wl,32gc`, probe `combined`, Cls head n_bins 16
**Target to beat:** transformer `t2_cls` (260 k steps, vali −10.817), paired ratio 1.000 by definition
**Best GCNN to date:** `bench_v5_pool_head_w64` — paired **0.918** vs `t2_cls`, vali −10.600

## Why this round was rewritten before it ran

An earlier draft of bench_v6 existed (six ConvNeXt configs, all step-matched at 210 k). Its premises came from
the 16-cosmology `chain_grid_*` median, which has a ~2.4 % floor. Re-running every v17 combined run through
`deep_lss/apps/tuning/run_comparison.py` on the 1000-mock paired route changed two things that matter:

1. **"2× budget bought ZERO signed gain" is sign-reversed.** `bench_v5_default` → `bench_v5_pool_head` is a
   pure budget test (config diff: `n_steps` 190 k → 380 k, nothing else). Paired FoM **+7.3 %** on **73 %** of
   1000 mocks, vali_total −10.086 → −10.653. The old draft's central design decision — "bench_v6 does NOT vary
   the budget" — was therefore built on an artifact, and the round as drafted declined its second-largest
   available lever.
2. **Every lever magnitude was inflated**, typically by ~50 %: width +15.6 → **+8.4 %**, DropPath +12.7 →
   **+8.5 %**, attention +7.9 → **+5.6 %**. Signs and rank order survived; the composition arithmetic did not.

A third finding reframes the round: **budget is conditional on the readout.** With the flatten readout more
steps actively *hurt* (`v3_cls`: paired 0.545 → 0.302 → 0.270 across 150 k/200 k/310 k; `bench_v5_w64` is the
worst run of bench_v5 at 0.680). With the mean-pool readout the sign flips positive. Every bench_v6 config
uses `map_pool: mean`, which is exactly why this round should be *spending* budget rather than freezing it.

## Design

**Wall-clock-matched, not step-matched** — reverting to the bench_v4/v5 house methodology. `n_steps` is sized
so the cosine anneals fully inside the job, because the FoM gain lives in the anneal tail. Step-matching would
also have made the ConvNeXt cost saving unspendable: pinned at 210 k, the ConvNeXt arms would idle ~40 % of
their wall, which is the entire justification for the block.

**Everything anchors on `bench_v5_pool_head_w64`** (base 64, mean-pool readout, classic body, 210 k, 0.918) —
already finished, no re-run needed. Each config is one knob from it or from the round anchor.

The stages are the folder structure: `stage1_levers/`, `stage2_budget/`, `_deferred/`. The submit glob is
**per stage** (`stage1_levers/*.yaml`) — the stages have different walls and different launch preconditions,
so they must never be globbed together. Round map: `configs/deepsphere/combined/bench_v6/README.md`.

### `stage1_levers/` — 4 × (2 × 12 h), one knob each

| config | one-knob change vs | question it answers | n_steps |
|---|---|---|---|
| **`droppath_classic.yaml`** | **the champion** | Does DropPath's +8.5 % transfer to the **classic** block? Inherits no other premise. | 210 k (step-matched) |
| `convnext.yaml` | the champion | Is ConvNeXt still quality-neutral at base 64? (cost lever + attention prerequisite) | ~350 k * |
| `convnext_droppath.yaml` | `convnext.yaml` | DropPath on the ConvNeXt body, as measured in bench_v5 | ~330 k * |
| `convnext_droppath_attn.yaml` | `convnext_droppath.yaml` | Does attention's +5.6 % transfer, and do the levers compose? | ~290 k * |

**`droppath_classic.yaml` is the lead config**, and it only exists because the DropPath ↔ ConvNeXt coupling
turned out to be an implementation accident rather than a design constraint (see *Decoupling* below). It is
one knob from a finished run, adds zero parameters, and predicts 0.918 × 1.085 ≈ **1.00**.

It is also the round's one **step-matched** config, deliberately. Wall-clock matching is the house rule and is
right when comparing *different geometries*, where equal steps silently favour the slower one. Here the
comparison is against an exact architectural twin that has already run at 210 k, so equal steps costs nothing
and buys a perfect one-knob contrast with no step correction. DropPath is ~4 % overhead, so it fits.

### `stage2_budget/` — 4 × 12 h each; run `budget_classic`, then pick

| config | one-knob change vs | when to run |
|---|---|---|
| `budget_classic.yaml` | the champion (`n_steps` 210 k → 420 k, **nothing else**) | **UNCONDITIONAL** — may launch alongside stage 1 |
| `droppath_classic_long.yaml` | `stage1_levers/droppath_classic.yaml` at 2× budget | if DropPath transfers to the classic block; ~1.07 predicted, **zero untested architecture** |
| `convnext_droppath_attn_long.yaml` | `stage1_levers/convnext_droppath_attn.yaml` at 2× budget | if the whole ConvNeXt line holds; ~1.13 predicted but conditional on three premises at once |

The last two are the round's two routes past the transformer, trading risk against ceiling. Their difference,
if both run, is itself the answer to "does the ConvNeXt line earn its complexity?"

`*` **PROVISIONAL — benchmark before launch.** ConvNeXt has never been run at base 64. Sizing is extrapolated
from *real* 4-GPU rates (classic b32 4.99, convnext b32 5.80, classic b64 2.95 it/s) via a fixed-overhead +
C²-scaling body model → convnext b64 ≈ 4.4 it/s. Synthetic `step_ms` over-predicted the ConvNeXt advantage by
~10 % in bench_v5, so do not size from it. Reset `n_steps = it/s × 79.2 ks` (floor 10 k) from a real job-1
rate. The three **classic-body** configs need no re-benchmark — 2.95 it/s is measured for exactly that
geometry, and DropPath adds ~4 %.

## Decoupling DropPath from ConvNeXt

The first draft of this round treated a ConvNeXt body as a **prerequisite** for DropPath. That was never a
scientific claim — `DropPath` is a per-sample Bernoulli mask on the residual branch with **no trainable
variables** — it was purely that the layer had been written alongside `GCNN_ConvNeXtLayer` and only wired into
it, so `GCNN_ResidualLayer` had no way to receive a rate.

The cost of that accident was structural: it put *every* DropPath config downstream of an untested
ConvNeXt-at-base-64 change, so a failure of ConvNeXt neutrality would have moved the whole stage together and
confounded DropPath's transfer with a body swap.

Fixed by adding `drop_path_rate` to `GCNN_ResidualLayer` / `Healpy_ResidualLayer` (deepsphere) and wiring it
through **both** branches of `resnet.py`. Notes on the implementation:

* **The mask is exact here.** `resnet.py` builds `Healpy_ResidualLayer` without a block-level activation (it
  lives inside `layer_kwargs`), so `GCNN_ResidualLayer` takes its `activation is None` path and the skip is the
  pure `x + branch` form. A dropped sample passes through as the exact identity — same semantics as the
  ConvNeXt block, not an approximation.
* **The post-activation form is not**, and warns. With `act(branch + alpha*in)` a dropped branch yields
  `act(alpha*in)` rather than the identity. `RuntimeWarning`, not an error, since the approximation is mild for
  a near-linear activation — and `act_before=True` is exact for any activation with f(0)=0.
* **Checkpoint lineage is preserved** (no new variables), so the knob can be flipped without invalidating an
  existing checkpoint. Do *not* exploit that to warm-start the benchmark run: the cosine and the regularizer
  must both act from initialization for the comparison to mean anything.
* Tests: `test_GCNN_ResidualLayer_drop_path` (exact identity for dropped samples, 1/keep_prob rescale for kept
  ones, inference is a no-op), `test_GCNN_ResidualLayer_drop_path_post_activation_warns`, and
  `test_HealpyGCNN_residual_drop_path` (the rate survives the wrapper → `_get_layer` → block handoff).
  **Not yet run — pytest needs `tf_env` on a compute node.**

### Pre-registered expectation

From the champion's measured 0.918, if the levers composed multiplicatively:

```
0.918 × 1.085 (droppath)                                 ≈ 1.00   → droppath_classic.yaml
0.918 × 1.073 (budget)                                   ≈ 0.985  → budget_classic.yaml
0.918 × 1.085 × 1.056 (attn)                             ≈ 1.05   → convnext_droppath_attn.yaml
0.918 × 1.085 × 1.073                                    ≈ 1.07   → droppath_classic_long.yaml
0.918 × 1.085 × 1.056 × 1.073                            ≈ 1.13   → convnext_droppath_attn_long.yaml
```

Note the first two lines: **parity is predicted from single levers on the champion's own architecture.** If
either lands, the "GCNN loses on combined" result was substantially about budget and regularization rather
than about the encoder.

Levers rarely compose in full and three of the four were measured at base 32, so treat 1.13 as an optimistic
bound and ~1.00–1.05 as the realistic target. **Anything ≥ 1.00 is the first GCNN to match the transformer on
the combined probe.** ~0.95 would still be the best GCNN result to date.

### Why most of stage 2 waits

`convnext_droppath_attn_long.yaml` inherits all of stage 1's untested premises at once and costs 4 × 12 h.
Gate it on: (i) `convnext` ≥ `bench_v5_pool_head_w64` step-corrected, (ii) `convnext_droppath` > `convnext`,
(iii) `convnext_droppath_attn` > `convnext_droppath`, (iv) `budget_classic` has not regressed. If (iii) fails
but (ii) holds, run `convnext_droppath` at 2× budget rather than dropping the arm; if (i) fails outright the
ConvNeXt line is dead and the budget belongs to `droppath_classic_long.yaml`, which shares none of those
premises.

**(iv) is the hard gate on all of stage 2**, because budget is conditional on the readout — see finding 2
above. If `budget_classic` regresses against the champion, every stage-2 arm is void. That is exactly why it
is the unconditional one and can launch alongside stage 1.

**Cost:** stage 1 is 4 × 24 h = 96 node-hours; stage 2 is 48 h per selected arm. `budget_classic` plus one of
the two long arms gives 192 node-hours, comparable to bench_v5's 168.

## Deferred

`_deferred/` (excluded from the submit glob): `attn.yaml` (attention without DropPath — already attributable as
`convnext_droppath_attn / convnext_droppath`), `droppath_strong.yaml` (dose 0.2 — premature before the 0.1
transfer is confirmed, and now most interesting paired with a long arm),
`droppath_deep.yaml` (depth × DropPath — bench_v4 killed depth alone; lowest EV, highest cost). See
`_deferred/README.md`; **their headers still carry the old inflated 16-mock numbers and must be rewritten
before revival.**

## Evaluation

1. **In-training proxy:** `loss/vali_total`. Bar: `t2_cls` −10.817; anchor −10.600.
2. **Headline:** paired FoM(Ωm, S8) over the 1000 `mcmc_samples.h5` mocks via `run_comparison.py`, referenced to
   `t2_cls`. Median of per-mock ratios; **never** the ratio of medians, never the 16-mock grid route.
3. **Seed floor 1.5 %.** Mock CIs are much tighter than that and bound mock-sampling error only.
4. **Step correction.** Wall-clock matching means variants differ in steps; correct with the slope from
   `long.yaml` / `bench_v5_pool_head_w64` (base 64, classic), not the single global +7.3 %.
5. **Do NOT diagnose overfitting from the train–vali gap.** That is what sank the previous draft: budget widened
   the gap while validation loss improved. Use `vali_total`.
6. **Out of scope here:** robustness (posterior bias on the systematics mocks), estimator validity (SBC/TARP/HPD),
   real data (PPC). DES FoM is unsigned under misspecification and is not to be ranked on.

## Status

Configs written and internally validated (all 7 parse with the intended knobs; `budget_classic.yaml` verified
to differ from `bench_v5/2x/pool_head_w64.yaml` in `n_steps` alone). The deepsphere/resnet DropPath change
compiles and is style-clean but its **pytest gate has NOT been run — it needs `tf_env` on a compute node**,
and it is uncommitted in both repos (as is the ConvNeXt block work it sits alongside).

**Nothing benchmarked, nothing submitted.** Next actions, in order:
1. Run `pytest tests/test_gnn_layers.py tests/test_healpy_networks.py` in `tf_env` — the classic-body DropPath
   path is new code on the critical path of the round's lead config.
2. Real 4-GPU benchmark of the convnext-at-base-64 geometry, to replace the three provisional `n_steps`.
3. `droppath_classic.yaml` needs neither, and is the cheapest informative run in the round.
