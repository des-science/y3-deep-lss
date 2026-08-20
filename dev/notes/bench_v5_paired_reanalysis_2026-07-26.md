# bench_v5 re-analysis on the 1000-mock paired route (2026-07-26)

**What this is:** every v17 combined-probe run re-ranked with `deep_lss/apps/tuning/run_comparison.py` against the
transformer `t2_cls`, on the **1000-mock `mcmc_samples.h5`** set with **per-mock pairing** on the full
`(i_sobol, i_signal, i_noise)` tuple.

**Why it was redone:** the bench_v5 conclusions — and therefore every premise written into the
`configs/deepsphere/combined/bench_v6/*.yaml` headers — came from the legacy **16-cosmology `chain_grid_*`
median** route. That route has a ~2.4 % floor and cannot resolve anything below ~7 %. Two of its readings
are wrong by more than their own claimed error bar, and **one of them is sign-reversed**.

**Runs:** `/iopsstor/scratch/cscs/athomsen/deep_lss/runs/v17/baseline/maps/combined/`
**Mock set:** 1000 mocks / 1000 cosmologies (2 signal × 5 noise), 1 row per cosmology → unclustered bootstrap.
**Metric:** FoM(Ωm, S8) = det(cov)^−½ per mock; reported number is the **median of the per-mock ratios**.
**Floor:** run-to-run (training-seed) scatter ~1.5 %. The mock CIs below are much tighter than that; they
bound the mock-sampling error only, never the seed error.

## Headline table — everything vs `t2_cls`

| run | steps | paired ratio | 95 % CI (mocks) | win % | vali_total |
|---|---|---|---|---|---|
| `t2_cls` (transformer) | 260 k | 1.000 | — | — | −10.817 |
| **`bench_v5_pool_head_w64`** | 210 k | **0.918** | [0.908, 0.925] | 25 % | −10.600 |
| `bench_v5_convnext_droppath` | 200 k | 0.886 | [0.874, 0.892] | 16 % | −10.370 |
| `bench_v5_convnext_attn` | 200 k | 0.853 | [0.846, 0.861] | 13 % | −10.269 |
| `bench_v5_pool_head` | 380 k | 0.848 | [0.838, 0.857] | 13 % | −10.653 |
| `bench_v5_convnext` | 210 k | 0.807 | [0.799, 0.818] | 8 % | −10.189 |
| `bench_v5_pool_head_unet` | 320 k | 0.789 | [0.781, 0.797] | 7 % | −10.141 |
| `bench_v5_default` | 190 k | 0.784 | [0.776, 0.798] | 8 % | −10.086 |
| `bench_v5_attn_body` | 180 k | 0.777 | [0.768, 0.784] | 8 % | −9.962 |
| `bench_v5_pool_head_bodydrop` | 190 k | 0.746 | [0.733, 0.753] | 6 % | −8.222 |
| `bench_v5_w64` | 210 k | 0.680 | [0.666, 0.695] | 4 % | −9.392 |

The rank order is unchanged from the 16-mock route; the **levels** differ (anchor 0.918 not 0.900, round
baseline 0.784 not 0.801) and the **lever sizes** differ a lot. `bench_t2_b20_260k` reproduces `t2_cls` at
ratio 1.000 / identical vali, confirming it is the same run promoted under a curated name — a useful
end-to-end null for the pairing code.

## Lever table — each contrast against its own base

Every pair below was verified by diffing the two runs' own `configs.yaml` snapshots, so the listed knob
plus `n_steps` is the *entire* difference.

| lever | contrast | steps | bench_v6 header claims | **re-derived (1000 mocks)** | win % | verdict |
|---|---|---|---|---|---|---|
| mean-pool readout @ base 64 | `pool_head_w64` / `w64` | 210 k / 210 k | — | **1.321** [1.298, 1.351] | 95 % | biggest lever in the programme; step-matched, no correction needed |
| **budget ×2** | `pool_head` / `default` | 380 k / 190 k | "**ZERO** gain, 8/16, 1.006" | **1.073** [1.065, 1.083] | 73 % | ⚠️ **SIGN-REVERSED — budget is real** |
| DropPath 0.1 (convnext body) | `convnext_droppath` / `convnext` | 200 k / 210 k | +12.7 %, 14/16 | **1.085** [1.078, 1.095] | 78 % | real, ~⅔ the claimed size |
| width ×2 (on pool readout) | `pool_head_w64` / `pool_head` | 210 k / 380 k | +15.6 %, 15/16 | **1.084** [1.074, 1.091] | 80 % | real — and won with 45 % **fewer** steps |
| global attention (convnext body) | `convnext_attn` / `convnext` | 200 k / 210 k | +7.9 %, 13/16 | **1.056** [1.048, 1.063] | 72 % | real, smaller |
| ConvNeXt body vs classic | `convnext` / `default` | 210 k / 190 k | −0.5 %, 7/16 | **1.024** [1.016, 1.035] | 59 % | ≈ neutral once the +10 % steps are removed — claim survives |
| global attention (CLASSIC body) | `attn_body` / `default` | 180 k / 190 k | +0.5 %, 8/16 (null) | **0.984** [0.978, 0.990] | 43 % | null-to-negative → the body-type interaction is confirmed |
| graph-U-Net body | `pool_head_unet` / `pool_head` | 320 k / 380 k | — | **0.934** [0.926, 0.941] | 24 % | loses; "expressive body needs budget" is dead |
| body dropout | `pool_head_bodydrop` / `default` | 190 k / 190 k | dead | **0.943** [0.933, 0.951] | 32 % | dead confirmed (vali −8.222 is the worst of the round) |

### Step-correcting the confounded rows

bench_v5 sized `n_steps` to wall-clock, so most contrasts carry a step difference. Using the one measured
budget slope (+7.3 % per doubling, from the clean `default`→`pool_head` pair) to first order:

| lever | raw | step-corrected to equal steps |
|---|---|---|
| width ×2 on pool readout | 1.084 | **≈ 1.15** (it gave up 0.86 doublings) |
| DropPath 0.1 | 1.085 | ≈ 1.09 |
| global attention (convnext) | 1.056 | ≈ 1.06 |
| ConvNeXt body | 1.024 | **≈ 1.01 → inside the seed floor, i.e. neutral** |

The width correction lands on ≈ +15 %, which is where the 16-mock route put it (+15.6 %) — that agreement
is a coincidence of two errors cancelling, not a validation of the old route. Treat the corrected column as
indicative: the slope is measured at a single point, on one architecture, and is assumed log-linear.

## The two findings that matter

### 1. Budget is a real lever. The "2× budget bought zero gain" result was an artifact of the 16-mock floor.

`bench_v5_default` → `bench_v5_pool_head` is a **pure budget test**: the config diff is `n_steps`
190 k → 380 k and `checkpoint_every` (plus the run directory). Nothing else. Paired result:

* FoM **+7.3 %**, CI [1.065, 1.083], winning **73 %** of 1000 mocks — ~5× the seed floor.
* `vali_total` **−10.086 → −10.653**, an 0.567-nat improvement.

The bench_v5/v6 reading was that the extra steps bought *overfitting*, evidenced by the train–vali gap
flipping +0.225 → −0.509. That inference does not hold: the **validation** loss improved substantially and
so did the FoM. A widening train–vali gap with a falling vali loss is not overfitting, it is just a train
loss falling faster. Budget belongs back on the lever list, between attention (+5.6 %) and DropPath (+8.5 %).

### 2. The budget lever is conditional on the readout — flatten overfits with steps, mean-pool does not.

The one place where "more steps hurt" is real is the **flatten** readout. `v3_cls` (flatten readout,
`map_pool: null`) carries three evaluated checkpoints, and its paired ratio vs `t2_cls` **degrades
monotonically**:

| `v3_cls` checkpoint | 150 k | 200 k | 310 k |
|---|---|---|---|
| paired ratio vs `t2_cls` | **0.545** | 0.302 | 0.270 |

The fully-annealed 310 k endpoint is half as good as a mid-anneal 150 k checkpoint. (Caveat: 150 k/200 k are
mid-cosine, so their LR state differs — this is suggestive, not a clean budget ladder. The clean evidence is
the `default`/`pool_head` pair above.) The same signature appears in bench_v5 proper: `w64` (flatten, base 64)
is the round's worst run at 0.680 with vali −9.392, while the identical body behind a mean-pool readout
(`pool_head_w64`) reaches 0.918 / −10.600 — the **+32.1 %** readout lever, the largest measured anywhere.

So the readout is not merely the best single knob. It is what makes budget *safe to spend*. Every bench_v6
config already uses `map_pool: mean`, which is exactly why the round should be spending budget rather than
freezing it.

## Consequences for bench_v6 as currently written

1. **The round's central design decision is void.** Every `bench_v6/*.yaml` header carries "BUDGET: n_steps is
   STEP-MATCHED to 210 k across every config … bench_v5 found ZERO paired grid gain … so bench_v6 does NOT
   vary the budget." That is built on the sign-reversed null. As written the round declines a +7.3 % lever.
2. **Worse, step-matching at 210 k makes the ConvNeXt cost saving unspendable.** ConvNeXt is ~21 % cheaper at
   base 32; at base 64 the inverted bottleneck dominates and the saving should be larger (classic block =
   2 ChebK convs ≈ 12·N·C²; ConvNeXt = depthwise ChebK + pointwise MLP ≈ 8·N·C², so ≈ 0.67 asymptotically).
   Pinned at 210 k, six ConvNeXt configs would leave ~⅓ of their allocated 2×12 h wall idle — spending the
   cost lever on nothing. Its whole justification in the header is "it buys back the wall-clock the width
   costs", and step-matching then refuses the refund.
3. **The lever arithmetic needs restating.** `droppath_attn.yaml` predicts 0.900 × 1.127 × 1.079 = 1.09.
   Re-derived: 0.918 × 1.085 × 1.056 = **1.05**, and both lever measurements were made at base 32 on a
   ConvNeXt body, so transfer to base 64 is assumed rather than measured. Adding the budget lever back:
   0.918 × 1.085 × 1.056 × 1.073 ≈ **1.13**.
4. **Claims that survive unchanged:** ConvNeXt is quality-neutral (so it is a pure cost lever); attention pays
   only on a ConvNeXt body; DropPath is real at 5 blocks; width needs the pooled readout; body dropout is dead.

## Tooling

`run_comparison.py` gained a wandb fallback for `vali_total`: a login node has neither tensorboard nor pip, so
the column was silently `n/a` — half the ranking recipe missing. It now reads `loss/vali_total` from
`wandb/run-*/files/wandb-summary.json` (highest `global_step`), and **raises** if that step disagrees with the
evaluated checkpoint, since a copied run directory inheriting another run's wandb id is a known failure mode
here (`t1_cls`'s lensing twin). Values agree with the previously-recorded tensorboard numbers to all printed
digits. FoM and pairing logic untouched.

Reproduce:

```bash
cd /users/athomsen/dlss/repos/y3-deep-lss
.venv/bin/python3 -m deep_lss.apps.tuning.run_comparison \
  --root /iopsstor/scratch/cscs/athomsen/deep_lss/runs/v17/baseline/maps/combined \
  --reference t2_cls bench_v5_pool_head_w64 bench_v5_convnext_droppath   # ... etc
```

(The repo-root `.venv` has no pyyaml; `y3-deep-lss/.venv` does and is what the command above uses.)

## Scope

Informativeness only. Robustness (posterior bias on the systematics mocks), estimator validity (SBC/TARP/HPD)
and real-data behaviour (PPC) are separate questions with separate instruments, and none of the numbers above
speaks to them. DES FoM is unsigned under misspecification and was deliberately not computed.
