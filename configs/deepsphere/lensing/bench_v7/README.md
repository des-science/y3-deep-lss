# bench_v7 — LENSING arms

**Status: CONFIGS WRITTEN, NOTHING LAUNCHED** (2026-08-06).

Part of the three-probe bench_v7 round. **Shared rationale, the full-vs-simple contrast, the trunk-width
table, the v18 specifics and the scoring rules all live in
[`../../combined/bench_v7/README.md`](../../combined/bench_v7/README.md)** — read that first. This file
holds only what is specific to lensing.

| config | body | DropPath | attention | `base_channels` | trunk | it/s | `n_steps` |
|---|---|---|---|---|---|---|---|
| `simple.yaml` | classic | 0.0 | none | 64 | 512 | 3.02 | **110 k** |
| `full.yaml` | convnext | 0.1 | every 2 | 64 | 512 | 3.50 | **130 k** |

**Both run as ONE 12 h job, not a 2 × 12 h chain** — that is the single-probe convention, and it is
what keeps the winner directly adoptable as a 1-job default. `n_steps = it/s × 37.8 ks`, floored to
10 k. Rates **measured on v18** (jobs 3020918 / 3020973); see the shared README's sizing section for
the estimator and the three ways these numbers are deliberately conservative.

Plus the transformer reference arm, `../../../transformer/lensing/bench_v7/transformer.yaml`
(renamed from `default.yaml` on 2026-08-11) — the v17
unified recipe **unchanged** (trunk 1024 already matches combined; no fix needed on this probe).
**It runs 150 k in ONE 12 h job, not a 2 × 12 h chain: 150 k is this probe's measured optimum and
250 k regressed on all FoM variants.** Do not extend it to match the GCNN arms' wall.

## No GCNN lensing reference has ever existed

Every DeepSphere bench round (v4/v5/v6) ran on **combined only**. The sole single-probe GCNN runs on
v17 are `lensing/v3_cls` and `clustering/v3_cls`, both using the **flatten readout at base 32** — the
configuration measured at 0.399 against the two-point baseline on combined.

So **no lever in the bench_v6 table has been verified on lensing alone.** Treat these arms as the
first real measurement on this probe, not as a confirmation of the combined result. The magnitudes may
not transfer; the readout finding is the only one there is any structural reason to expect to.

## Attention is the weakest-justified knob here

Its stated mechanism on combined was long-range **cross-probe** mixing — bench_v4 fingered cross-probe
mixing as the source of the combined-probe gap — and there is no cross-probe mixing in a single-probe
run. It measured 1.005–1.056 on combined at base 64, straddling the 1.5% seed floor.

It is kept in `full.yaml` anyway, for one reason: all three probes share one recipe, so the
combined-vs-single sanity gate compares like with like. It costs ~3% of step time. **If the lensing
full-vs-simple gap comes out smaller than combined's, attention losing its cross-probe rationale is the
first thing to suspect.**

## Geometry

Lensing enters at native nside 512; `pool_layers 3 + conv_layers 2` = 5 stages, each halving nside, so
the residual body sits at **nside 16 (~448 footprint tokens)** — identical to clustering and combined.
No `smooth_nside`: lensing is the only active probe and stays native throughout.

`base_channels: 64` → trunk `64 × 2^3 = 512`, matching the other two probes.

## Sizing — measured on v18, 2026-08-06

Both rates come from real 4-GPU `training.sh SKIP_EVAL=1` probes at batch 16, not from the synthetic
single-GPU sweep. **`full` 3.50 it/s, `simple` 3.02 it/s** → ×37.8 ks → 132.3 k / 114.2 k → floored to
**130 k / 110 k**.

ConvNeXt is **×1.16 faster** than the classic block here, reproducing bench_v6's ×1.17 at base 64. So
`full` buys ~18% more steps than `simple` at equal wall — roughly +1.5% of unearned FoM advantage,
which runs *against* the round's hypothesis and therefore makes a `simple` win more convincing, not
less.

**The lower-bound argument this file used to carry was thinner than it claimed.** It assumed
lensing-only must be faster than combined's 3.25 because it drops the clustering branch and the
multi-res fusion. That held — 3.50 — but by only 2.5%, not the comfortable margin implied. The
measurement replaces the argument.

See the shared README for the estimator (second-half window, not tqdm's cumulative figure, which
under-reads by ~9%) and the asymmetry rule.
