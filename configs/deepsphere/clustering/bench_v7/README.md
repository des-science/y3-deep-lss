# bench_v7 — CLUSTERING arms

**Status: CONFIGS WRITTEN, NOTHING LAUNCHED** (2026-08-06).

Part of the three-probe bench_v7 round. **Shared rationale, the full-vs-simple contrast, the trunk-width
table, the v18 specifics and the scoring rules all live in
[`../../combined/bench_v7/README.md`](../../combined/bench_v7/README.md)** — read that first. This file
holds only what is specific to clustering.

| config | body | DropPath | attention | `base_channels` | `pool_layers` | trunk | it/s | `n_steps` |
|---|---|---|---|---|---|---|---|---|
| `simple.yaml` | classic | 0.0 | none | **128** | 2 | 512 | 3.07 | **110 k** |
| `full.yaml` | convnext | 0.1 | every 2 | **128** | 2 | 512 | 3.59 | **130 k** |

**Both run as ONE 12 h job, not a 2 × 12 h chain** — the single-probe convention.
`n_steps = it/s × 37.8 ks`, floored to 10 k. Rates **measured on v18** (jobs 3020974 / 3020975).

Plus the transformer reference arm, `../../../transformer/clustering/bench_v7/transformer.yaml`
(renamed from `default.yaml` on 2026-08-11).
**Clustering is the only probe where the transformer recipe also had to change**: it runs at nside 256,
so it gets 4 nested levels against lensing's and combined's 5, and its trunk was **512 against their
1024** — the same defect and the same 2× factor as the GCNN here, fixed the same way
(`base_embed_dim` 32 → 64). Read that file's header before keeping it: unlike the GCNN case the
mismatch was *not* mechanically breaking anything (`map_feature_dim: 512` already equalizes what
reaches the head), the arm is no longer the validated v17 clustering architecture, and body FLOPs go
~4×. Its `n_steps` is an **assumed, unmeasured** rate — measure on job 1.

## `base_channels: 128` — the one setting bench_v7 changes on its own authority

Trunk width = `base_channels × 2^pool_layers`. Clustering enters at nside 256 and therefore uses
`pool_layers: 2` (not 3), so base 64 would give a **256**-wide trunk against lensing's and combined's
512. `128 × 2^2 = 512` restores the match.

The old `clustering/maps+cls.yaml` deliberately used base 64 to match a **base-32** lensing config
(`64·2² == 32·2³ == 256`). That pairing is obsolete now that lensing moves to a 512 trunk.

**This matters mechanically, not just for tidiness.** With `map_pool: mean` the pooled map vector *is*
the trunk width. At base 64 clustering would emit a 256-d map vector into a concat whose Cls branch
contributes 512-d — silently unbalancing the fusion that `embedding_layers: [..., 512]` exists to
balance, and breaking comparability with the other two probes. The old default got away with it only
because it used the flatten readout, where trunk width and map-vector width are decoupled by the
`map_feature_dim` projection.

## No GCNN clustering reference has ever existed

Same as lensing: every bench round ran on combined only, and `clustering/v3_cls` is a flatten-readout
base-64 run. No lever in the bench_v6 table has been verified on this probe. See
[`../../lensing/bench_v7/README.md`](../../lensing/bench_v7/README.md) for the attention caveat, which
applies here identically — there is no cross-probe mixing in a single-probe run.

## Geometry

`pool_layers 2 + conv_layers 2` = 4 stages, each halving nside, from 256 → **nside 16 (~448 footprint
tokens)** — the same body resolution as lensing and combined, so the attention blocks cost the same.

`smooth_nside: {clustering: 256}` keeps the per-probe smoothing kernel at nside 256 (5.9× faster than
smoothing at native). It requires `scale_cuts.clustering.theta_fwhm_base` in the scales config;
`8wl,32gc` has it (40.4).

## Sizing — measured on v18, and `base_channels: 128` costs nothing

This was the round's one genuinely unknown rate: `base_channels: 128` is a width the GCNN had **never
been run at**, so unlike lensing there was no lower-bound argument available, only a projection — and
a projection across a width change is exactly what over-predicted the ConvNeXt advantage by ~25% in
bench_v6 and nearly killed three runs.

**Measured 2026-08-06, and it resolved in the good direction: 3.59 it/s on `full` — the FASTEST arm in
the whole sizing sweep**, marginally quicker than lensing's identical architecture (3.50). `simple`
came in at 3.07. The pixel-channel argument held: stage 1 is nside-256 × 128 ch against combined's
nside-512 × 64 ch, roughly half the work.

→ ×37.8 ks → 135.7 k / 116.0 k → floored to **130 k / 110 k**.

ConvNeXt is **×1.17 faster** than the classic block here, matching bench_v6's base-64 figure exactly.
`full` therefore buys ~18% more steps than `simple` at equal wall (~+1.5% of unearned FoM advantage),
a bias that runs *against* the round's hypothesis.

See the shared README for the estimator and the asymmetry rule.
