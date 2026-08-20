# bench_v10 — CLUSTERING arms

**Status: CONFIGS WRITTEN, NOTHING LAUNCHED** (2026-08-17).

**The round rationale, the arm table, the 1×/2× argument, the scoring rules and the
dimensionality confound all live in
[`../../combined/bench_v10/README.md`](../../combined/bench_v10/README.md) — read that first.**
This file holds only what is specific to clustering.

| config | knobs vs `bench_v7/full.yaml` | jobs | `wall_budget_seconds` | readout dim |
|---|---|---|---|---|
| `mean_std_1x.yaml` | `map_pool` mean → mean_std | 1 | 39 600 | 1024 |
| `mean_std_2x.yaml` | same | 2 | 79 200 | 1024 |
| `mean_std_k20_1x.yaml` | + `n_neighbors` 60 → 20 | 1 | 39 600 | 1024 |
| `mean_std_k20_2x.yaml` | same | 2 | 79 200 | 1024 |
| `unet_multiscale_1x.yaml` | U-net schedule + k20 + `map_pool_multiscale` | 1 | 39 600 | **1920** |

The `_1x`/`_2x` pairs are **byte-identical below `name:` except for the two budget lines** — the
siblings were generated from the 2× parent rather than hand-copied, so the bodies cannot drift.

## Clustering is the round's hardest transfer, which is why it is worth running

The density field is smoothed with a ≥ 57′ kernel and enters at **nside 256**, so it has the least
small-scale structure of the three inputs and the fewest scales left to resolve. Two consequences:

- `mean_std`'s prior is weakest here — the across-sky variance of a feature on a heavily smoothed
  field has the most chance of being redundant with the probe's own Cls, which the composite already
  feeds the head directly.
- It is also the probe where the **trunk machinery matters most**: `full` beats `simple` by **8.5%**
  here against 2.7% on lensing. So readout-only conclusions drawn on combined do not transfer for
  free, and this is the arm that would catch it.

`unet_multiscale_1x` sharpens the same question. If moments-across-scales is a real mechanism it
should be *weakest* here (four downsampling stages against lensing's five, on the smoothest field);
if it is merely the width patch the combined numbers suggest, it should transfer as well here as
anywhere. The two readings predict **opposite orderings** against the lensing arm.

## Geometry — `base_channels` is the one setting these files decide on their own authority

Trunk width = `base_channels × 2^(pool_layers + conv_layers − 1)`, and the trunk must be **512 wide
at nside 16 (~448 tokens)** on every probe. Clustering enters at nside 256, so it needs one stage
fewer than lensing and combined:

| file | entry nside | stages | `base_channels` | trunk |
|---|---|---|---|---|
| `mean_std*.yaml` | 256 | pool 2 + conv 2 | **128** | 128 × 2² = 512 |
| `unet_multiscale_1x.yaml` | 256 | pool 1 + conv 3 | **64** | 64 × 2³ = 512 |
| *(lensing/combined U-net, for contrast)* | 512 | pool 1 + conv 4 | 32 | 32 × 2⁴ = 512 |

**This is mechanical, not tidiness.** With a pooled readout the trunk width *is* the map-vector
width, so base 64 on the `mean_std` arms would emit a 512-d map (not 1024) into a concat whose Cls
branch contributes 512 — silently unbalancing the fusion that `embedding_layers: [..., 512]` exists
to balance, and breaking comparability with the other two probes. Same reasoning as bench_v7's
`base_channels: 128`; see [`../bench_v7/README.md`](../bench_v7/README.md).

`smooth_nside: {clustering: 256}` keeps the per-probe smoothing kernel at nside 256 (5.9× faster than
smoothing at native) and requires `scale_cuts.clustering.theta_fwhm_base` in the scales config —
`8wl,32gc` has it (40.4). Clustering is the only active probe, so there is **no multi-resolution
encoder and no injection seam** (confirmed from the bench_v7 training logs: no `ResNetMultiResEncoder`
line); the composite takes the plain `self.gcnn` path. `unet_multiscale_1x` therefore has **4 taps**
— 64@128, 128@64, 256@32, 512@16 = 960 ch — and its log must print `(4 scale taps)`, **five on
lensing**. A silently dropped tap is a narrower readout that still trains and still scores.

## Sizing

Wall-clock budget throughout — `n_steps: auto`, so no rate probe is needed and `n_steps` is an
**output** in `throughput.json`. For orientation only, `bench_v7_full` measured **3.59 it/s** on v18
(job 3020974), which puts the 1× arms near 145 k steps and the 2× near 284 k. Note that k=20 saves
*less* here than on combined: k enters only the sparse `L @ x` term, whose share is
`(K−1)(k+1) / [(K−1)(k+1) + K·Fout]`, and this probe's convs run at wider `Fout`. Do not assume the
1.51× measured on combined.
