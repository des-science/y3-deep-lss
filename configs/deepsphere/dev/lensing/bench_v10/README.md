# bench_v10 — LENSING arms

**Status: CONFIGS WRITTEN, NOTHING LAUNCHED** (2026-08-17).

**The round rationale, the arm table, the 1×/2× argument, the scoring rules and the
dimensionality confound all live in
[`../../combined/bench_v10/README.md`](../../combined/bench_v10/README.md) — read that first.**
This file holds only what is specific to lensing.

| config | knobs vs `bench_v7/full.yaml` | jobs | `wall_budget_seconds` | readout dim |
|---|---|---|---|---|
| `mean_std_1x.yaml` | `map_pool` mean → mean_std | 1 | 39 600 | 1024 |
| `mean_std_2x.yaml` | same | 2 | 79 200 | 1024 |
| `mean_std_k20_1x.yaml` | + `n_neighbors` 60 → 20 | 1 | 39 600 | 1024 |
| `mean_std_k20_2x.yaml` | same | 2 | 79 200 | 1024 |
| `unet_multiscale_1x.yaml` | U-net schedule + k20 + `map_pool_multiscale` | 1 | 39 600 | **1984** |

The `_1x`/`_2x` pairs are **byte-identical below `name:` except for the two budget lines** — the
siblings were generated from the 2× parent rather than hand-copied, so the bodies cannot drift.

## Why lensing is where the default gets decided

**Q1.** Lensing convergence is the most non-Gaussian input in the programme, so it is where a
second-moment readout has the strongest prior — `mean` keeps only the monopole of each of the 512
feature maps, and the across-sky scatter of a nonlinear local feature is exactly the non-Gaussian
signal the map analysis exists to capture.

**Q2, and this is the binding constraint.** Source clustering acts on lensing, and it is the axis
that discriminates architectures (0.0 → 1.66σ across bench_v8/v9, against `dmo`'s +0.58…+0.97σ for
*every* architecture regardless of compression). The pattern so far is that `sc_gatti` S8 bias rises
monotonically with FoM ratio — more informative compressions are more exposed — and the transformer
buys its 12.3% combined lead at 1.65σ. **An arm that wins Q1 here and loses Q2 is not the answer.**
Gate on `source_clustering_{gatti,in_place}`, not on the worst-overall number.

## Geometry — unchanged from bench_v7

`base_channels 64`, `pool_layers 3`, `conv_layers 2`: five stages from nside 512 down to
**nside 16 (~448 footprint tokens)**, trunk 512. No `smooth_nside` — lensing is the only active probe
and stays native throughout, so there is **no multi-resolution encoder and no injection seam**. The
composite therefore takes the plain `self.gcnn` path, confirmed from the bench_v7 training logs
(no `ResNetMultiResEncoder` line).

`unet_multiscale_1x.yaml` is the exception and rescales the schedule to base 32 / pool 1 / conv 4 —
the same U-net as `combined/bench_v9/unet_multiscale.yaml`, since lensing enters at the same nside
512. Its five taps (32@256, 64@128, 128@64, 256@32, 512@16 = 992 ch) are the **first seam-free**
multi-scale readout in the programme; on combined the coarsest tap is the fused clustering stream.
The log must print `(5 scale taps)`.

## Sizing

Wall-clock budget throughout — `n_steps: auto`, so no rate probe and no measured it/s is needed and
`n_steps` is an **output** in `throughput.json`. For orientation only, `bench_v7_full` measured
**3.50 it/s** on v18 (job 3020918), which puts the 1× arms near 140 k steps and the 2× near 275 k;
k20 should be roughly 1.5× that. **This is the first lensing round on the wall-clock budget** —
bench_v7 used a fixed `n_steps: 130000` — so the 1× arms are not exactly bench_v7's budget and are
not a re-run of it.
