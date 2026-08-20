# bench_v10 — CLUSTERING arms

**Status: COMPLETE** (2026-08-20) — all seven jobs ran and were scored. Results below; the text above is left as written, as the record of what was predicted.

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

## Results

Paired FoM vs `bench_v7_full` (fixed `n_steps` 130 000), `run_comparison`, v18/default, 1000 mocks
over 1000 cosmologies, highest evaluated checkpoint. Seed floor 0.049.

| arm | steps as run | ratio | 95% CI | win% | `vali_total` |
|---|---|---|---|---|---|
| `unet_multiscale_1x` | 212 300 | **1.075** | [1.058, 1.089] | 68% | −9.541 |
| `mean_std_k20_2x` | 389 000 | **1.070** | [1.055, 1.088] | 66% | −8.119 |
| `mean_std_k20_1x` | 221 000 | **1.070** | [1.057, 1.080] | 68% | −9.276 |
| `mean_std_2x` | 283 100 | **1.056** | [1.048, 1.068] | 67% | −9.098 |
| `bench_v7_full` | 130 000 | 1.000 | — | — | −9.188 |
| `mean_std_1x` | 140 200 | 0.982 = | [0.973, 0.992] | 45% | −9.244 |

Both 2× arms also carry a mid-chain checkpoint (138 500 and 145 800), at cosine-midpoint with a live
learning rate; the table is the final checkpoint throughout.

**The hard transfer half-worked.** `mean_std` alone is a wash at 1× (0.982) and only +5.6% at 2× —
the weakest showing of the three probes, as the ≥57′ smoothing predicted. But it does not fail.

**Clustering saturates at one job, unlike lensing.** `mean_std_k20` is **1.070 at both 1× and 2×** —
a doubled budget bought nothing, against +11.8% for the same doubling on lensing. So the single-probe
budget convention can stay at one job *here*; the "single probes are budget-limited" conclusion from
the master README is a lensing statement, not a general one.

**k=20 is the clearest win in the round on this probe**: 1.070 vs 0.982 at 1×, on a cheaper step. On
the sparser graph the readout transfers; on the dense one it does not.

**`unet_multiscale` is the top arm here (1.075)** and, unlike on lensing, this result **is
interpretable**: the dimensionality confound only makes a high-ratio arm's *loss* ambiguous, and this
is a win at 3.75:1 against the pinned 512 Cls embedding. Multi-scale readout is real on the smoothed
density field. Lensing put the same arm last (0.937) — the opposite ordering the arm was designed to
produce, which is the round's cleanest evidence that the effect is about scale content and not a
generic width patch.

Note the ordering against the master README's expectation: this is the probe where `full` beats
`simple` by 8.5%, i.e. where trunk machinery matters most, and it is also where the *readout* arms
cluster tightly (1.056–1.075, all within a floor of each other). Nothing here separates the top four.
