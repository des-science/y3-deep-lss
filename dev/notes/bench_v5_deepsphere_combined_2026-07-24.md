# bench_v5 — DeepSphere/GCNN body & readout HPO for the COMBINED probe (round 2)

**Date staged:** 2026-07-24 (configs written + benchmarked; **not yet submitted** — held pending go-ahead)
**Configs:** `y3-deep-lss/configs/deepsphere/combined/bench_v5/*.yaml` (+ `2x/`, `_deferred/`)
**Submission driver:** `y3-deep-lss/submissions/clariden/training.sh`
**Benchmark driver:** `y3-deep-lss/submissions/clariden/benchmark/benchmark_v5.sh`
**Benchmark results:** `/iopsstor/scratch/cscs/athomsen/deep_lss/claude/bench/resnet/v5/benchmark_results.jsonl`
**Data:** v17/baseline (standard-NLA, bta-free), scales `8wl,32gc`, probe `combined` (lensing @nside512 + clustering @nside256), Cls head n_bins 16.
**Predecessor:** `dev/notes/bench_v4_deepsphere_combined_2026-07-23.md` (read that first — bench_v5 is built entirely on its findings).

## Goal (unchanged from bench_v4)

Find a DeepSphere (GCNN) architecture that beats the transformer on the **combined** probe, ideally while being more robust to model misspecification than the transformers. DeepSphere wins each probe individually but loses combined.

- **Target to beat:** transformer `t2_cls` (combined; vali_total −10.82, grid FoM 2842, DES FoM 2378).
- **Round anchor / in-family baseline:** `bench_v5/default.yaml` = a verbatim copy of the **bench_v4 `pool_head` winner** (grid 2306 / DES 1785 / vali −9.98 — best GCNN on every metric, cheapest, most grid-stable). Every bench_v5 variant anchors on this recipe and changes essentially one thing.

## What bench_v4 established (the premises bench_v5 builds on)

bench_v4 ran 13 one-knob variants at a fixed 1×12 h wall-clock. The diagnosis:

1. **The combined gap is a READOUT problem, not a body-capacity problem.** `pool_head` (mean-pool the (B, 448, C) conv features over pixels instead of the flatten-and-crush Dense; Cls emb → 256; fused 512-d) was the clear winner. Pure capacity axes — `w64` (width ×2), `wide_shallow`, `deep_trunk` — barely moved and `w64` was *worse* than default on the VMIM proxy. `w64` was the only capacity variant strong on DES FoM (2nd overall) but was the most step-starved (90 k), so width is left as an *equal-steps* question, not an equal-wall-clock one.
2. **The local cross term and a single tail-attention block did nothing.** `bilinear` (elementwise probe cross term) ≈ default; `global_attn` (one HealpyGlobalAttention block as a *tail* after the whole body) ≈ default (grid 1947). So neither "the 3×2pt cross is a local product" nor "bolt one attention block on the end" is the answer — but the tail placement was a weak proxy for whether global mixing *inside* the body would help.
3. **Grid→DES transfer is the real hazard.** `deep_trunk` (grid 2238 / DES 1321) and `mlp_head` (grid 2222 / DES 1351) looked strong on grid mocks but collapsed on the DES data vector; `mlp_head` overfit hardest (best train loss, worst vali). Only `pool_head` and `w64` were strong on *both*. This is the same mock→data gap tracked in `project_constraining_power_gap_priors`.
4. **Real multi-scale conv did not pay off** at 1×12 h: `graph_unet` (real convs @128/64/32) ≈ default; `graph_unet_256` (real conv at native nside 256) was the *worst* variant (grid 1518) and the slowest. But it was also the most step-starved, so "real conv is expressive but needs more budget/steps" survives as a chained-run hypothesis, not a dead one.

**bench_v5 therefore splits into three tracks:** (A) keep the winning `pool_head` readout fixed and test better *bodies* at equal wall-clock; (B) re-run the round-1 positive levers at the transformer's *actual* 2×12 h budget; (C) shelve the axes bench_v4 already killed.

## Optimization is FIXED (not a variable in this round)

Every bench_v5 config uses the settled recipe — **Adam 1e-4, cosine→0, 5 k warmup (init 1e-5), decay_alpha 0.0, clip_by_global_norm 1.0, batch 16.** This is not re-litigated here because:
- The v16 transformer `bench_t4` ablation already isolated it: `t4_cls` (flat LR) vs `t4_cosine` (only `scheduler: cosine` added) → cosine won and became the standard (`bench_t8/default.yaml`, and the finalized `dev/notes/transformer_v17_unified_recipe_2026-07-19.md`). `t4_adamw` (wd 0.05), `t4_ema` (0.9999) and `t4_lr1e-3` (10× LR) were all tried and **dropped**.
- Direct check on the DeepSphere reference sets (2026-07-24): v17 `v3_cls` (cosine, 190 k) vs v16 `v8_cls` (flat 1e-4, 300 k), DES chains, FoM(Ωm,S8): lensing **1153 vs 879** (+31 % at ⅔ the steps); clustering `v3_cls` cos 160 k **446** vs `v8_cls` flat 600 k **433**. The flat runs kept crawling (lensing 769→792→879 over 100→300 k) or plateaued (clustering 405→403→433 over 200→600 k) — the cosine-to-zero anneal is what sharpens the final minimum. (Confounded by v16→v17 data/codebase, so read it as an upper bound consistent with the clean t4 isolation.)

**Operational consequence for bench_v5:** the FoM gain lives in the anneal *tail*, so `n_steps` must be sized so the cosine actually reaches 0 within the wall-clock — the same rule as bench_v4. Over-sizing `n_steps` (especially in the 2× chained runs) forfeits the gain. "Train longer" is not a free lever (extending the transformer budget past its sized point *hurt* lensing).

## Sizing methodology

Same principle as bench_v4: compare at fixed **wall-clock** (not equal steps), `n_steps` set so the cosine anneal completes within the job.

- **1×12 h round (top-level configs):** anchored to the pool_head **real-train** measurement (job 2880789: 180 k in 10.23 h → 190 k for ~11 h train + ~0.6 h eval/infer tail). Siblings scale by their synthetic `benchmark_v5` step-time ratio, floored to 10 k. Equivalently `n_steps ≈ floor₁₀ₖ(30.3M / step_ms)`; e.g. convnext 130.2 ms → 230 k.
- **2×12 h chained round (`2x/`):** `n_steps` = **total** budget across the chain; cosine spans it. `n_steps ≈ real 4-GPU it/s × 79.2 ks` (≈2×11 h train), floor 10 k. Launched by hand as a 2-job `--dependency=afterany` chain (job 1 TIMEOUTs by design → job 2 restores the last checkpoint; `afterok` would not fire). **Re-size from job 1's actual it/s** before/at launch if the geometry hasn't been real-train-measured.

**Caveat (same as bench_v4):** variants train to different `n_steps`, so this ranks architectures at equal wall-clock — the practical selection criterion. A slower variant that would win at *equal steps* can lose here; that is exactly what the 2× round is for.

## Track A — 1×12 h round (equal wall-clock body comparison)

All share the `pool_head` recipe (mean-pool readout, `map_feature_dim: null`, Cls emb `[512,512,512,256]`, `embedding_dropout_rate 0.1`, `spmm_backend: csr`, `input_norm: true`, multi-res `smooth_nside.clustering 256`, batch 16). Each changes ~one knob vs `default`.

| config | one-knob change vs default | axis | params_M | step_ms | peak_gb | n_steps |
|---|---|---|---|---|---|---|
| **default** | — (= bench_v4 `pool_head` winner) | round anchor | 5.397 | 165.2 | 4.19 | 190k |
| **pool_head_bodydrop** | `body_dropout_rate 0.1` (SpatialDropout1D after each residual block) | trunk regularization / sim→data transfer | 5.397 | 156.9 | 4.10 | 190k |
| **convnext** | `residual_block_type convnext`, `drop_path_rate 0.0` | modern depthwise-separable block (½ the L@x cost) | 4.752 | 130.2 | 4.53 | 230k |
| **convnext_droppath** | convnext + `drop_path_rate 0.1` | stochastic-depth ablation on the convnext block | 4.752 | 130.1 | 4.55 | 230k |
| **attn_body** | + `residual_attention` (HealpyGlobalAttention after res layers 2 & 4) | global cross-probe mixing *inside* the body | 6.682 | 170.7 | 4.79 | 180k |
| **convnext_attn** | convnext body + interleaved attention (both above) | CoAtNet/MetaFormer hybrid | 6.037 | 136.7 | 5.11 | 220k |

Rationale per variant:
- **pool_head_bodydrop** — in the combined path the DeepSphere map branch is built from `get_conv_layers()` (body only), so the head `dropout_rate` never touches it: the map trunk that mines the fine-scale non-Gaussian features is the one unregularized part, and exactly where CosmoGrid→DES Y3 misspecification would bite. `SpatialDropout1D` (channel-wise, not element-wise: neighboring HEALPix pixels are correlated so element drop is recoverable). Zero trainable variables → **lineage-preserving** and ~free compute (identical 5.397 M params). New code: `body_dropout_rate` kwarg in `nets/encoders/maps/gcnn/resnet.py`. **Read by coverage (SBC/TARP/HPD, wide analysis prior) + DES-space PPC, not FoM** — misspecification is signless: dropout can deflate a real constraint (lost info) as easily as relax a spurious one.
- **convnext** — swaps the classic two-ChebK `Healpy_ResidualLayer` for the depthwise-separable ConvNeXt block (`Healpy_ConvNeXtLayer`, arXiv:2201.03545): one depthwise ChebK conv + LN + inverted-bottleneck pointwise GELU MLP (mlp_ratio 4) + LayerScale (1e-6). One graph conv per block ⇒ ~½ the sparse cost ⇒ 17 % faster ⇒ more steps in the same wall-clock. DropPath OFF (at only 5 blocks stochastic depth is a poor fit and would be applied flat).
- **convnext_droppath** — the single DropPath-on ablation of the convnext block (0.1), same 130 ms (DropPath is a cheap Bernoulli mask).
- **attn_body** — the corrected version of bench_v4's failed `global_attn`: instead of one attention block as a *tail*, interleave `HealpyGlobalAttention` *inside* the residual body (after layers 2 & 4 of 5) so graph convs act on globally-mixed features (CoAtNet-style). The body lives in `ResNetMultiResEncoder.gcnn_post` — post-fusion, at nside 16 (~448 tokens, so global attention is cheap) — so these blocks mix the two *fused* probes exactly where 3×2pt cross-info lives. Zero-init pos-emb + LayerScale 1e-4 → near-identity start, keeping the local inductive bias. +1.28 M params, +3.3 % step time (nearly free).
- **convnext_attn** — deliberately stacks the two not-yet-FoM-validated levers, justified because the ConvNeXt block's step-time saving buys the wall-clock room for the attention at ~no budget cost (user call 2026-07-24). A proper token-mix/channel-mix transformer body: depthwise conv = local token-mixer, pointwise MLP = channel-mixer, attention = global token-mixer.

## Track B — 2×12 h chained round (`2x/`, transformer-budget rematch)

`t2_cls` was trained for two 12 h jobs; bench_v4 gave the GCNNs one. This re-runs the bench_v4 round-1 positive levers at the *same* budget so the comparison is finally like-for-like. Selection is bench_v4's structural analysis: the two positive levers were the READOUT (`pool_head`) and, weakly, deep-body WIDTH (`w64`).

| config | body | readout | n_steps (total) | sizing status |
|---|---|---|---|---|
| **pool_head** | base32 anchor body | mean-pool | 380k | real anchor (4.89 it/s × 79.2 ks) |
| **w64** | base64 (deep body 512) | flatten Dense (map_feature_dim 64) | 210k | real anchor (2.72 it/s, job 2880790) |
| **pool_head_w64** | base64 (deep body 512) | mean-pool | 210k | confirmed (job 2891966: 290.7 ms ≈ w64 290.9; 18.55M, 7.15 GB) |
| **pool_head_unet** | graph-U-Net (base16, pool 2, conv 3, conv_widen; real convs @128/64/32) | mean-pool | 320k | synthetic-validated (172.8 ms ≈ v4 graph_unet 173.6 → 4.11 it/s proxy) |

- **pool_head / w64 / pool_head_w64** = the round-1 winner, the capacity axis at equal steps, and the two stacked. `w64` keeps its (losing) flatten readout on purpose so `pool_head_w64` isolates width-on-the-winning-readout.
- **pool_head_unet** = the "expressive body needs more budget" hypothesis. Puts bench_v4's `graph_unet` body on the winning readout (it was confounded by the flatten readout in v4) and gives it the 2× budget a slower-ramping body needs. First post-fusion real conv at nside 128 (one octave finer than pool_head's 64) = more equivariant cross-probe convolution at high resolution — the most misspecification-robust direction.

## Track C — deferred (`_deferred/`, shelved not deleted)

Excluded from the submit/benchmark glob (`bench_v5/*.yaml` does not recurse). Each re-tests a body-capacity axis bench_v4 already found flat or bad on the classic residual block; all are ConvNeXt-body re-skins of a dead v4 axis. Revive only if `convnext` beats the `default` anchor at equal budget (then re-test capacity on the winning block).

| config | axis | bench_v4 precedent | step_ms |
|---|---|---|---|
| `convnext_deep` (res 10) | depth | `deep_trunk` overfit grid→DES | 167.8 |
| `convnext_wide` (base 64) | width | `w64` barely moved; step-starved here | 239.4 |
| `convnext_bigk` (poly 8) | spectral reach | `poly8` lost | 199.6 |
| `convnext_poolsplit` (conv @256, base 64) | fine-res real conv | `graph_unet_256` = worst v4 variant; slowest | 355.0 |
| `injection_conv` (1 ChebK conv @256 on fused stream) | fine-res real conv | same axis as `graph_unet_256`; 351.9 ms → only 80 k steps @1×12 h = too step-starved (user call 2026-07-24) | 351.9 |

## How to submit (NOT yet run — held)

Track A, one 12 h job per top-level config:
```bash
cd /users/athomsen/dlss/repos/y3-deep-lss
for f in configs/deepsphere/combined/bench_v5/*.yaml; do
  name=$(basename "${f%.yaml}")
  ARCH=deepsphere PROBE=combined \
    NET_CONFIG="$PWD/$f" MODEL_DIR="bench_v5_${name}" \
    sbatch --parsable --job-name="bv5_${name}" \
    --export=ALL,RUN_NUM=1 submissions/clariden/training.sh
done
```
`pool_head_bodydrop` single job (example): `MODEL_DIR=bench_v5_pool_head_bodydrop`, `--job-name=bv5_ph_bdrop`.

Track B (2×), per config, launched by hand as a 2-job chain (see each `2x/*.yaml` header for the exact `J1=$(sbatch --parsable … RUN_NUM=1)` then `sbatch --dependency=afterany:$J1 … RUN_NUM=2`).

- `ARCH=deepsphere` is a wandb tag only; architecture is set by `network.name: resnet`.
- `PROBE=combined` → `PROBE_CONFIG=combined_nla`, `LOSS` auto-selects `vmim` (flow head).
- Each job runs train → `run_evaluation.py` → `run_inference.py` automatically. Cls asinh cache already exists (no per-job precache race).
- Outputs: `/iopsstor/scratch/cscs/athomsen/deep_lss/runs/v17/baseline/maps/combined/bench_v5_<name>/`.

## Evaluation & selection

1. **Fast in-training proxy:** `loss/vali_total` (VMIM). Bar: `t2_cls` ≈ −10.82; `default`/pool_head ≈ −9.98 (the gap to close). Higher MI = better.
2. **Headline metric:** FoM from the inference chains, param pair (Ωm, S8), same pipeline as bench_v4. Grid FoM = median over the 16 grid mock chains; DES FoM = the single `chain_DESy3` (cleanest cross-run number). Median-over-grid still carries ~3–5 % seed noise — do not over-read <5 % differences.
3. **Sanity gate:** combined FoM ≥ each single-probe DeepSphere FoM (still a TODO carried from bench_v4 — pull single-probe DeepSphere FoM with the same pipeline).
4. **Transfer gate (critical here):** watch grid-vs-DES. A variant strong on grid but weak on DES is overfitting (`deep_trunk`/`mlp_head` in v4). Trust only variants strong on both.
5. **`pool_head_bodydrop` special-cases (2)/(4):** it is a *regularization* probe on a signless axis. Judge it by **coverage (SBC/TARP/HPD, wide analysis prior) + DES-space PPC**, with FoM secondary: lower FoM + better coverage = win; lower FoM + unchanged coverage = lost information.

Diagnostic reads:
- `convnext`/`convnext_droppath` beats `default` → the modern block is the lever (then Track C capacity axes reopen on it).
- `attn_body` or `convnext_attn` beats `default` → global cross-probe mixing *inside* the body is the discriminator (vindicates the "readout+mixing, not capacity" diagnosis; v4's tail-only `global_attn` was just the wrong placement).
- `pool_head_bodydrop` improves coverage/PPC → the map trunk was overfitting sim-specific texture (relevant to the CosmoGrid→DES gap).
- Track B: `w64`/`pool_head_w64` beats `pool_head` → width *does* help but only at equal steps (v4's equal-wall-clock round suppressed it); `pool_head_unet` beats `pool_head` → real multi-scale conv needed the longer budget.

## Status

All configs written and **synthetic-benchmarked (all `status: OK`, including `pool_head_w64` — job 2891966, 290.7 ms, so every 2× n_steps is now measured-anchored)**; nothing submitted. Standing hold: no full training run launches without explicit go-ahead.
