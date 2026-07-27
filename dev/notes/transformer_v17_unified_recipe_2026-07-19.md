# v17 transformer benchmark wrap-up and the unified cross-probe recipe (2026-07-19)

Summary of the v17/baseline transformer benchmark campaign (`t0_*`, `t1_cls`, `bench_t2_*` under
`runs/v17/baseline/maps/{lensing,clustering,combined}`) and the decision it produced: **one fixed
architecture + training recipe for all probes**, now encoded in the top-level
`configs/transformer/<probe>/maps{,+cls}.yaml`.

FoM throughout = 1/sqrt(det Cov(Om, S8)), S8 = s8*sqrt(Om/0.3), three variants per run:
mock = `chain_fiducial_bench_mean`, grid_med = median over the 16 `chain_grid_*`, DES = `chain_DESy3`.

## The decision

Fix the bench_t8/default architecture with the b20 / cosine / 250k-step training recipe everywhere.
The only per-probe differences are input geometry: clustering smoothed at nside 256 (sets the working
resolution for clustering-only; injected at the nside-256 level for combined), lensing at native 512.

- Architecture: `nested_transformer`, base_embed_dim 32, growth double, 4 heads, 3 window levels,
  1 local block/level, 1 global block, mlp_ratio 4, no layerscale, no pos-encoding, concat merge,
  fp32_softmax, bf16, input_norm, token_nside 16; head dropout 0.1.
  maps+cls: map_feature_dim 512, Cls branch [512 x 4] MLP, asinh_per_feature, embedding dropout 0.1.
- Training: local batch 20 (global 80 on 4x GH200), Adam 1e-4, cosine -> 0 with 5k warmup
  (init 1e-5), global-norm clip 1.0, checkpoint_every 10000. **n_steps is per-probe, sized to the
  12 h wall (finalized 2026-07-20 after the escalation runs, below): lensing 150k (one job),
  clustering 250k (one job), combined 250k (two chained jobs).** The architecture, batch, and
  schedule shape are identical across probes; only n_steps and the input geometry differ.

Rationale for a single recipe: the per-probe optimum is this exact config on combined and lensing,
and within ~2-3% of the clustering optima on mock/DES (best grid_med outright). The apparent
clustering architecture preferences (embed64, no head dropout) were rescues of an under-trained
b48/120k baseline, not real preferences — see below.

## Per-probe results

### Lensing (t0 sweep, all b20/150k cosine; t0_default arch = the recipe above)

| run | mock | grid_med | DES |
|---|---|---|---|
| **t0_default** (= t1_cls copy) | **1446.0** | **1071.9** | **1090.4** |
| t0_no_dropout | 1404.1 | 1089.4 | 1081.6 |
| t0_block_dropout | 1427.6 | 1011.8 | 1109.1 |
| t0_flat | 1361.2 | 926.9 | 1092.8 |
| t0_masked | 1333.9 | 974.8 | 1069.6 |
| t0_multiscale | 1267.2 | 1044.8 | 973.9 |

Default wins; every feature variant (flat LR, masked attention, multiscale, block dropout,
no dropout) is neutral-to-negative. (t0_geodesic never produced chains; the distance-bias idea was
already killed by the t7 symmetric factorial.)

### Clustering (bench_t2 sweep on the collapsed t1_cls baseline)

| run | knobs vs baseline | mock | grid_med | DES |
|---|---|---|---|---|
| t1_cls (baseline) | b48/120k cosine | 384.6 | 480.3 | 242.3 |
| **bench_t2_b20** | **b20/250k cosine (= unified recipe)** | **629.3** | **621.5** | **376.1** |
| bench_t2_embed64 | b20/240k, base_embed_dim 64 | 647.9 | 531.4 | 384.3 |
| bench_t2_embed64_no_dropout | both knobs | 622.7 | 561.4 | 375.4 |
| bench_t2_no_dropout | b48/120k, head dropout null | 577.8 | 543.8 | 421.6 |
| bench_t2_flat | b48/120k, flat LR | 632.5 | 574.9 | 384.6 |
| bench_t2_gmm_head | b48/120k, GMM VMIM head | 566.1 | 598.2 | 379.7 |
| bench_t2_embed64_b48 | embed64 at b48/110k | 406.2 | 471.6 | 273.1 |
| bench_t2_global4 / local2 / block_dropout | depth/dropout variants | 421.6 / 425.3 / 356.5 | 427.3 / 456.5 / 466.6 | 255.1 / 268.4 / 248.6 |

Reading: the b48/120k-cosine baseline sits at a trainability threshold and many *different* knobs
rescue it (no_dropout, embed64, flat LR, GMM head, b20/250k) to a common ~600-650 mock plateau; the
knobs do **not** stack (embed64_no_dropout <= either parent). The pure budget/batch knob (b20/250k)
reaches the plateau with the unmodified architecture, so no per-probe architecture change is
justified.

### Combined (bench_t2 on the healthy t1_cls baseline)

| run | knobs | mock | grid_med | DES |
|---|---|---|---|---|
| t1_cls (baseline) | b20/130k cosine | 3297.0 | 2405.2 | 1844.9 |
| bench_t2_no_dropout | head dropout null | 3027.6 | 2526.8 | 1771.9 |
| bench_t2_b10 | b10/240k | 3676.7 | 2812.0 | 2173.7 |
| bench_t2_b10_no_dropout | b10/240k + no dropout | 3641.0 | 2727.4 | 2089.9 |
| **bench_t2_b20_260k** | **b20/260k, 2 chained jobs** | **3979.3** | **2841.6** | **2378.2** |

b20_260k = +21/+18/+29% over the baseline and 1.69/1.56/1.66x the DeepSphere v8_cls combined best
(2359.7/1816.0/1435.2). 2x-updates-AND-2x-examples (b20_260k) beats updates-only (b10), especially
on DES (+9%); dropout removal does nothing for combined.

## Durable lessons

1. **Total optimizer budget helped combined but is NOT a universal monotone lever** (revised
   2026-07-20 — see the budget-escalation results section). It lifted combined (b20_260k), but
   lensing regressed at 250k vs 150k, and clustering at 500k split (DES up, grid_med down). "Train
   longer / more jobs" is not the next escalation; per-probe budgets differ and 150k is already
   optimal for lensing.
2. Dropout removal / embed64 / flat LR were clustering-collapse rescues only; on healthy runs they
   are neutral or negative. Head dropout stays at 0.1.
3. Knobs found in isolation do not stack (clustering embed64_no_dropout, combined b10_no_dropout).
4. Median-over-grid FoM does not remove the ~3-5% training-seed noise; differences below that are
   not decisions.

## Wall-clock sizing at 250k (measured, 4x GH200, 12 h wall, ~35 min eval+inference tail)

| probe | measured rate (b20) | 250k training | jobs |
|---|---|---|---|
| clustering | 7.34 it/s (job 2781360) | 9.5 h | **1** (total 10.0 h) |
| lensing | 3.85 it/s (job 2768132) | 18.0 h | 2 chained (job 2 ~6.5 h + tail) |
| combined | 3.35-3.61 it/s (jobs 2790186/87) | ~20 h | 2 chained (job 2 ~9 h + tail) |

Chained pattern (validated end-to-end 2026-07-19): submit job 1 normally; it hits the 12 h wall
mid-training (expected TIMEOUT, cosmetic); resubmit the same command with `RUN_NUM=2` — n_steps is
the TOTAL budget, run_training resumes from the last checkpoint and falls through to eval when the
budget is reached. Sizing rule: real rates come in ~10% below the K-calibrated synthetic bench;
size for training + tail <= wall - 1 h.

## t2_cls: the unified-recipe reference runs

`runs/v17/baseline/maps/<probe>/t2_cls`:

- lensing: copy of `t0_default` — exact match to the finalized 150k default (the default architecture
  was lifted from this very run), no retrain. Chains at `ensemble_flow_150000`; FoM 1446/1072/1090.
  The earlier 250k run (jobs 2794791/92) was swapped OUT of the baseline and preserved as
  `t2_cls_250k` — it is the escalation evidence (250k regressed, see results section), not baseline.
- clustering: copy of `bench_t2_b20` — exact config match (250k / ckpt 10k), no retrain. 629/621/376.
- combined: copy of `bench_t2_b20_260k` — near-match (260k steps, ckpt 5k vs the 250k/10k default;
  the delta is within seed noise, but its chains are `ensemble_flow_260000`). 3979/2842/2378.

All three are maps+cls runs (there are no maps-only reference runs). Each corresponds to its
top-level `configs/transformer/<probe>/maps+cls.yaml` default. Copies inherit the source wandb run id
(inert unless resumed with --wandb).

## Budget-escalation results (2026-07-20, both chains COMPLETED)

Both chains ran exactly as sized: lensing 12 h TIMEOUT + 6.0 h resume (~18 h); clustering 500k
12 h TIMEOUT + 7.3 h resume (~19 h). FoMs (Om, S8), same three variants:

| run | mock | grid_med | DES |
|---|---|---|---|
| lensing t0_default @150k (winner) | 1446.0 | 1071.9 | 1090.4 |
| **lensing t2_cls @250k** | 1351.0 | 960.5 | 1032.1 |
| clustering bench_t2_b20 @250k (recipe) | 629.3 | 621.5 | 376.1 |
| **clustering bench_t2_b20_500k @500k** | 609.8 | 485.9 | 461.8 |

Two results that **revise lesson 1**:

- **Lensing @250k did NOT beat @150k** — it regressed (mock -6.6%, grid_med -10.4%, DES -5.3%). The
  mock/DES moves are near the seed-noise band but grid_med is clearly beyond it, and nothing improved.
  Extending the cosine budget past 150k does not help lensing; if anything the longer schedule (more
  steps at high LR before the anneal) hurts. **The 250k default is not validated for lensing** — 150k
  was already at/past this probe's optimum.
- **Clustering @500k is a split decision**: mock flat, grid_med -21.8% (real regression), but
  **DES +22.8% -> 461.8, which beats the previous transformer best (422, no_dropout@120k) AND the
  DeepSphere v8_cls clustering DES best (439)** — the first time the transformer wins clustering DES.
  Diverging grid_med (sim) vs DES (data) at higher budget is the data-vs-sim compression-sharpening
  signature from the constraining-power investigation, here pointing the "wrong" way (data up, sim
  down). Do NOT read the DES win as a clean recipe upgrade: grid_med is the sim-faithful metric and
  it dropped a fifth.

Net: **total budget is NOT a universal monotone lever** — it helped combined (b20_260k), was neutral
-to-negative for lensing at 250k, and gave clustering a data/sim split at 500k. The 250k unified
default still stands as the cross-probe compromise (it is at/near each probe's plateau and cheap), but
"just train longer" is dead as the next escalation. Revisit lesson 1 below with this caveat.

## Known gaps / open items

- **Clustering DES gap: closed at 500k, but at a grid_med cost.** 461.8 @500k beats 439 (DeepSphere)
  and 422 (no_dropout@120k), but grid_med fell 621->486. Whether the DES win is real signal or the
  data-sim mismatch flattering a sharper compression on the one real sightline is unresolved — a PPC
  on the 500k clustering summary would discriminate. Not promoting 500k to the default on a single
  seed with a regressed grid_med.
- **RESOLVED (2026-07-20): per-probe 12 h budgets adopted.** Rather than 250k everywhere, each probe
  is sized to fill about one 12 h job, with combined (heavier data) allowed two. Lensing dropped to
  150k (its validated optimum, ~11 h, one job); clustering stays 250k (~10 h, one job); combined
  stays 250k (~20 h, two chained jobs). Top-level configs updated. This keeps every single-probe run
  to one submission and matches the evidence (lensing gains nothing past 150k; clustering's 500k DES
  win came with a grid_med regression we are not banking on a single seed).
- v17 v1_cls/v2_cls (DeepSphere) combined chains were absent at analysis time; the v8_cls comparison
  numbers are the v16-era references.
