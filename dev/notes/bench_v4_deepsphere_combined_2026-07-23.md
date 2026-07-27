# bench_v4 — DeepSphere/GCNN architectural HPO for the COMBINED probe

**Date submitted:** 2026-07-23
**Configs:** `y3-deep-lss/configs/deepsphere/combined/bench_v4/*.yaml`
**Submission driver:** `y3-deep-lss/submissions/clariden/training.sh`
**Data:** v17/baseline (standard-NLA, bta-free), scales `8wl,32gc`, probe `combined` (lensing @nside512 + clustering @nside256), Cls head n_bins 16.

## Goal

Find a DeepSphere (GCNN) architecture that beats the transformer on the **combined** probe, ideally while being more robust to model misspecification than the transformers.

- **Target to beat:** transformer `t2_cls`
  `/iopsstor/scratch/cscs/athomsen/deep_lss/runs/v17/baseline/maps/combined/t2_cls`
- **In-family baseline:** DeepSphere `v3_cls`
  `/iopsstor/scratch/cscs/athomsen/deep_lss/runs/v17/baseline/maps/combined/v3_cls`
  (`bench_v4/default.yaml` is a verbatim copy of the `v3_cls` recipe = `configs/deepsphere/combined/maps+cls.yaml`.)

## The controlling fact + hypothesis

DeepSphere `v3_cls` **beats** the transformer on each probe **individually** but **loses combined**. The wins-individually filter eliminates everything the combined config *shares* with the winning single-probe configs (fine-branch depth, resolution, smoothing, the local concat+Dense fusion plumbing — all row-alignment-asserted and input-norm-balanced, no bug). The one combined-specific thing left is **capacity**: the graph-conv body has the *same* width/depth as the single-probe body but must now carry lensing + clustering + their (local) cross-correlation.

- Default DeepSphere: `base_channels=32` → deep body **256 ch** @nside16 (~448 px), fusion pinch = 32.
- Transformer: body grows to **1024 ch** (4× DeepSphere's deep width) — oversized for one probe (hence it *loses* individually) but adequate for two.

Two earlier hypotheses were **rejected**: (a) fusion-vector width 64-vs-512 — the VMIM summary handed to the flow is only ~9-dim (`out_features`), so a 64-dim intermediate upstream is not the information bottleneck; (b) missing global attention — the 3×2pt cross signal is between fields at the *same* sky position and is captured by local graph conv + pooling. `bench_v4` still probes both (`pool_head`/`mlp_head`, `global_attn`) as controls, but the **lead axis is capacity** (`w64` is the PRIMARY candidate).

## Sizing methodology (this round = ONE 12 h job per variant, no chaining)

This round compares architectures at a fixed **wall-clock** budget (one 12 h job), not equal steps. `n_steps` per variant is set from its measured benchmark step time so the **cosine LR schedule completes within 12 h** — a slow variant that only reaches a fraction of `n_steps` would never anneal its LR, making the comparison unfair.

Formula per variant:
```
measured step_ms  (benchmark_v4, batch 16)
  → real 4-GPU it/s  via K = 1.37 (synthetic-benchmark → real 4-GPU)
  → × ~11 h effective train (39.6 ks; reserves ~1 h for the eval+inference tail)
  → floor to nearest 10k
```
Anchor: `default` = job 2878896 = 160.9 ms/step → 4.54 real it/s → 170k. Siblings scale as `170000 × 160.9 / step_ms`.

**Caveat for the writeup:** because step time differs, variants train to different `n_steps`. This ranks architectures at equal wall-clock, which is the practical selection criterion — but a slower/expensive variant that would win at *equal steps* could still lose here. Read winners with that in mind before promoting to a chained run.

## The 13 configs

All share: `spmm_backend: csr`, `input_norm: true`, multi-res (`smooth_nside.clustering: 256`), `map_feature_dim: 64`, Cls `embedding_layers [512,512,512,64]`, batch 16, Adam 1e-4 cosine, warmup 5000. Each changes essentially ONE knob vs `default`.

| config | one-knob change vs default | resulting geometry | axis | step_ms | n_steps | bench job |
|---|---|---|---|---|---|---|
| **default** | — (= `v3_cls` recipe) | base32, ds3, cheby2, res5 → deep **256** @16, pinch 32 | baseline | 160.9 | 170k | 2878896 |
| **w64** | `base_channels 32→64` | deep **512**, pinch 64 | **width ×2 (PRIMARY capacity)** | 290.9 | 90k | 2878896 |
| **wide_shallow** | `base 32→64` + `res 5→2` | deep 512, shallow trunk | width vs depth trade | 218.1 | 120k | 2879368 |
| **deep_trunk** | `residual_layers 5→10` | deep 256, deeper trunk | depth capacity | 233.6 | 120k | 2878896 |
| **less_cheby** | `ds 3→4`, `cheby 2→1` | extra doubling → deep **512**, trunk 16 | cheap strided pooling vs real convs | 176.5 | 150k | 2879708 |
| **graph_unet** | base16, ds2, cheby3, `cheby_layers_double` | deep 256, real convs @128/64/32 | real multi-scale conv (U-Net) | 173.6 | 150k | 2879981 |
| **graph_unet_256** | base16, ds1, cheby4, `cheby_layers_double` | deep 256, real convs @**256**(½-width)/128/64/32 | real conv at native shared res | 289.2 | 90k | 2880248 |
| **poly8** | `poly_degree 5→8` | deep 256 | larger receptive field / conv | 256.2 | 110k | 2878896 |
| **bernstein** | Bernstein basis, `poly_degree 3` | deep 256 | filter basis (cost-matched to Cheby-K5) | 258.8 | 100k | 2879894 |
| **global_attn** | + `global_attention` @nside16 | deep 256 + attn tail | global mixing (transformer-like) | 163.1 | 160k | 2879368 |
| **mlp_head** | `dense_layers []→[256]`, `dropout 0.1→null` | nonlinear fusion head | head nonlinearity | 159.3 | 180k | 2878896 |
| **pool_head** | `map_pool mean`, `map_feature_dim null`, Cls emb `…→256` | mean-pool readout, fused 512 | readout + wider fusion | 156.6 | 180k | 2878896 |
| **bilinear** | `fusion concat→bilinear` | + elementwise cross term x·inj | probe-fusion cross term | 171.6 | 160k | 2878896 |

Notes:
- **bernstein** was originally a Chebyshev→Bernstein basis swap at K=5 (~3.9× cost). A true O(K) Bernstein is impossible (no single-op recurrence), and the L-ladder rewrite only reaches ~1.13×. FINAL decision: keep the Bernstein basis but drop to `poly_degree 3` so it is **cost-matched** to Chebyshev-K5 (9 spmm/conv, 258.8 ms, ~1.61× default, *fewer* params 11.9M vs 13.4M). Cost-match > order-match.
- **graph_unet / graph_unet_256** use `cheby_layers_double` (couple channel doubling to each real-conv downsampling — the graph-U-Net schedule). csr has no 2^31 output ceiling (that's coo-only), so the nside-256 real conv is feasible; it's kept half-width (Fin 16) to bound compute.
- Dropped before submission (dominated / premature): `more_cheby` (flat-256 width, 2.2× graph_unet cost for the same reach), and the pre-staged 2×12 h `default_2x`/`w64_2x` (pre-committed winners before the comparison).

## How the jobs were submitted

One 12 h job per config via `training.sh`, driven by env vars:
```bash
for f in configs/deepsphere/combined/bench_v4/*.yaml; do
  name=$(basename "${f%.yaml}")
  ARCH=deepsphere PROBE=combined \
    NET_CONFIG="$PWD/$f" \
    MODEL_DIR="bench_v4_${name}" \
    sbatch --parsable --job-name="bv4_${name}" submissions/clariden/training.sh
done
```
- `ARCH=deepsphere` is a wandb tag only; the architecture is set by `network.name: resnet` in the config.
- `PROBE=combined` → `PROBE_CONFIG=combined_nla`, and `LOSS` auto-selects `vmim` (flow head) — the finalized combined default.
- `MODEL_DIR=bench_v4_<name>` keeps the runs grouped under `maps/combined/` and clear of `v3_cls`/`t2_cls` (the default `MODEL_DIR` would be the bare config basename, e.g. `default` — too generic).
- Each job runs train → `run_evaluation.py` (grid+DES+mocks) → `run_inference.py` (4-flow MAF, sample posterior) automatically. The Cls asinh cache (`rebinned_nb16_8wl,32gc.h5`, 2.3 GB) already exists, so no per-job precache race.

**SLURM job IDs (2026-07-23):**

| job | MODEL_DIR |
|---|---|
| 2880774 | bench_v4_bernstein |
| 2880776 | bench_v4_bilinear |
| 2880778 | bench_v4_deep_trunk |
| 2880780 | bench_v4_default |
| 2880782 | bench_v4_global_attn |
| 2880784 | bench_v4_graph_unet |
| 2880785 | bench_v4_graph_unet_256 |
| 2880786 | bench_v4_less_cheby |
| 2880787 | bench_v4_mlp_head |
| 2880788 | bench_v4_poly8 |
| 2880789 | bench_v4_pool_head |
| 2880790 | bench_v4_w64 |
| 2880791 | bench_v4_wide_shallow |

Outputs: `/iopsstor/scratch/cscs/athomsen/deep_lss/runs/v17/baseline/maps/combined/bench_v4_<name>/`
Per-step logs: `…/bench_v4_<name>/logs/<jobid>_1_mirrored_{training,evaluation,inference}.log`
SLURM launcher stdout: `submissions/clariden/slurm/slurm-<jobid>.out`

## Evaluation & selection

1. **Fast in-training proxy:** `loss/vali_total` (VMIM). Reference: transformer `t2_cls` vali_total ≈ −10.82; DeepSphere `v3_cls` ≈ −8.86 (the gap to close). Higher MI = better.
2. **Headline metric:** FoM from the inference chains (same `ensemble_flow_*` → FoM pipeline as `v3_cls`/`t2_cls`). Median-over-grid FoM still carries ~3–5 % training-seed noise — do not over-read <5 % differences.
3. **Sanity gate:** combined FoM must be ≥ each single-probe DeepSphere FoM.
4. **Diagnostic reads:**
   - `w64`/`wide_shallow` win → **capacity confirmed** (width is the lever).
   - `deep_trunk` wins → depth, not width.
   - `graph_unet`/`graph_unet_256` win → real multi-scale convolution matters (and this is the *most* misspecification-robust direction — aligns with the reason for using DeepSphere at all).
   - `global_attn` wins → global mixing was the discriminator after all.
   - `pool_head`/`mlp_head`/`bilinear` win → it's the readout/fusion, not the body.
   - `poly8`/`bernstein` win → spectral filter expressiveness/conditioning.

## Round-1 RESULTS (2026-07-24)

All 13 jobs `COMPLETED` (none TIMEOUT/FAILED), each reached its full `n_steps`, and every run wrote the full eval + `ensemble_flow_*` chain set. Training-only wall was 8.6–10.7 h with only a ~0.6 h eval+infer tail (I had reserved ~1 h), so several variants under-used the 12 h budget — **the configs were resized on 2026-07-24 to fill ~11 h of training** (`n_steps ← floor₁₀ₖ(old × 11.0 / measured_train_h)`); the results below are from the *original* (shorter) `n_steps` in the table above.

FoM = `det(cov(param1,param2))^(-0.5)` (numerical_results.ipynb `FoM_from_chain`), param pair **(Om, S8)** with S8 = s8·√(Om/0.3); param order `[Om,s8,w0,Aia,n_Aia,bg1,bg2,bg3,bg4]`. Grid FoM = **median over the 16 grid mock chains**; DES FoM = the single `chain_DESy3` data vector (the cleanest cross-run number — identical data). Higher = better.

| run | steps | vali_total | grid FoM (Om,S8) | DES FoM (Om,S8) | grid FoM (S8,w0) |
|---|---|---|---|---|---|
| **t2_cls (TARGET, transformer)** | 260k | **−10.82** | **2842** | **2378** | **477** |
| **pool_head** | 180k | **−9.98** | **2306** | **1785** | **354** |
| w64 | 90k | −8.99 | 2226 | 1619 | 304 |
| wide_shallow | 120k | −9.14 | 2156 | 1569 | 318 |
| deep_trunk | 120k | −9.01 | 2238 | 1321 | 322 |
| mlp_head | 180k | −8.36 | 2222 | 1351 | 286 |
| less_cheby | 150k | −8.80 | 2065 | 1348 | 305 |
| poly8 | 110k | −8.97 | 1982 | 1379 | 303 |
| global_attn | 160k | −9.34 | 1947 | 1477 | 260 |
| graph_unet | 150k | −9.07 | 1895 | 1425 | 324 |
| bilinear | 160k | −9.24 | 1874 | 1439 | 321 |
| default (= v3_cls recipe) | 170k | −9.37 | 1873 | 1528 | 298 |
| bernstein | 100k | −8.64 | 1821 | 1282 | 288 |
| graph_unet_256 | 90k | −8.54 | 1518 | 1318 | 254 |
| v3_cls (baseline; grid unstable) | 310k | −8.86 | *559* | 1674 | *100* |

**Findings:**
1. **No variant beats the transformer.** `t2_cls` still leads on every metric; the best GCNN (`pool_head`) reaches 81 % of its grid FoM and 75 % of its DES FoM. The gap is not closed by any single architectural knob at this budget.
2. **`pool_head` is the clear round winner** — best on VMIM proxy, grid-median FoM, DES FoM, *and* the highest grid-FoM floor (min 1608 vs default's 379 → most robust across cosmologies), while being the **cheapest** variant (5.4M params, fastest step). The lever is the **readout**: replace the 114688→64 flatten-and-crush Dense with a permutation-invariant mean-pool over footprint pixels (mirrors the transformer's token pool), Cls branch widened to 256, fused 512-d.
3. **Capacity hypothesis (the pre-registered lead) is NOT supported as the main lever.** `w64`/`wide_shallow`/`deep_trunk` sit only modestly above `default` on grid FoM and `w64` is *worse* than `default` on VMIM. Pure width/depth is not the discriminator. (`w64` is 2nd-best on DES FoM and trained only 90k, so width may still help at equal steps — a chained-run candidate.)
4. **Watch grid-vs-DES transfer.** `deep_trunk` (grid 2238 / DES 1321) and `mlp_head` (grid 2222 / DES 1351) look strong on grid mocks but collapse on the DES data vector — the mock→data transfer gap flagged in `project_constraining_power_gap_priors`. `pool_head` and `w64` are the only variants strong on *both*, so they are the trustworthy leads. `mlp_head` also overfits hard (best train `loss/main` −11.69, worst `vali_total` −8.36).
5. **`graph_unet_256` and `bernstein` are the weakest** — a real conv at native nside 256 and the Bernstein basis both underperform `default`. `graph_unet` (real convs at 128/64/32) ≈ `default`. Real-multi-scale-conv did not pay off here.
6. **`v3_cls` grid FoM (559) is a genuine instability**, not a computation artifact: same grid indices as the others, and it is actually *tighter* than `default` on some grid points but bimodal (p25 = 296, min = 51) across the 16 — half its grid cosmologies blow up. Its DES FoM (1674) is normal. `bench_v4_default` (170k, batch 16, flow head) is markedly more grid-stable than `v3_cls` (310k, batch 20) — consistent with the batch-16 + flow-head/standardization regime being the better baseline. **Use `v3_cls`'s DES FoM, not its grid FoM, as the baseline.** `default` from this round supersedes `v3_cls` as the in-family reference.

**Sanity gate (combined FoM ≥ each single-probe DeepSphere FoM): NOT yet checked** — needs the single-probe DeepSphere FoM pulled with the same pipeline. TODO before promoting a winner.

## Next round (planned)

**Selected for the 2×12 h chained round: `pool_head` (primary) and `w64` (capacity check at equal steps).** `pool_head` because it wins on both grid and DES FoM and is cheapest; `w64` because it is 2nd on DES FoM and was the most step-starved (90k), so the equal-steps/longer-budget regime is exactly where it could show the capacity effect that this equal-wall-clock round suppressed. Consider a `pool_head`+width combo (`pool_head` readout on a `base_channels 64` body) as a follow-up if both help.

The winning architecture(s) get a **2×12 h chained** run to compare against `t2_cls`/`v3_cls` at full budget (12 h was enough for a single probe but NOT combined for the transformers, so the winners likely need the longer budget to show their true ceiling). Mechanism: `training_chainer.sh` (job 1 TIMEOUTs → resubmit `RUN_NUM=2 --restore_checkpoint`). The pre-staged `*_2x` configs were deleted this round on purpose — recreate them for the *actual* winners, not pre-committed guesses. If a slow-but-strong variant (e.g. `w64`, `graph_unet_256`) looks competitive at its reduced `n_steps`, it is a prime candidate for the chained run where equal-steps rather than equal-wall-clock applies.
