# bench_v9 — COMBINED, the readout round

**Status (2026-08-14): three arms configured; the readout code they need is written but NOT yet
run — see "Verification" below. Nothing submitted.** `unet_k20.yaml` needs no code and could launch
today; `moments.yaml` and `unet_multiscale.yaml` depend on the new readout paths.

**Goal.** bench_v8 established two things that together determine this round. First, the readout is
the only lever in the programme that has paid twice — `flatten -> mean` was +32.1%, `mean ->
mean_std` +5.3% — and it is the cheapest place in the network to work, acting on the trunk's final
nside-16 map with no new depth, no spatial operator and negligible FLOPs. Second, **every bench_v8
arm that added machinery to the trunk lost**: `injection_conv` 0.836, `poolsplit` 0.887, `fuse_act`
0.905, `unet_k8` 0.933. So this round stays on the readout, and its single trunk arm is bundled with
a readout change rather than run for its own sake.

**Anchor.** `bench_v8/mean_std.yaml`, as run on v18/default in
`runs/v18/default/maps/combined/bench_v8_mean_std` (260000 steps, 2×12 h) — the bench_v8 champion
and the only arm in that round to beat `bench_v7_full`.

## Where bench_v8 left the numbers

Paired FoM against `bench_v7_full`, final checkpoints, from `deep_lss.utils.run_comparison`:

| run | steps | ratio | 95% CI |
|---|---|---|---|
| `bench_v7_transformer` | 340000 | 1.123 | [1.108, 1.134] |
| **`bench_v8_mean_std`** | 260000 | **1.053** | [1.044, 1.065] |
| `bench_v8_k20` | 406500 | 1.020 | [1.013, 1.029] |
| `bench_v7_full` | 250000 | 1.000 | — |
| `bench_v8_injection_conv_k8` | 305400 | 0.970 | [0.962, 0.978] |
| `bench_v7_simple` | 230000 | 0.961 | [0.954, 0.969] |
| `bench_v8_unet_k8` | 335300 | 0.933 | [0.924, 0.941] |
| `bench_v8_fuse_act` | 229000 | 0.905 | [0.897, 0.913] |
| `bench_v8_poolsplit` | 103000 | 0.887 | [0.875, 0.898] |
| `bench_v8_lr_3e4` | 252200 | 0.876 | [0.867, 0.887] |
| `bench_v8_injection_conv` | 118900 | 0.836 | [0.830, 0.844] |
| `bench_v7` (Cls two-point) | 1000000 | 0.726 | [0.711, 0.739] |

`bench_v8_long` (2× budget) was still chaining when this round was written; `long_4` lands
2026-08-15 and **its result should be read before committing a budget here** — if 2× is worth another
+7%, the U-net arm wants 158400 s rather than 79200 s.

Two bench_v8 conclusions that shaped the arms below:

- **The fusion bet failed on its own terms.** `fuse_act` isolates the seam nonlinearity with almost
  no step penalty (229000 vs the anchor's 250000) and loses 9.5%. The "the seam is entirely linear"
  finding is a correct description of the code that does not translate into a lever.
- **The recipe is not LR-limited.** `lr_3e4` at 0.876 with matched steps closes that question.

## Robustness, measured for the first time on v18 (2026-08-14)

Posterior shift in units of the fiducial posterior width, worst over the eight contaminated mocks
and (Om, S8, w0). Instrument: the `chain_<label>_mean.npy` chains in each run's
`ensemble_flow_<steps>/`; the `_stack` variants would put an error bar on these and were not used,
so the ~0.1σ spread among the GCNNs is almost certainly not resolved.

| run | worst | where | `sc_inplace` S8 | `sc_gatti` S8 | `dmo` S8 |
|---|---|---|---|---|---|
| `bench_v8_poolsplit` | 0.58σ | dmo | +0.15 | −0.25 | +0.58 |
| `bench_v8_unet_k8` | 0.68σ | dmo | +0.13 | −0.17 | +0.68 |
| `bench_v7_full` | 0.70σ | dmo | +0.40 | +0.12 | +0.70 |
| **`bench_v8_mean_std`** | **0.72σ** | dmo | +0.44 | +0.32 | +0.72 |
| `bench_v8_k20` | 0.75σ | dmo | +0.50 | +0.32 | +0.75 |
| `bench_v7_simple` | 0.78σ | dmo | +0.27 | −0.05 | +0.78 |
| `bench_v7_transformer` | **1.65σ** | sc_inplace | **+1.65** | **+1.32** | +0.92 |
| `bench_v7` (Cls) | **1.66σ** | sc_gatti | +0.20 | **+1.66** | +0.97 |

Three things follow, and they are the gate for this round:

1. **`mean_std` cost nothing in robustness** — 0.72σ against the anchor's 0.70σ. That is the
   precedent both arms here lean on: a readout change is not obviously a robustness change.
2. **The transformer's 12.3% FoM lead is bought on this axis**, at more than twice any GCNN's bias
   and specifically on source clustering. The GCNN remains the preferred compression.
3. **Gate on `source_clustering_in_place` and `source_clustering_gatti`, not on the worst-overall
   number.** `dmo` biases every architecture by +0.58 to +0.97σ regardless of compression — a baryon
   marginalization property, not an architecture one — and will always win a max. Source clustering
   spreads 0.0 → 1.66σ across runs and is the axis that actually discriminates.

## The arms

| file | knob vs parent | what it tests | readout dim | budget | jobs |
|---|---|---|---|---|---|
| `moments.yaml` | `map_pool` `mean_std` → `moments`, off `bench_v8/mean_std.yaml` | the moment ladder: does standardized skewness + kurtosis extend mean_std's +5.3%? | 1024 → 2048 | 79 200 s | 2 |
| `unet_k20.yaml` | `n_neighbors` 8 → 20, off `bench_v8/unet_k8.yaml` | was unet_k8's 0.933 the U-net schedule, or the sparse graph? | 512 (unchanged) | 79 200 s | 2 |
| `unet_multiscale.yaml` | `map_pool` mean → mean_std at every scale, off `unet_k20.yaml` | multi-scale readout, on the only trunk where it has content | 512 → 1984 | 79 200 s | 2 |

**Every link is one knob.** The U-net side is a chain, and the three results are only interpretable
read together:

```
bench_v8/unet_k8.yaml  --(k 8->20)-->  unet_k20.yaml  --(readout)-->  unet_multiscale.yaml
       0.933                        measures the kernel              measures the readout
```

`unet_k20` earns its slot twice over. It answers a question bench_v8 left open — that round's only
clean single-knob k test was `bench_v8_k20` at 1.020, the one graph-sparsity arm that did not lose,
while both k8 arms moved the architecture as well — and it is the control that makes
`unet_multiscale` attributable. It also **absorbs the step-count confound**: the expensive part of
the U-net side is the k=20 kernel, and since both arms run the same graph at the same wall budget
they should land within a few percent of each other on steps, leaving the readout contrast clean.

Note both U-net arms are quoted **against each other**, not against the `bench_v8_mean_std` anchor —
they still carry all five of `unet_k8`'s schedule knobs, including its width confound (half the width
at every downsampling level except the last, forced by holding the trunk pin at 512).

**One residual ambiguity in `unet_multiscale`'s "one knob".** It is two YAML keys, because "apply the
second moment at every scale" bundles the second moment (worth +5.3% at the trunk alone) with the
multi-scale application (never measured). A ~5% win over `unet_k20` is therefore equally consistent
with "the trunk-only second moment transferred to the U-net and multi-scale added nothing". Only a
substantially larger win is evidence for multi-scale specifically. Separating them cleanly would need
a fourth link (`unet_k20_meanstd`, mean_std at the trunk only), which is deliberately not in this
round — and since the +5.3% reference comes from a different trunk, even subtracting it is soft.

**Why multi-scale belongs on the U-net and essentially nowhere else.** In the default trunk the first
three stages are pure strided `HealpyPseudoConv` and no graph convolution runs until nside 64, so
tapping the readout there would pool barely-processed downsampled maps. The U-net schedule
(`pool_layers` 1, `conv_layers` 4, `conv_widen`) is the only configuration in the repo with genuine
convolved features at nside 256/128/64/32. That is what makes moments-across-scales meaningful, and
it is also the structure the classical statistic already has — moments of the convergence field as a
function of smoothing scale is the starlet/scattering family.

**What was deliberately left out.** A learned Deep-Sets φ (pointwise MLP then mean), GeM pooling, and
soft quantiles all generalize the readout further and were all deferred. The seed floor here is
**1.5%**, so an arm must clear ~2% to be real and — since adopting it permanently raises the
complexity of the default network — ~4-5% to be worth keeping. `mean_std` cleared that at +5.3%. A
mechanism whose plausible payoff is low single digits does not. The `mean_wide` control
(`map_pool: mean` at a matched fused width) was also cut; see the confound note below for what that
costs.

## The confound both arms share

Every arm in this round widens the map readout against a fixed 512-d Cls embedding: `mean_std` moved
it to 1024, `moments` moves it to 2048, `unet_multiscale` to 1984. So each arm changes the maps/Cls
dimensionality balance as well as adding its statistic, and **a win cannot be cleanly attributed
between the two**. This is inherited from `mean_std`, which had the same problem and whose +5.3% is
therefore itself not fully attributed. The control that separates them (`mean_wide`) was cut, so:

> **If either arm wins, `mean_wide` becomes a required follow-up before the result is quoted as
> being about moments or scales at all.**

Pinning the width with `map_feature_dim` would confound differently, by inserting a projection the
pooled-readout lineage deliberately has none of — the 59M-param flatten projection is exactly what
`map_feature_dim` was removed to avoid.

## Sizing

Both arms use the wall-clock budget: `n_steps: auto` + `wall_budget_seconds: 79200` over a 2-job
chain, `job_budget_seconds: 39600`. `n_steps` is an **output** (`throughput.json`), matched on wall
to every bench_v8 arm so the comparison is like-for-like.

- `moments` is nearly free — the readout adds two reductions over the pixel axis — so expect a step
  count within a few percent of `mean_std`'s 260000. No rate probe needed.
- **`unet_multiscale` must be rate-probed before the chain is committed.** k enters only the sparse
  `L @ x` term and the share it can shrink depends only on the conv's output width
  (`spmm_share = (K-1)(k+1) / [(K-1)(k+1) + K*Fout]`), so a U-net's narrow high-resolution convs are
  exactly where k dominates. `unet_k8` measured 221 ms/step and reached 335300 steps; k=20 will reach
  materially fewer. Under equal wall that is a real handicap that is not the mechanism being tested,
  and bench_v8 measured +7.3% for a 2× budget — so decide between 79200 s and 158400 s **before**
  launch. Resizing a live chain means editing `<run_dir>/configs.yaml`, not the repo file.

## Verification status

The readout code below is **written and static-checked, not yet executed**. TensorFlow is not on the
login node, so nothing here has been through a forward pass. Before launching:

1. Smoke-test the two new readout paths on a compute node — `moment_pool` shapes and values, the tap
   count and per-tap widths for the U-net schedule, and that `count_scale_taps` agrees with the
   number of taps actually returned.
2. Rate-probe `unet_k20` and `unet_multiscale` in the same pass so they share a rate reference, and
   decide 79200 s vs 158400 s for both together.
3. Confirm from the training log that each arm built what its config says. The
   `ResNetMapsPlusCLSNetwork:` line now reports `map_pool`, `map_pool_multiscale` and the tap count;
   diff it across arms.

## Code — written 2026-08-14

New module `deep_lss/nets/layers/maps/readout.py` holds the shared readout helpers (`moment_pool`,
`forward_with_pool_taps`, `assemble_scale_taps`, `count_scale_taps`). It lives in `layers/maps/`
rather than in the composite because the multi-res encoder needs the tap helpers too — it owns the
fused seam and therefore has to assemble its own tap list — and an encoder importing a composite
would be backwards.

- **`moments`**: a fourth `map_pool` branch computing `[m1, m2, m3/m2³, m4/m2⁴]`. The third and
  fourth are STANDARDIZED — raw central moments of a LayerNorm'd feature map span orders of magnitude
  across channels and would reach `map_norm` with a conditioning problem the mean/std pair does not
  have. `m2` is `sqrt(var + 1e-6)`, so std ≥ 1e-3 and the divisions inherit that guard; a dead
  channel gives `0/eps**k` rather than a NaN.
- **`map_pool_multiscale`**: `ResNetMultiResEncoder.call(..., return_taps=True)` returns the seam
  plus one tensor per downsampling stage plus the trunk output; the composite reduces each with
  `moment_pool`, applies a **per-tap LayerNorm**, and concatenates. The per-tap norms (`scale_norms`)
  are created at construction from `count_scale_taps` rather than lazily, so `summary()` is complete
  before the trace and the checkpoint structure is fixed. They are `None` when multiscale is off,
  which keeps them untracked and leaves every existing lineage's object graph unchanged.

Both keys are plumbed through all three build sites — `run_training.py:980`,
`run_evaluation.py:413`, `benchmark_resnet.py:200`. That plumbing is the highest-risk part of the
change and was verified by grep, not assumed: bench_v8 found `run_evaluation.py` silently not
forwarding `injection_conv_layers`, which would have trained *with* the conv and evaluated *without*
it. `expect_partial()` swallows that class of mismatch.

`flake8 deep_lss` is back at the 8-finding baseline, all E501, none in the touched files.

## Historical note — what the code prerequisites were

`nets/composite/resnet_maps_plus_cls.py` validates `map_pool not in (None, "mean", "mean_std")`
(:126) and branches on it in `call()` (:197-210). Both arms need new branches.

**`moments` (small):** a fourth validation entry and a readout branch computing
`[m1, m2, m3/m2^3, m4/m2^4]`. Standardize the third and fourth — raw central moments of a LayerNorm'd
feature map span orders of magnitude across channels and would arrive at `map_norm` with a
conditioning problem the mean/std pair does not have. `m2` is already `sqrt(var + 1.0e-6)`, so the
division inherits that guard and a dead channel gives `0/eps^k` rather than a NaN.

**`unet_multiscale` (larger):** the encoder returns only its final feature map today. It needs
(1) `ResNetMultiResEncoder` and the `HealpyGCNN` path it wraps to optionally return the per-stage
outputs, (2) `resnet_maps_plus_cls.py` to apply the mean_std reduction to each tap and concatenate
before `map_projection`/`map_norm`, (3) a `map_pool_multiscale` key threaded through.

**Both:** plumb through **all three** build sites — `apps/run_training.py:977`,
`apps/run_evaluation.py:409`, `apps/benchmark_resnet.py:199` — and **verify rather than assume**.
bench_v8 found `run_evaluation.py` silently not forwarding `injection_conv_layers`, which would have
trained *with* the conv and evaluated *without* it. That failure does not raise; `expect_partial()`
swallows the mismatch. It is the single most likely way either arm produces a plausible, wrong number.

Keep both keys at `network.*` level, **not** under `kwargs:` — `kwargs` is a bare `**` splat and
silently swallows unknown names, which is the commonest way an arm ends up secretly running the
control.

**Both arms are fresh checkpoint lineages** (readout width changes), and `unet_multiscale` doubly so:
`n_neighbors` changes the Laplacian but **no weight shape** — the Chebyshev kernel is `[K*Fin, Fout]`
and the depthwise kernel `[C, K]`, neither depending on k — so a k=8 checkpoint loads into a k=20
network with no error and meaningless filters. Launch both `RUN_NUM=1` into their own `MODEL_DIR`.

## Launching

Via the `submit` skill — do not write a new submission script.
`MODEL_DIR=bench_v9_moments` and `MODEL_DIR=bench_v9_unet_multiscale`, `RUN_NUM=1`, 2-job `afterany`
chains. Rank the results with the `compare-runs` skill against `bench_v8_mean_std`; never by eye.
