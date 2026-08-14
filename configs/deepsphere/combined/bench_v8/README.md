# bench_v8 — COMBINED, closing the transformer gap without losing robustness

**Goal.** bench_v7 put the combined nested transformer **12.3% ahead** of the best GCNN
(`bench_v7_transformer` 1.153 vs the `simple` anchor; `bench_v7_full` 1.041). The GCNN is
nonetheless the preferred compression because it is **robust to misspecification**. This round tries
to close the FoM gap *while keeping that*, so every arm carries a robustness gate, not just a FoM
number.

**Anchor.** `bench_v7/full.yaml` — the bench_v6 champion recipe (ConvNeXt + DropPath 0.1 +
interleaved attention, trunk 512, mean-pool readout), as run on v18/default in
`runs/v18/default/maps/combined/bench_v7_full` (250 k steps, 2×12 h; the config now says 200 k for a
fresh chain — see its `!! PROVENANCE !!` block).

## Why the gap is a *fusion* gap, not a capacity gap

The GCNN ties or wins both single probes and loses only combined, so the deficit is specific to
combining. The multi-res encoder log says where it comes from:

```
fine group nside=512 (4 ch, 1 layers), coarse group nside=256 (4 ch)
injected at 64 channels (15 layers after the fusion), fusion=concat, injection_conv_layers=0
```

Three pure strided `HealpyPseudoConv` stages take 512 → 64 **before any graph convolution runs**, so
the first genuine conv is at nside 64, no equivariant conv ever touches the finest and most
non-Gaussian scales, and the two probes are combined only by a **pointwise Dense** at the seam. The
transformer does content-dependent mixing at 512 and 256 through its local windows.

Note what is *already* correct and should not be "fixed": the GCNN does use the multi-res injection
(`ResNetMultiResEncoder`, clustering at native 256), and the downsampling is a *learnable* strided
pseudo-conv, which preserves the NEST child-order gauge. Orientation matters enormously here —
bench_t7's symmetrized merge cost ~2.25–3× FoM.

## The arms

| file | knob(s) vs `bench_v7/full.yaml` | what it tests | budget | jobs |
|---|---|---|---|---|
| `long.yaml` | training budget 79.2 ks → 158.4 ks | does budget still pay at the champion's operating point? | 158 400 s | 4 |
| `lr_3e4.yaml` | `learning_rate` 1.0e-4 → 3.0e-4 | is the recipe LR-limited? (**first optimizer arm in the programme**) | 79 200 s | 2 |
| `mean_std.yaml` | `map_pool` `mean` → `mean_std` | does the readout's discarded second moment carry signal? | 79 200 s | 2 |
| `fuse_act.yaml` | `network.fuse_act` absent → `relu` | does a *nonlinearity* at the linear seam pay, with no spatial mixing? | 79 200 s | 2 |
| `injection_conv.yaml` | `network.injection_conv_layers` 0 → 1 | does one high-res cross-probe conv beat ~2× the optimizer steps? | 79 200 s | 2 |
| `poolsplit.yaml` | **three knobs** (`base_channels` 64→128, `pool_layers` 3→2, `conv_layers` 2→3) | does moving the first real conv to nside 128 pay? | 79 200 s | 2 |
| `injection_conv_k8.yaml` | `injection_conv_layers` 0→1 **and** `n_neighbors` 60→8 — i.e. **one knob off `injection_conv.yaml`**, not off the anchor | can a sparser graph buy back the step budget the high-res conv costs? | 79 200 s | 2 |
| `unet_k8.yaml` | **five knobs** (`n_neighbors` 60→8, `pool_layers` 3→1, `conv_layers` 2→4, `conv_widen` off→on, `base_channels` 64→32) | the graph-U-Net schedule — real convs at nside 256/128/64/32 — now that a sparse graph makes it affordable | 79 200 s | 2 |
| `k20.yaml` | `n_neighbors` 60→20 — **one knob, architecture held fixed** | is a smaller graph a better or worse *kernel*? The clean k test the other two k arms cannot give | 79 200 s | 2 |

**`fuse_act` and `injection_conv` are a matched pair and must be read together.** They differ by
*exactly* the spatial mixing:

```
injection_conv :  graph conv (K=5, relu) + LayerNorm      <- spatial mixing + nonlinearity
fuse_act       :                   relu  + LayerNorm      <- nonlinearity only
```

`injection_conv` alone confounds the two, because its graph conv carries the relu that the seam
otherwise lacks entirely. `fuse_act` is not parameter-matched (a graph conv also adds *K·Fin·Fout*
weights) — it isolates the nonlinearity, not the capacity.

**`lr_3e4` is the first time the optimizer has been varied anywhere in this programme.** Every round
from bench_v4 on ran one fixed recipe — Adam, 1e-4, cosine→0, 5 k warmup, `decay_alpha` 0.0, clip 1.0,
no weight decay — so the entire architecture search has happened at a single, never-checked point in
optimizer space. It is also the only arm besides `long` that **cannot** cost robustness, the network
being bit-identical to the parent. It tests the same "these runs end under-trained" hypothesis as
`long` (which measured +7.3% for 2× budget) but by bigger steps rather than more of them, at half the
compute — so read the two together:

| | `long` wins | `long` null |
|---|---|---|
| **`lr_3e4` wins** | budget-limited; combine (long at the better LR) | LR-limited — `long` is the expensive way to buy it |
| **`lr_3e4` null** | genuinely budget-limited; leave the LR alone | operating point is fine; the gap is architectural |

**Every arm is sized by wall clock, not by `n_steps`.** All four set `n_steps: auto` plus
`wall_budget_seconds` (79 200 s of training over a 2-job chain; 158 400 s over 4 for `long`) and
`job_budget_seconds: 39600` — 11 h of each 12 h allocation, leaving ~1 h for the eval tail. See
"Sizing" below for why this replaced the rate probes.

The arms therefore buy **different step counts at equal wall**, and by design: `injection_conv` and
`poolsplit` are roughly 2× the parent's step cost, so they trade steps for the knob. That coupling is
inherent to holding the wall fixed and is stated, not corrected — read the actual counts from each run's
`throughput.json` before quoting a ratio.

**`poolsplit` is deliberately three knobs and cannot be attributed to any one of them** — the trunk is
`base_channels × 2^pool_layers`, so rebalancing the stages alone would narrow it 512 → 256, and
`base_channels` must double to hold the pin. Read its header before quoting it.

**`mean_std` and `poolsplit` were added on 2026-08-11 after the round was opened**, and `mean_std`
required code: `map_pool: "mean_std"` in `nets/composite/resnet_maps_plus_cls.py`. A `fusion_width`
kwarg was added to `ResNetMultiResEncoder` in the same pass for a fourth arm, `wide_pinch`, which was
then **deferred unrun** — see below. Both are plumbed through `run_training`, `run_evaluation` and
`benchmark_resnet`; `fusion_width` defaults to the previous behaviour and is inert unset. `mean_std`
is a **fresh checkpoint lineage**.

## What the seam actually looks like — the one review finding that changed the round

`wide_pinch` (`fusion_width` 64 → 256) was written, implemented, sized, probed, and then **deferred to
`_deferred/` without being launched**, because its premise does not survive contact with the code:

- The coarse stream carries **4 channels** (`bench_v7_full` logs `coarse group nside=256 (4 ch)`), and
  `injection_proj` is a *pointwise* Dense over those 4 channels, so its output spans a ≤4-dimensional
  subspace of R⁶⁴. The concat entering `injection_fuse` has effective per-pixel dimension ≤ 64 + 4 =
  **68**, not the nominal 128 — so fusing to 64 drops **at most four directions**.
- The "8× narrower than the trunk" framing compares the seam to the *body*, two widening pools
  downstream. What it actually feeds is `HealpyPseudoConv(Fout=128)`, which consumes 4 × 64 = 256 and
  emits 128, so the seam does not even limit its immediate consumer.
- Hence any `fusion_width` above ~68 adds no cross-probe information. **If revisited, use 128, not
  256.**

**The finding that does matter: the seam is entirely linear.** `injection_fuse` is a Dense with *no
activation*, and `HealpyPseudoConv` is a `Conv1D` whose relu follows its own linear op — so
fuse → concat → pseudo-conv is one composite linear map, and concat fusion provides **zero
multiplicative cross-probe interaction**. That is consistent with `fusion: bilinear` (which added
`x*inj` explicitly) scoring 1.010, a wash.

This has a direct consequence for attribution: **`injection_conv` adds the first seam nonlinearity
*and* spatial mixing at nside 256 in a single knob**, so a win there cannot be split between them. The
cheap control is a nonlinearity at the seam with no spatial mixing (`fuse_act` below).

> **A real bug was found and fixed in the same pass, and it would have silently corrupted
> `injection_conv`.** `run_evaluation.py` was not forwarding `injection_conv_layers`,
> `injection_conv_kwargs` or `spmm_backend` to the encoder, so a run *trained* with the conv would have
> been *evaluated* without it. Verify the fix is present before launching any arm here:
> `grep -n injection_conv_layers deep_lss/apps/run_evaluation.py` must hit. This class of failure is
> silent — the architecture-drift between train and eval does not raise.

### `long` — cheap, safe, known-signed. Run it first.

2× budget bought **1.073** with the pooled readout in bench_v5/v6. If it transfers it closes ~60% of
the transformer gap with **zero architectural change**, so it cannot cost robustness. Two caveats in
the file: budget is conditional on the pooled readout (with flatten it actively *hurts* —`v3_cls`
degrades 0.545 → 0.302 → 0.270 across checkpoints), and returns diminish (transformer lensing at
250 k regressed against 150 k).

### `injection_conv` — the mechanism bet, and the one that can cost robustness

Adds one channel-preserving graph conv (+LayerNorm) at nside 256 on the **fused** stream: it extracts
non-Gaussian features before the field is pooled down *and* does content-dependent cross-probe mixing
at high resolution.

**The risk is real and specific.** Source clustering is a coupling between source density and the
lensing field — a high-resolution cross-probe effect, i.e. exactly what this knob adds capacity to
exploit. On v17, every GCNN sits at ≤0.27σ posterior shift over the 7 systematics variations while
`t2_cls` is at 0.44σ, driven by `source_clustering` at **1.12σ** where every GCNN is ≤0.25σ. There
may be a genuine FoM/robustness frontier here rather than a free lunch. **Gate this arm on Q2
posterior shift, not FoM alone** — a win that imports `t2_cls`'s source-clustering bias is not a win.

**Break-even:** wall-matched, this arm buys ~2.2× fewer optimizer steps than its parent, and budget
is worth ~+7% per 2×, so the conv must buy ~7–8% just to draw level. Landing *within the seed floor*
of the parent therefore means the conv paid for itself, and justifies a cheaper variant
(`poly_degree` 3 at the injection, or a wider fusion pinch).

### `injection_conv_k8` — the cheaper variant, run *alongside* rather than after

Added 2026-08-13, on the user's call, one knob off `injection_conv.yaml`: **`n_neighbors` 60 → 8**.

**`n_neighbors` has never been varied in this programme.** It is `60` — the *maximum* of the four
supported values (`{8, 20, 40, 60}`, enforced at `healpy_networks.py:43`) — in **all 40 deepsphere
configs in the repo**, while `deepsphere`'s own `HealpyGCNN` default is 8 and the `deep_lss` model
classes default to 20. Every round from bench_v4 on has run on the most expensive graph available, by
inheritance rather than by measurement.

**Where it costs, and where it doesn't.** `k` enters *only* the sparse `L @ x` term, `(K−1)·nnz·Fin`
with `nnz = N_pix·(k+1)`; the dense term `N_pix·K·Fin·Fout` is `k`-independent. So the share `k` can
shrink is set by how *narrow* the stream is:

| | width | spmm share of that layer |
|---|---|---|
| downsampling convs, nside 64/32 | 256–512 ch | ~9% |
| ConvNeXt body, nside 16 | 512 ch | ~6% |
| **injection conv, nside 256** | **64 ch** | **~43%** |

On the anchor, `k`=8 would therefore buy only ~6% network-wide — inside the **24% node-to-node rate
variation** already measured on combined, i.e. unmeasurable. This knob is not a general economy
measure; it is what makes a *high-resolution* conv affordable, and it is only worth spending where the
stream is narrow. At nside 256, `nnz` falls from ~4.8e7 to ~7.1e6, predicting **~1.6× on that layer**.
Treat that as an upper bound: FLOPs say the parent's conv adds ~+21% network-wide while it *measures*
2.7× slower, so the layer is bandwidth-bound and the `K`-fold Chebyshev activation stack (~16 GB at
nside 256) is untouched by `k`.

**Why it was promoted to a live arm instead of a follow-up.** The parent sized its bet on "~1.7–2.1×
fewer optimizer steps". The **measured** figure from job 3072314 is **1.27 it/s against ~3.43 for this
round's cheap arms — 2.7×**. At ~+7% per 2× of budget, `injection_conv` now carries a ~−11% step
handicap unrelated to the mechanism it tests, and per the `bench-config` rule **a null or loss from an
arm that slow is uninterpretable**. This arm exists to shrink the handicap until the mechanism is what
is being measured.

**The one way it loses on physics rather than price.** `k` and `K` both buy reach. On HEALPix `k`=8 is
the immediate neighbour ring — a one-pixel hop, 13.7′ at nside 256 — so `K`=5 reaches ~1°; at `k`=60
the hop is ~4 px and `K`=5 reaches ~4°. For clustering smoothed to fwhm ≥ 57′ a 4° kernel largely
averages over its own beam, and for lensing the non-Gaussian signal is at small scales, so ~1° is
plausibly the *better-matched* kernel and not merely the cheaper one. Falsifiable: if the 4° reach was
load-bearing, this arm loses beyond the seed floor.

Worth recording for later — cost ∝ `K·k` while reach ∝ `K·√k`, so **at fixed reach, cost ∝ √k**: a
small graph with a high polynomial order is strictly cheaper than a wide graph with a low one. This arm
does *not* exercise that (`poly_degree` stays 5; raising it would be a second knob). Note `poly_degree:
8` measured 0.970 (loses) in bench_v4 — but at `k`=60, where it bought cost *without* shrinking the
graph, so that result is **not** evidence against the small-`k`/high-`K` direction.

**Robustness gate inherited, and it may cut the other way.** A tighter kernel has *less* reach to
exploit source clustering — a ~arcmin-scale cross-probe coupling — so `k`=8 could be both cheaper and
*more* robust than its parent. A prediction, not a result.

**Silent lineage trap, with no shape guard.** `n_neighbors` changes the Laplacian but **no weight
shape** — the Chebyshev kernel is `[K·Fin, Fout]`, the depthwise kernel `[C, K]`, neither depending on
`k`. A `k`=60 checkpoint restores into a `k`=8 network with **no error** and meaningless filters, and
the restore uses `expect_partial()`. Never restore across this knob; this arm is its own lineage.

### `unet_k8` — and the measurement that reopened the U-net

Added 2026-08-13. **Deliberately five knobs and unattributable, like `poolsplit`** — read its header
before quoting it. It puts a real graph conv at *every* remaining downsampling level (nside
256/128/64/32) on the modern recipe, holding the trunk at the pinned 512 via `base_channels` 32
(`32 × 2⁴`, verified against `resnet.py`'s build loop, not assumed). Secondary difference, documented
rather than hidden: the fusion seam narrows 64 → 32, because `split_Fin` **is** `base_channels`.

**Why a re-run of a measured-dead arm is legitimate here.** bench_v4's `graph_unet_256` scored 0.866,
but it ran at `base_channels` 16 with a 256 body and its nside-256 conv at `Fin` 16, on the **flatten**
readout (mean-pool arrived in bench_v5 at +32.1%), and it is **eight** knobs from the current recipe.
It measured *an old thin recipe that happens to have high-res convs*, not high-res convs. Its 1.64×
rate deficit was also incurred at `n_neighbors: 60`.

**The measurement that changed the round.** Probe 3072984 on `injection_conv_k8.yaml` — verified
`injection_conv_layers=1`, 17 post-fusion layers, `n_neighbors: 8` in the snapshot — gave a second-half
window of **4.18 it/s (239 ms/step)** against `injection_conv`'s **measured production 1.27 it/s
(787 ms/step)**, job 3072314. Read the probe ~9% optimistic (1500 steps never crosses
`checkpoint_every: 5000`), so ~260 ms. Even so:

| arm | graph | high-res conv | ms/step |
|---|---|---|---|
| `long` / `lr_3e4` / `mean_std` | k=60 | none | 292 |
| `fuse_act` | k=60 | none (relu only) | 326 |
| `poolsplit` | k=60 | nside 128, `Fout` 512 | 820 |
| `injection_conv` | k=60 | nside 256, `Fout` 64 | 787 |
| **`injection_conv_k8`** | **k=8** | nside 256, `Fout` 64 | **239** (probe) |

**k=8 makes the arm *with* the nside-256 conv faster than the controls that have none.** The premise
this round was designed around — "a high-resolution conv costs ~2× the optimizer steps" — is
substantially an artifact of running the maximum-neighbour graph by inheritance.

**Correction to the spmm-share reasoning above, kept because the error is instructive.** The FLOP-share
table predicted ~1.6× for this knob and ~6% network-wide; both were wrong. 787 → 239 ms cannot come
from cheapening the injection conv alone, so k=8 is also saving heavily on the nside-64/32 convs and
the ConvNeXt body. **Share-of-FLOPs badly understates spmm's share of TIME**, because the sparse matmul
runs at a small fraction of peak while the GEMMs run near it. The formula's *ordering* (narrow `Fout` →
k matters more) is probably still right; its magnitudes are not — which is also why `poolsplit`'s
`Fout` 512 conv is *expected* to gain little from k=8, but that expectation now deserves a probe rather
than an assertion.

**Consequence for the round's design.** `injection_conv` (k=60) is now best read as the **cost
reference** that makes the k knob attributable, rather than as the mechanism arm — its own FoM carries
a ~2.7× step handicap, and per the sizing rule a null or loss from it is uninterpretable. The same
applies to `poolsplit`.

### The comparison axis: fixed compute, steps floating — decided by the user, 2026-08-13

**These arms are to be ranked at a roughly fixed compute budget, and the differing step counts are
accepted as part of the answer, not as a confound to correct.** `wall_budget_seconds` fixes *training
seconds* and lets steps float, which is exactly this comparison, so no re-sizing or step-matched rerun
is wanted. The question being answered is **"best FoM per unit of compute"**, not "best FoM per
optimizer step" — so a cheap architecture that wins by training longer in the same wall has genuinely
won. Consequences to keep straight when reading the ratios:

- **`long` sits outside the fixed-compute set by design** (158.4 ks vs 79.2 ks). It is the arm that
  *measures* budget, so it is not comparable on this axis to the other seven — read it only against the
  anchor and against `lr_3e4`.
- **The anchor is not exactly wall-matched either.** `bench_v7_full` as run trained **23.81 h ≈ 85.7 ks**
  (bench_v7 RESULTS), against this round's 79.2 ks — about **8% more training wall**, worth ~+0.8% of
  FoM at the measured ~+7%-per-2× rate. That is inside the 1.5% seed floor, so it does not change any
  ranking, but anchor-relative ratios carry a small bias *against* every bench_v8 arm.
- **What fixed compute does NOT settle** is *why* an arm won. `injection_conv_k8` gets ~2.65× the steps
  of its k=60 parent, and `unet_k8` more still, so a win there is "cheaper architecture, same wall" and
  cannot be attributed to the kernel or the schedule on its own. That is accepted here; a mechanism
  question would need its own step-matched arm, which is deliberately not in this round.
- **Robustness is a separate axis entirely** and is unaffected by any of this. See the gates on
  `injection_conv`, `injection_conv_k8` and `unet_k8` — still unmeasured for every arm on v18.

### Measured step times — the whole round, compile-free second-half windows

Production runs unless noted. This is the table to size any follow-up from; do not re-derive it from FLOPs.

| arm | graph | first real conv | ms/step | it/s | vs control |
|---|---|---|---|---|---|
| `long` / `lr_3e4` / `mean_std` (control) | k=60 | nside 64 | 300 | 3.33 | — |
| `fuse_act` | k=60 | nside 64 (+relu at seam) | 335 | 2.99 | 1.12× |
| `injection_conv` | k=60 | nside 256, `Fout` 64 | 660 | 1.51 | 2.20× |
| `poolsplit` | k=60 | nside 128, `Fout` 512 | 763 | 1.31 | 2.54× |
| `injection_conv_k8` | k=8 | nside 256, `Fout` 64 | **249** | 4.02 | **0.83×** |
| `unet_k8` | k=8 | nside 256, `Fout` 64 (4 conv levels) | **220** | 4.55 | **0.73×** |

The two k=8 arms are the **cheapest in the round despite carrying the most high-resolution
convolution** — but note `unet_k8`'s 220 ms is substantially its half-width schedule, not the graph
alone (see its header's confound block).

`unet_k8`'s figure is two independent measurements in agreement — probe 3073018 gave 221 ms and
production job 3073034 gave 220 ms on a clean post-step-1000 window. **Both are provisional**: they were
taken ~7 min into training, and the bench_v7 logs show a rate can ramp for *hours* (the transformer /
clustering arm went 5.8 → 8.8 it/s over six hours). Confirm from `throughput.json` before quoting.
Two cautions on method, both learned the hard way here:

- **An early window that starts before ~step 500 straddles the post-compile dataloader ramp** and
  understates the rate badly — the same U-net run measured 503→1000 gave 304 ms, a 1.38× error against
  the steady-state 220. Start the window past step ~500.
- **`throughput.json`'s `sustained_it_per_s` is NOT sustained on a probe shorter than 2000 steps.**
  `bin_steps` is 2000, so a 1500-step probe leaves `n_bins: 0` and the field silently falls back to the
  *cumulative* average — compile and ramp included — under a name that invites trust. Both probes here
  hit it: 3072984 reported 2.63 it/s against a true 4.02, and 3073018 reported 2.99 against a true 4.55.
  **Sizing off those would have undersized by ~35%.** Check `n_bins` before reading the field, or run
  the probe past 2000 steps (`PROBE_STEPS` default is 6000 for exactly this reason — both probes here
  were deliberately shortened to fit the 30-min debug slot, which is what triggered it).

## Deliberately NOT in this round

- **`w128` / trunk 1024** — scrapped by the user. (The transformer's trunk is 1024 vs the GCNN's 512,
  so the asymmetry is real, but it is not being tested here.)
- **`wide_pinch` (`fusion_width` 64 → 256)** — in `_deferred/`, never run, premise disproved (above).
  Its rate probe 3057100 was cancelled. The `fusion_width` code stays and is inert unset.
- ~~`fuse_act`~~ — **implemented and promoted to a full arm** (see the table above), since without it a
  positive `injection_conv` result is unattributable.
- **`adamw_wd05`** — the other optimizer lever, and arguably the better-motivated one: the recipe
  already imports the ConvNeXt ingredients (ConvNeXt blocks, LayerScale 1e-6, DropPath 0.1) that were
  co-designed with AdamW at wd ≈ 0.05, while the optimizer stayed plain Adam at wd = 0. `optimizer:
  adamw` **is** supported (`deep_lss/utils/optimization.py`) and takes `weight_decay` through
  `optimizer_kwargs`. Held back only because LR is the higher-variance knob and this round can afford
  one optimizer arm, not two. Run it after `lr_3e4` reports.
- **Giving the coarse stream learned spatial features before fusion** — the deeper asymmetry vs the
  transformer: clustering enters the seam as 4 *raw smoothed* channels while lensing has already been
  through a conv stage. Arguably the real fusion gap; `injection_conv` only addresses it after the fact.
- **`mean_std_pinned`** (`mean_std` + `map_feature_dim: 512`) — the follow-up that separates the second
  moment from the doubled readout width, if `mean_std` wins. Not run now: adding a projection the
  parent does not have confounds it in a different way, so it is only informative against a `mean_std`
  result that already exists.
- **Anti-aliased downsampling** — the three pure pooling stages stride with *no* low-pass, so they
  alias exactly the small-scale power that carries the non-Gaussian signal (the BlurPool insight).
  Not implemented; would need a new layer in `deepsphere`. Cheapest untried idea in the whole
  downsampling phase, but it is code, not config.

### Measured dead — do not re-try

Re-scored **paired on 1000 mocks** from the bench_v4 runs, whose original 16-mock ranking was invalid:

| lever | ratio vs `bench_v4_default` | |
|---|---|---|
| `fusion: bilinear` | 1.010 [0.999, 1.023] | wash |
| `poly_degree: 8` | 0.970 | loses |
| `less_cheby` | 0.973 | loses |
| `bernstein` | 0.928 | loses |
| `graph_unet_256` | 0.866 | loses badly |
| `deep_trunk` | 0.980 | wash/loses |
| `wide_shallow` vs `w64` | 0.985 | wash — trunk depth 5→2 is neutral; that arm's gain was width |

`conv_widen` / `graph_unet` were already dead. The conv **basis and degree** in the downsampling
stages are settled: `poly_degree: 5`, `conv_type: cheby`.

## Launching

Use the **`submit` skill** — do not write a new script. `maps/training.sh` and
`maps/training_chainer.sh` already parameterize everything.

### Sizing — wall clock, and why the rate probes were abandoned

**All four arms are launchable. There is no rate-probe gate.** Every arm sets `n_steps: auto` +
`wall_budget_seconds` + `job_budget_seconds: 39600`, so it trains for a fixed number of *training
seconds* with the cosine parameterised by wall clock — annealing to zero exactly when the budget is
spent, at whatever rate the run achieves. `n_steps` becomes an **output**, written to `throughput.json`
next to the model.

This round was originally built the old way, with hand-sized `n_steps` and four blocking rate probes.
That was abandoned on 2026-08-11 because `f98facf` had already replaced it, and its reasoning applies
directly here:

- Sizing `n_steps` by hand is **asymmetric**: oversize and the cosine never anneals and
  `run_evaluation.py` / `run_inference.py` never fire, so a whole allocation yields nothing scorable —
  which has happened four times in this programme. Undersize and the leftover wall is wasted; bench_v7
  lost 15–37 k steps per arm that way.
- **Probing the rate then fixing `n_steps` does not fix it.** In bench_v7's own logs the
  transformer/clustering arm ramped 5.8 → 8.8 it/s over six hours, so no window in its first 20 k steps
  described the run — every one under-predicted by 21–25% — and a standalone probe of the *same* config
  did not ramp at all. The rate is not always a property of the config.
- Node-to-node variation is 24% on identical work (3.24 vs 2.62 it/s for this very recipe), an order of
  magnitude above the 1.5% seed floor. A fixed step count has to be sized for the worst node it might
  draw; a wall budget absorbs it, along with contention and restarts.

The probes (3056942, 3057099, 3057100, 3057101) were cancelled. Two of their rates are recorded in the
config headers as *reference only* — `injection_conv` ~1.56 it/s, `poolsplit` ~1.35 it/s — and must not
be converted back into an `n_steps`.

**The cost, which is real and accepted:** this is equal-**wall** benchmarking, not equal-**sample**. Two
runs of one config no longer take the same number of steps. A controlled sample-budget comparison would
need a fixed `n_steps`; this round is explicitly about the best use of a fixed compute budget.

`poolsplit` is the one arm that can hit the **cuSPARSE ceiling** (`output.shape[1] * nnz(a) > 2^31`),
being the widest here. If it dies with that error, the fix is a smaller batch — and note the budget makes
that safe to change mid-round, since a slower config now simply takes fewer steps rather than needing to
be re-sized.

**The chain length is `MAX_RUNS`, and it defaults to 2.** There is no `N_STEPS_TOTAL` env var anywhere
in `training.sh` or `training_chainer.sh` — an earlier version of this snippet passed one, which would
have been silently ignored while the chain ran at the default length. For `long` that would have
submitted a **2-job** chain against a 158.4 ks cosine, i.e. exactly the never-annealed partial chain this
round warns is worthless. **`MAX_RUNS` must match `wall_budget_seconds / job_budget_seconds`** — 4 for
`long`, 2 (the default) for the rest. Budgets are overridable at launch via
`TRAIN_EXTRA="--wall_budget_seconds=... --job_budget_seconds=..."`, which also works on the restore path
and is the only way to resize a live chain.

**`training_chainer.sh` is not executable** (mode 644, and it is not worth a mode change in the diff),
so invoke it as `bash <script>`. Running it directly fails with `Permission denied` — harmlessly, since
nothing is submitted, but it fails once per arm if you loop.

This is the exact command the round was launched with:

```bash
cd /users/athomsen/dlss/repos/y3-deep-lss
export VERSION=v18 SUBVERSION=default PROBE=combined ARCH=deepsphere
B=$PWD/configs/deepsphere/combined/bench_v8
C=submissions/clariden/maps/training_chainer.sh

# long: 4 jobs. MAX_RUNS must equal wall_budget_seconds / job_budget_seconds.
MAX_RUNS=4 NET_CONFIG=$B/long.yaml MODEL_DIR=bench_v8_long bash $C

# the other five: 2-job chains (MAX_RUNS defaults to 2). No sizing step needed.
for arm in lr_3e4 mean_std fuse_act injection_conv poolsplit; do
    NET_CONFIG=$B/$arm.yaml MODEL_DIR=bench_v8_$arm bash $C
done
```

`PROBE=combined` is the **plain** probe config, not `combined_nla`: v18/default is `extended_nla: True`.
`training.sh`'s own header is the authority here — the `_nla` variants are for v17, and the `submit`
skill's "v17+ needs `_nla`" line is stale. Getting it wrong does not raise; it silently marginalizes `ds`.

v18/default is `extended_nla: True`, so the **plain** `configs/probes/combined.yaml` is correct — not
`combined_nla`. Getting that wrong does not raise; it silently marginalizes `ds`.

**To resize a chain in flight, edit `<dir_model>/configs.yaml` or pass `TRAIN_EXTRA`, not this file.** A
restored job reads its own snapshot (`run_training.py:359`), so editing the repo YAML does nothing. The
CLI overrides (`--wall_budget_seconds`, `--job_budget_seconds`, `--n_steps`) are applied on the restore
path for exactly this reason. `afterany` continues past a failure, so check each handoff.

**Verify the handoffs.** `budget_progress.json` carries `consumed_seconds` and `warmup_end_seconds`
across a job boundary so the chain follows one continuous cosine; job N+1 logs the seconds already spent
by earlier jobs. If that number is 0 in a later job, the chain restarted its curve and the run is not
what it claims to be. With `long` there are three such handoffs instead of one.

**`long`'s cosine spans the whole 158.4 ks budget**, so a chain that dies after job 3 is not a
"three-quarter-length run" — it is a run whose LR never annealed, and the gain lives in that tail. It is
not comparable to anything.

## Scoring

Rank with the **`compare-runs`** skill, paired FoM(Ωm, S8), 1000 mocks, 1.5% seed floor. Reference
`bench_v7_full` for the architectural question and `bench_v7_transformer` for the actual target.

**Record each arm's realised step count from its `throughput.json`** when reporting. Under a wall budget
the step count is an output and differs between arms by design, so it is part of the result, not a
nuisance — an arm that wins on fewer steps is a stronger result than the same win on more.

**Then the robustness gate, which is the point of the round** — Q2 posterior shift on the 7
systematics mocks, per arm, against `bench_v7_transformer` and `bench_v7_full`. Note this has **never
been measured for any bench_v7 arm on v18**: the ≤0.27σ-vs-0.44σ numbers quoted above are
v17/bench_v6, and v18 deliberately moved source clustering *into* the shape noise, which changes what
the test means. The premise motivating this whole round is therefore itself unverified on v18 —
measuring it on the existing six bench_v7 arms is cheap and gates how much the `injection_conv`
result is worth.

**FoM on misspecified data is unsigned** — judge robustness by posterior bias, never by DES FoM.

## Status

- 2026-08-11 round created with `long` + `injection_conv`. `injection_conv` one-knob-verified by
  parsed-YAML diff and confirmed to build with the conv active (encoder logs `injection_conv_layers=1`,
  17 post-fusion layers vs the parent's 15). Both were hand-sized with fixed `n_steps` at this point.
- 2026-08-11 `mean_std`, `wide_pinch` and `poolsplit` added; `map_pool: mean_std` and `fusion_width`
  implemented and plumbed; the `run_evaluation.py` multires-kwarg bug fixed (see above).
- 2026-08-11 **round reviewed.** Completeness re-checked by parsed-YAML diff against the parent (no
  missing keys in any arm; knob counts as documented). `poolsplit`'s width arithmetic verified
  *statically* against `resnet.py`'s build loop, not just from the encoder log: pool stages widen after
  each append, so 128 → (256) → 512 holds the trunk at 512, five downsampling stages keep the body at
  nside 16, the seam still lands exactly on 256 with `split_Fin` = 128, and the first real graph conv
  does move to nside 128 — every claim in that header stands. `run_evaluation`'s multires kwargs
  compared arg-by-arg against `run_training`: identical bar `max_batch_size`, which legitimately
  differs. **`wide_pinch` deferred unrun** and its probe cancelled (premise disproved, above).
- 2026-08-11 **all four arms switched to the wall-clock budget** (`n_steps: auto` +
  `wall_budget_seconds` + `job_budget_seconds`), and the four rate probes cancelled. The round had been
  built with hand-sized `n_steps` and a probe gate that `f98facf` — committed four days earlier, on
  2026-08-07 — already exists to remove. Reasoning in "Sizing" above; the short version is that the
  bench_v7 logs show the rate is not always a property of the config, so no probe could have sized these
  correctly. Verified before switching: `optimization.get_optimizer` takes the
  `scheduler == "cosine" and budget is not None` branch first, so `n_steps: auto` never reaches the
  `n_steps - warmup_steps` arithmetic at `optimization.py:71`, and all four arms use `scheduler: cosine`.
  Observed probe rates kept in the headers as reference only: `injection_conv` ~1.56 it/s,
  `poolsplit` ~1.35 it/s.
- 2026-08-11 **`fuse_act` added** as the control that makes `injection_conv` attributable, and
  **`lr_3e4` added** as the programme's first optimizer arm. `fuse_act` needed code: a `fuse_act` kwarg
  on `ResNetMultiResEncoder` (activation + LayerNorm on `injection_fuse`'s output, applied in `call()`
  because a bare activation is not a layer `HealpyGCNN`'s build loop can walk), plumbed through all six
  construction sites. `lr_3e4` is config-only. The `fusion_width` docstring was corrected in the same
  pass — it still asserted the disproved "8× pinch".
- 2026-08-11 smoke test **3072286** (debug, 150 steps) on `fuse_act.yaml`, because the `fuse_act` edit
  changed `ResNetMultiResEncoder`'s **shared** constructor — a mistake there would break all six arms,
  not one.
- 2026-08-11 **ROUND LAUNCHED — all six arms, 14 jobs.** Smoke test 3072286 passed first: the encoder
  built with `fuse_act=relu` and still logged *"fused to 64 (15 layers after the fusion)"*, byte-identical
  in structure to the parent (so the activation is a true control, not a structural change), training at
  3.13 it/s — inside the parent's 2.62–3.24 range. That also cleared the shared-constructor edit for all
  six arms.

  | arm | jobs | chain |
  |---|---|---|
  | `long` | 3072303, 3072304, 3072305, 3072306 | 4 × 12 h |
  | `lr_3e4` | 3072307, 3072308 | 2 × 12 h |
  | `mean_std` | 3072310, 3072311 | 2 × 12 h |
  | `fuse_act` | 3072312, 3072313 | 2 × 12 h |
  | `injection_conv` | 3072314, 3072315 | 2 × 12 h |
  | `poolsplit` | 3072316, 3072317 | 2 × 12 h |

  Dependencies verified after submission: each follower carries `afterany:<predecessor>`, and `long` is a
  proper 4-link chain (304→303, 305→304, 306→305). Chain heads correctly carry none — note `squeue`
  prints `(null)` rather than an empty string for "no dependency", which will fool a naive check.

- !! **THE CODE IS UNCOMMITTED AND THE RUNS READ IT LIVE.** !! Configs are snapshotted into each run
  dir at first launch, but the Python is re-imported from the working tree **every time a job starts** —
  and the follower jobs start ~11 h apart. Editing `resnet_multires.py`, `run_training.py` or
  `run_evaluation.py` before this round finishes would silently change the architecture mid-chain, and
  the `expect_partial()` restore would not raise. Leave those three files alone until the chains are
  done, or commit them now.
- **Everything in this round is uncommitted**, including the code changes — deliberately, per the user.
  Note `injection_conv` depends on the uncommitted `run_evaluation.py` fix.
- 2026-08-13 **`n_neighbors` opened as a lever for the first time in the programme, and two arms added
  and launched** — both **config-only**, so neither touches the uncommitted-code hazard above
  (`n_neighbors` is already plumbed through `run_training` *and* `run_evaluation`).

  | arm | jobs | chain |
  |---|---|---|
  | `injection_conv_k8` | 3072994, 3072995 | 2 × 12 h |
  | `unet_k8` | 3073034, 3073035 | 2 × 12 h |

  `injection_conv_k8` one-knob-verified against its parent by parsed-YAML diff (`n_neighbors` the only
  value that differs) and smoke-tested first: probe **3072984** (debug, 1500 steps) confirmed the conv
  still active with the sparse graph and measured **4.18 it/s / 239 ms/step vs the parent's production
  1.27 / 787** — see the section above, including the correction to the FLOP-share estimate that
  under-predicted this by 2×. `unet_k8` was launched on the user's call **without waiting for its rate
  probe 3073018** (queued behind 3072984; debug QOS runs one job at a time), so its step time is read
  from its own run instead.
- 2026-08-13 **`k20` added and launched** (jobs **3073106**, 3073107; 2 × 12 h). One knob off the
  anchor, `n_neighbors` 60 → 20, architecture untouched — verified by parsed-YAML diff (only
  `n_neighbors` and the standard budget block differ; three further hits are comment text with no value
  change). **This is the clean k test the other two k arms cannot provide**, since both of those change
  the graph *and* add high-resolution convolution. With it the k axis has three points on one
  architecture: k=60 (`bench_v7_full` as run), k=20 (this arm), and k=8 — which is **not** tested on the
  plain anchor and is the obvious next run if the axis turns out to matter. Config-only, so it does not
  touch the uncommitted-code hazard.
- **The dates on the four entries above this one say 2026-08-11 but the round was launched 2026-08-13**
  (jobs 3072303–3072317, and smoke 3072286 ended 09:34 on the 13th). Left as-is rather than guessed at —
  the entries were presumably authored on the 11th and the launch slipped two days. Worth a pass by the
  user, since "2026-08-11" in a status log reads as when it *happened*.
