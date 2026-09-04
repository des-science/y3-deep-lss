# bench_v12 — the SHARED-CORE round

**Status: 14 jobs** (`staged` added 2026-09-02).

**Host-RAM OOM, resolved 2026-09-02.** Four GCNN jobs (3242160, 3242161, 3245093, 3245095) were
OOM-killed 2-8 h in, at 444.4-445.7 GiB against the 450 GiB cgroup. It was **not** a config
difference: every GCNN job of this round ran the identical dataloader block, and two runs of the
same architecture landed 41 GiB apart (432.0 vs 391.1 GiB). It was `tf.data`'s autotuner, whose
`ram_budget` defaults to half the NODE's memory -- and on a GH200 that figure is the 870 GB UNIFIED
Grace+HBM pool, so it budgeted ~425 GiB against an allocation of 450. msfm `f6bdca9` now caps it
from `SLURM_MEM_PER_NODE`; since msfm is editable-installed into `tf_env`, that fix reaches
**restored** jobs too, whose frozen `configs.yaml` still says `n_workers: null`.

**Status: QUEUED, 12 jobs** (2026-08-31). Twelve submitted at 11:48 (3242160-3242173); none had
started when the round was revised at 17:0x, so nothing was lost.

| arm | jobs | note |
|---|---|---|
| `classic_nodrop` | 3242160 / 3242161 | original submission, kept |
| `transformer` | 3242167 / 3242168 | original submission, kept |
| `transformer_nodrop` | 3242169 / 3242170 | original submission, kept |
| `transformer_nodrop_droppath` | 3242171 / 3242173 | original submission, kept |
| `staged_nodrop` | **3245093 / 3245094** | replaces `convnext_nodrop` (3242162/3, cancelled) |
| `staged_nodrop_droppath` | **3245095 / 3245096** | replaces `convnext_nodrop_droppath` (3242164/5) |

Only the two ConvNeXt chains were cancelled and resubmitted; the other eight keep the priority
they had accrued since 11:48. The two new chains therefore start later than the rest, which costs
latency but confounds nothing -- every arm is `n_steps: auto` on its own 79 200 s budget.

**The revision:** the two ConvNeXt GCNN arms now carry the **staged all-ConvNeXt layout** that was
drafted as its own round, `bench_v13`. That round is gone -- folded in here rather than run after,
because a queue that had not moved made it free to test the layout we actually intend to ship
instead of the one we intend to replace. See "The bench_v13 fold" below for what it costs.
`bench_v13/README.md` is now `STAGED_LAYOUT.md` in this directory.

This file carries the shared rationale for **both architectures and all three probes**.
`lensing/bench_v12/` and `clustering/bench_v12/` point here, as do the three
`configs/maps/dev/transformer/<probe>/bench_v12/` directories.

## What the round is for

Pick **one default per architecture** — a GCNN and a vision transformer — that

1. **share as many components as the code allows**, so `paper_2_tex/figures/networks.tex` can draw
   them as one story rather than two unrelated networks;
2. are **not more complicated than the evidence supports**. Most of the differences this programme
   has measured are inside the 0.049 seed floor, so the standing rule applies: **a tie goes to the
   simpler network**;
3. still post a **decent CosmoGrid FoM**.

Two questions, and nothing else:

- **A. The classic GCNN vs the all-ConvNeXt network.** Originally one knob (the block); since the
  bench_v13 fold it is the **package** — ConvNeXt blocks *and* the staged layout — because that is
  the network the paper would actually ship. The one-knob block contrast is not lost, it is simply
  already answered: bench_v11 measured the block alone at **~+0.7% once its step confound is paid
  back**, a wash. What is open is whether the full package holds up, not which half of it carries.
- **B. Is the regularization in the right place?** This pipeline inherited a **bare body with
  dropout in the regression head**. Modern practice is the exact reverse: a bare head, and
  stochastic depth in the body. Both halves are tested — head dropout off, and DropPath on.
  **Weight decay is not tested**: AdamW has not paid off in previous rounds here, and adding a
  third regularizer to a round that expects washes only widens the multiple-comparison problem.

## The shared core — fixed in every arm, both architectures

| | |
|---|---|
| **readout** | mean pool over the **448 footprint pixels/tokens at nside 16** |
| **head** | `get_regression_head(...)` → `LayerNorm → [Dropout] → Dense(10)` |
| **optimizer** | Adam 1e-4, cosine → 0, 5 k warmup, global-norm clip 1.0 |
| **batch** | 16 |
| **budget** | `n_steps: auto` + `wall_budget_seconds` — 39 600 s single probe, 79 200 s combined |
| **inputs** | smoothing + `input_norm` per probe; clustering enters at nside 256, never upsampled |

**The readout is not a free choice, and that is the point.** The transformer's map encoder
mean-pools its tokens itself and exposes **no `map_pool` option at all**
([`transformer_summary.py`](../../../../deep_lss/nets/composite/transformer_summary.py)), so
`mean` is the **only** setting at which the two architectures share a readout.

That is why **bench_v11's `mean_std` is deliberately not used here**, despite winning on combined.
It is probe-dependent — **+6.9% combined, a wash on lensing, a resolved −6.7% LOSS on clustering** —
and it has no transformer counterpart, so adopting it would buy a flagship-only gain at the cost of
the shared figure and of "works across the board". If the round ends with a strong case for it,
that is a deliberate, separately-argued departure, not a default.

**The head is literally the same function on both paths.** `ResNetLayers` and
`TransformerSummaryNetwork` both call `get_regression_head`, so question B is **one** question with
one answer, not two analogous ones.

**What deliberately stays different, and stays out of the round:** trunk width (GCNN 512,
transformer 1024 — pinned *within* an architecture across probes, never *across* architectures, per
bench_v7), `map_feature_dim` (null on the GCNN, 512 on the transformer, because only the
transformer's pooled feature needs crushing to match the Cls embedding), and bf16 + XLA on the
transformer, which is a memory decision. The figure already draws all three explicitly.

## Two architecture changes this round carries (2026-08-31)

Neither is a knob **of** the round — every arm carries both, on every probe — but both change what
the transformer builds, so they are recorded here rather than in a run log.

**1. The maps+Cls fusion is now the same shape on both architectures.** `map_feature_dim: 512 → null`
in every transformer config: the pooled map feature goes straight to `map_norm → concat → head`,
with no projection `Dense`, exactly as on the GCNN side. Together with the pre-pool `LayerNorm`
removal (see the transformer READMEs), this makes the **whole readout** — pool, norm, fuse, head —
structurally identical across the two networks, for maps-only *and* maps+Cls.

The concat is consequently **unbalanced**: 1024 (map) + 512 (Cls) on the transformer against the
GCNN's 512 + 512. That was a deliberate call. Rebalancing by widening the Cls embedding to 1024 was
considered and declined — 512 is the validated two-point width, and there is no evidence a 1:1
concat matters. Deleting the `Dense` also **removes ~0.52 M parameters** rather than adding any.

**2. Clustering's transformer trunk now matches the other probes.** `base_embed_dim: 32 → 64`,
clustering only. Trunk = `base_embed_dim × 2^num_nested_levels`, and clustering runs at nside 256 so
it gets **four** levels where lensing and combined get five — 32 ended at 512, half the others. 64
gives `[64,128,256,512,1024]` → trunk 1024.

This is the transformer's version of the GCNN's own fix on this probe (`base_channels: 128` with
`pool_layers: 2`, which `deepsphere/prod` adopted and `transformer/prod` never did). It was
benchmarked in v17 `bench_t2` and then dropped on the floor.

| | |
|---|---|
| **Q1** (`embed64` vs `b20`, 1000 paired mocks) | **1.010** [1.003, 1.023] — a **wash** at the 0.049 floor |
| **Q3** coverage | in-cohort both ways; paired HPD **+0.0031**, CI spans 0 |
| **Cost** | **~5%** throughput, 7.34 → 7.00 it/s (batch 20 vs 16, so approximate) |

Adopted for **cross-probe comparability, not for FoM**. The ~5% is the point worth remembering: the
FLOP count says ~4×, but clustering's step is smoothing-dominated (21 kernel applications at nside
256 against ~4 at 512), so quadrupling the body moves about a fifth of the step. That reasoning does
**not** transfer to lensing or combined, where the body does dominate.

**If any arm of this round ships, `transformer/prod/<probe>/maps+cls.yaml` must be updated to match
before it is used as a default.**

## The arms

Optimizer is **Adam 1e-4 in every arm** — see question B on why weight decay is not tested.

### GCNN — `configs/maps/dev/deepsphere/<probe>/bench_v12/`

| file | layout | block | head dropout | body DropPath | knob vs |
|---|---|---|---|---|---|
| `classic.yaml` | legacy | classic | 0.1 | 0.0 | — (the base) |
| `convnext.yaml` | legacy | **convnext** | 0.1 | 0.0 | `classic` |
| `classic_nodrop.yaml` | legacy | classic | **null** | 0.0 | `classic` |
| `staged_nodrop.yaml` | **staged** | **convnext** | **null** | 0.0 | `classic_nodrop` (a package, 9 keys) |
| `staged.yaml` | **staged** | **convnext** | **0.1** | 0.0 | `staged_nodrop` (1 key) |
| `staged_nodrop_droppath.yaml` | **staged** | **convnext** | **null** | **0.1** | `staged_nodrop` (1 key) |

The two `_deferred/` files, `convnext_nodrop.yaml` and `convnext_nodrop_droppath.yaml`, are the
legacy-layout versions of the last two rows. They were queued and cancelled unrun; they are kept
because they are what closes the 2 × 2 if the package ever needs taking apart.

**What the 2 × 2 became.** `classic`/`convnext`/`classic_nodrop`/`convnext_nodrop` was a clean
2 × 2 in (block, head dropout). Folding the layout in replaces its fourth cell with a two-knob
package, so the block×dropout interaction is no longer readable from this round alone. Accepted:
three of the four cells are unchanged, the dropout edge still reads on the classic block, and the
block edge is bench_v11's already-measured wash.

**`staged.yaml` was added on 2026-09-02, after the other five arms had launched.** It is
`staged_nodrop` with head dropout back on -- the cell the fold destroyed. Two reasons it is worth a
2 x 12 h chain rather than being inferred:

1. **B1 would otherwise read only on the classic block.** The transformer has already answered it in
   the opposite direction to the round's expectation -- `transformer_nodrop` LOST 7.3% paired FoM at
   matched wall, well outside the 0.049 floor, with `vali_total` agreeing independently (-6.51 vs
   -10.77). So "expect washes on the regularization arms" is already falsified once. Whether that
   transfers to a ConvNeXt body is exactly the kind of block-dependence DropPath's sign flip
   (+5.0% ConvNeXt, 0.946 classic) says not to assume.
2. **It gives question A a second, independent reading.** `staged` vs `bench_v11_simple` is the
   all-ConvNeXt package against the classic GCNN at head dropout ON, mirroring
   `staged_nodrop` vs `classic_nodrop` at OFF. Two readings that agree are a far better basis for
   the paper's network than one contrast at one regularization setting.

It carries **two non-knob deviations**, both stated on the lines themselves. `n_workers` is 128
rather than Null in the training and eval loaders: left Null, the interleave/parse/augment stages
all go to `tf.data.AUTOTUNE`, whose `ram_budget` defaults to half the NODE's memory -- half a
GH200's 870 GB UNIFIED pool against a 450 GB cgroup. That killed four jobs of this round. It is
host-side I/O parallelism and changes no sample, gradient or step. `examples_shuffle_buffer` is
deliberately held at the parent's 256 rather than prod's 1024, and `n_batches` at 50 rather than
prod's 100, because both of those WOULD be real knobs against the partner it is scored on.

`staged_nodrop → staged_nodrop_droppath` is untouched by the fold and remains the clean one-knob
pair that tests question B's second half: **is a bare head plus stochastic depth better than a
regularized head plus a bare body?**

`staged_nodrop_droppath` is the round's **modern-practice candidate**. `drop_path_rate: 0.1` is the
value used throughout bench_v8 and bench_v11 and is **not** being tuned here. In the staged layout
it now reaches **all 7 blocks** rather than 5, because there are no bare convolutions left outside
a residual branch — a slightly stronger dose of the same knob, worth noting when reading its margin.

**There is no `classic_droppath` arm.** DropPath measured **+5.0% on ConvNeXt and 0.946 — a loss —
on the classic block**, so that arm would spend a 12 h slot re-measuring a known negative.

### Transformer — `configs/maps/dev/transformer/<probe>/bench_v12/`

| file | head dropout | body DropPath | one knob vs |
|---|---|---|---|
| `transformer.yaml` | 0.1 | 0.0 | — (prod + the two changes above) |
| `transformer_nodrop.yaml` | **null** | 0.0 | `transformer` |
| `transformer_nodrop_droppath.yaml` | **null** | **0.1** | `transformer_nodrop` |

The arm sets now **mirror each other on question B**: empty the head, then fill the body, on both
architectures. `DropPath` was added to the nested transformer on **2026-08-31** by importing the
same `deepsphere.gnn_layers.DropPath` the GCNN's ConvNeXt block already used, applying it at the
same place (`x + DropPath(branch)`, after `LayerScale`) at a **constant rate across depth** rather
than timm's linear ramp — because that is what `gcnn/resnet.py` does. So the knob means the same
thing on both sides and a cross-architecture read of it is legitimate.

**Do not assume the GCNN's DropPath result transfers.** Its sign was already block-dependent
within one architecture (+5.0% on ConvNeXt, **0.946 — a loss — on classic**). A sign that flips
between two blocks of one network is not a sign to assume across two networks. The transformer
arm is a first measurement.

**The transformer DropPath path has never been run.** `DropPath` is variable-free so it cannot
break a checkpoint, but the plumbing (network kwarg → local stages → global blocks) has not been
exercised. **Smoke-test it on `--partition=debug` before spending a 12 h slot.**

## Two arms need no runs

| file | run on disk | steps | sustained it/s |
|---|---|---|---|
| `classic.yaml` | `bench_v11_simple` | 235 500 | 3.005 |
| `convnext.yaml` | `bench_v11_convnext` | 278 000 | 3.518 |
| `transformer.yaml` | *(none — see below)* | — | — |

`config_check diff` reports **exactly one** key between each of those two files and the run on
disk: `dset.validation.n_batches` 50 → 100, raised on 2026-08-31 to match the transformer arms
(see below). It is validation monitoring — no gradient flows from it — so the trained model on
disk is the model these files describe, and both remain the round's anchors.

**But it is not entirely free, and the asymmetry runs one way.** Validation happens inside the
wall-clock budget, so the six arms this round actually launches pay ~50 extra forward batches
every `vali_every: 1000` steps: roughly **2 % of the step count**, or **~0.2 % of FoM** at the
measured +7.3 %-per-2× budget elasticity. The two anchors on disk did not pay it. That handicap
is uniform across all six new arms — so it confounds nothing *within* the round — and at ~0.2 %
it is a twentieth of the 0.049 floor and a tenth of the block confound already documented below.
Worth stating, not worth correcting for.

`transformer.yaml` is architecturally the prod recipe, but **the prod recipe has never been run at
its own wall budget**: every transformer run on disk (`bench_v7_transformer`) was sized under the
legacy fixed-`n_steps` mode at 180 k / 340 k / 260 k. Lensing's 180 k at 4.78 it/s = 37.7 ks is close
enough to 39 600 s to serve as an approximate anchor; combined's and clustering's are not. So the
transformer's own dropout contrast needs `transformer.yaml` run alongside it.

## The step confound, and which way it runs

**ConvNeXt is cheaper per step than the classic block, so at equal wall it trains on more samples.**
Measured, from each run's `throughput.json`:

| probe | classic it/s | convnext it/s | ratio | steps at equal wall | worth (at +7.3% per 2×) |
|---|---|---|---|---|---|
| combined | 3.005 | 3.518 | **1.17×** | 235 500 → 278 000 | ~+1.8% |
| lensing | 3.172 | ~3.85 *(projected)* | **~1.21×** | 125 300 → ~152 k | ~+2.3% |
| clustering | 3.144 | ~3.88 *(projected)* | **~1.23×** | 124 400 → ~154 k | ~+2.4% |

The lensing and clustering figures are projected from `bench_v11_convnext_mean_std` (3.804 / 3.825),
adjusted for `mean` being ~1.4% cheaper than `mean_std` as measured on combined. **They are not
measured for this exact geometry — confirm against job 1 and correct the table.**

Consequences, and they are not symmetric:

- **A ConvNeXt win smaller than ~2–2.5% is not evidence for the block.** That is precisely the trap
  bench_v6 fell into.
- **A ConvNeXt loss is unambiguous and strong** — it lost despite the extra samples.
- Equal wall is the decision-relevant comparison (the deployment question is "what is the best
  network in one 12 h job"), so the confound is stated rather than engineered away by shortening
  ConvNeXt's budget.

## What is already known, so the round does not re-ask it

- **Bare ConvNeXt at `mean`, combined: 1.025 `=` vs classic** (bench_v11), i.e. a wash once its
  1.17× step bonus is paid back. This round adds the two missing probes.
- **The full ConvNeXt stack** (block + DropPath + attention) is the current prod default and scores
  1.104 vs `simple` on combined at `mean_std`. bench_v12 does **not** include it: the round is about
  the simplest defensible net, and every partial strip-down of that stack has landed at or below
  plain classic.
- **DropPath's sign is block-dependent** — +5.0% on ConvNeXt, **0.946 on classic**. Both block arms
  therefore pin `drop_path_rate: 0.0`, which is what makes the block contrast clean.
- **Head dropout 0.1 has never been varied in any maps config, on either architecture.** It is an
  unexamined constant of the same kind `layer_scale_init` still is.
- v17 ran `t0_no_dropout` / `t0_block_dropout` (lensing) and `bench_t2_no_dropout` /
  `bench_t2_block_dropout` (clustering, combined). **Those run dirs have no `configs.yaml`**, so
  `run_comparison`'s comparability gate refuses them. Do not quote a hand-computed number from them.

## Is head dropout how it is done nowadays? No.

`Flatten → LayerNorm → Dropout(0.1) → Dense(10)` is the pre-DeiT recipe. ConvNeXt, DeiT/timm, Swin
and the ViT recipes everyone now follows set **head dropout to zero** and regularize with

1. **stochastic depth (DropPath)** in the body — characterized on the GCNN, newly available on
   the transformer, and what the two `*_droppath` arms test;
2. AdamW weight decay ~0.05 — available (`deep_lss/utils/optimization.py`) but **not tested here**:
   it has not paid off in previous rounds, and a third regularizer in a round that expects washes
   buys multiple comparisons rather than an answer;
3. augmentation and label smoothing, neither of which maps onto this task.

There is also a task-specific reason to doubt it: with `map_pool: mean` the dropout acts on an
**already spatially-averaged 512-d vector**, dropping ~51 pooled channels per step. That is a much
blunter regularizer than dropping activations inside a body, and the pooled representation is
exactly the thing the VMIM objective is trying to keep informative.

**Counter-argument, which is why this is a measurement and not an edit:** this is not ImageNet.
Training draws fresh noise realizations over a 1000-cosmology grid every epoch, so overfitting
pressure is weak, and the regularizer that is not needed is also the one that costs nothing to keep.
Expect washes on the regularization arms, and treat a wash as an argument to **drop** the knob —
a tie goes to the simpler network, and "no dropout, no DropPath" is simpler than either. The one
asymmetry: if `convnext_nodrop_droppath` ties `convnext_nodrop`, the tie favours **dropping
DropPath**, not keeping it for its ImageNet pedigree.

Note the train−vali gap cannot referee this: it is computed with dropout **active** on the train
side, so it is confounded across exactly the configs this round compares.

## Launching

Via the **`submit` skill** — no new submission script. `MODEL_DIR` mirrors the basename with a
`bench_v12_` prefix. Combined arms are 2-job `afterany` chains (`MAX_RUNS=2`, 79 200 s total).
`RUN_NUM=1` and a fresh `MODEL_DIR` for every arm — the block change alters weight shapes and
`expect_partial()` does **not** raise on a mismatch.

### The bench_v13 fold (2026-08-31, before any job started)

`bench_v13` was drafted as a follow-on round: one arm, testing whether the GCNN's two bare
Chebyshev convolutions could be replaced by ConvNeXt blocks so the network is nothing but strided
pseudo-convs and ConvNeXt blocks — a far simpler thing to draw beside the transformer in
`networks.tex`. It was smoke-tested (job 3244154, builds OK, +1% params, +6% step time, +35% peak
memory) and was waiting on this round's regularization answer.

It was folded in instead, because **none of the twelve jobs had started**. Testing the layout we
intend to ship costs the same twelve slots as testing the one we intend to replace, and it removes
a serial dependency: bench_v13 would otherwise have had to wait for bench_v12 to resolve, then run
a further two 12 h jobs, and then re-open the regularization question on a network neither arm of
this round had used.

**What the fold costs, stated plainly:**

1. **`classic_nodrop → staged_nodrop` is two knobs**, block and layout, not one. It is scored as a
   package and must never be quoted as a block measurement. bench_v11 already has the block alone
   at ~+0.7%, a wash, which is why this is acceptable rather than merely convenient.
2. **The legacy-layout ConvNeXt at this exact regularization is never run.** `bench_v11_convnext`
   is the nearest thing on disk and differs by head dropout (0.1, not null), so the layout knob is
   not cleanly attributable from anything that will exist. If `staged_nodrop` *loses* outside the
   floor, that attribution is what the round will want, and `_deferred/convnext_nodrop.yaml` is
   ready to run as the tiebreak — a further 2 jobs, only if needed.
3. `staged_nodrop_droppath` doses 7 blocks where the legacy arm dosed 5.

**What it buys:** the round now answers "should we ship the all-ConvNeXt GCNN?" directly, at no
extra cost and two weeks earlier than the sequential plan.

### THE ROUND IS COMBINED-ONLY (2026-08-31)

**Decide on `combined`, then transplant the winner to the other two probes.** The lensing and
clustering arms are written and parked in `<probe>/bench_v12/_deferred/` — nothing about them is
wrong, they are simply not part of the measurement. When combined resolves, the winning
combination's twin is moved back out of `_deferred/` and run once per probe.

This cuts the round from **26 training jobs to 12**, and the transplant adds **4** (lensing and
clustering are single 12 h jobs, one arm per architecture per probe).

Why this is sound rather than merely cheap: question **B** — where the regularization belongs — is
a property of the *shared head*, which is the same object on all three probes and both
architectures. Combined is also the probe the paper leads with and the only one with a 2×12 h
budget, so it is the least seed-noisy place to resolve a knob expected to be small.

**The one risk, stated:** bench_v11 already produced a probe-DEPENDENT answer once (`mean_std`:
+6.9 % combined, wash on lensing, −6.7 % resolved LOSS on clustering), so "combined says X" is not
proof that lensing and clustering agree. The transplant runs are themselves the check — each is
scored against its own probe's `bench_v11_simple` anchor, so a probe that disagrees shows up there
rather than being assumed away.

### No rate probe — `n_steps: auto` already removes the reason for one

An earlier draft of this plan opened with a `rate_probe.sh` job. It was dropped, because with
`n_steps: auto` + `wall_budget_seconds` the step count is an **output**, not an input: nothing
downstream needs a rate measured in advance, and the realised sustained rate lands in each run's
own `throughput.json` for free. The equal-wall confound can be read off those files afterwards,
which is when it is actually needed.

What the probe was really buying was a **smoke test**: no transformer arm here has ever executed
the three changes of 2026-08-31 (pre-pool `LayerNorm` removal, `map_feature_dim: null`, the
`drop_path_rate` plumbing). That does not need its own job either. `run_training.py` constructs
the network and traces it through `_build_trace()` at **line ~860**, before the input-norm
statistics are measured (line ~1073) and long before the training loop, so a plumbing error is a
crash within minutes of the job starting — not a wasted 12 h slot.

**So the only precaution left is ordering.** Launch `transformer_nodrop_droppath` **first and
alone** — it is the arm that exercises all three changes at once — and release the other five once
its log shows `Built transformer network nested_transformer`. It is a job the round needs anyway,
so this costs nothing. Worth it because the chains are `afterany`: a build-time crash in job 1
still releases job 2, so a typo launched across six chains wastes twelve queue slots instead of
two.

### The round — 12 jobs, all combined

| arm | architecture | jobs | tests |
|---|---|---|---|
| `classic` | GCNN | **0** — on disk as `bench_v11_simple` | anchor |
| `convnext` | GCNN | **0** — on disk as `bench_v11_convnext` | A (block) |
| `classic_nodrop` | GCNN | 2 | B1 (head dropout) on the classic block |
| `staged_nodrop` | GCNN | 2 | **A** — the all-ConvNeXt package vs the classic GCNN |
| `staged` | GCNN | 2 | **B1 on the ConvNeXt block**, and A read a second time at dropout ON |
| `staged_nodrop_droppath` | GCNN | 2 | B2 (body stochastic depth), one knob |
| `transformer` | transformer | 2 | the first wall-matched transformer anchor |
| `transformer_nodrop` | transformer | 2 | B1 |
| `transformer_nodrop_droppath` | transformer | 2 | B2 |

Both zero-job rows are verified identical by `config_check diff`, not assumed — see below.

### The transplant — 4 jobs, after the round resolves

One arm per architecture per probe: the winning (block, regularization) combination on GCNN and
the winning regularization on the transformer, run once on lensing and once on clustering. Move
the matching file out of `<probe>/bench_v12/_deferred/` rather than writing a new one — the
rationale headers are already correct there, including clustering's `base_embed_dim: 64` trunk
match.

**The fold added one step to the GCNN half.** The parked per-probe ConvNeXt files are all
**legacy-layout**, written before bench_v13 existed. If a staged arm wins, swap their four
`base_channels`/`pool_layers`/`conv_layers`/`residual_layers` keys for `stage_widths`/
`stage_blocks` before launching. **Lensing shares combined's trunk arithmetic** (`base_channels:
64`, `pool_layers: 3`) and so takes the stage list verbatim: `[64, 128, 256, 512, 512]` /
`[0, 0, 1, 1, 5]`. **Clustering does not** — it runs `base_channels: 128` with `pool_layers: 2`,
reaching the same 512-wide nside-16 trunk from a lower input nside in one halving fewer. Its stage
list must be derived from its own parked file; copying combined's would silently change its trunk.

## Scoring

**`compare-runs` (`run_comparison`), paired FoM, never by eye.** Floor **0.049**; a ratio inside
`1.000 ± 0.049` prints `=` and is a wash, and a tie goes to the simpler network.

- Reference `bench_v11_simple` per probe for the GCNN arms (it *is* `classic.yaml`).
- `--cross_modality` against `cls/<probe>/bench_v7` for the gain-over-two-point number.
- **Q3 `coverage` is REQUIRED** before promoting anything: the dropout arms change the head, and a
  pathological summary defeating the flow is the one failure mode Q1 and Q2 cannot see.
- **Q2 `robustness`** as a gate, not a ranking — most GCNN-vs-GCNN differences are washes at the
  0.24 σ floor. Compare only runs within 1.2× on step count; the ConvNeXt arms will be ~1.2× the
  classic arms, right at the warning threshold.
- DES FoM is unsigned and ranks nothing.

## Verified, not eyeballed

```bash
P=/users/athomsen/dlss/repos/y3-deep-lss/.venv/bin/python3
D=configs/maps/dev/deepsphere/combined
$P -m deep_lss.utils.config_check diff $D/bench_v11/simple.yaml $D/bench_v12/classic.yaml
$P -m deep_lss.utils.config_check diff $D/bench_v11/convnext.yaml $D/bench_v12/convnext.yaml
#   dset.validation.n_batches 50 -> 100                  -> 1 key each, validation monitoring
#   only; the trained model on disk IS the one these files describe
$P -m deep_lss.utils.config_check diff $D/bench_v12/classic.yaml $D/bench_v12/convnext.yaml
#   residual_block_type + mlp_ratio + layer_scale_init   -> 3 keys, 1 knob (the last two are
#   ConvNeXt-only kwargs resnet.py never reads on the classic branch -- inert there)
$P -m deep_lss.utils.config_check diff $D/bench_v12/classic.yaml $D/bench_v12/classic_nodrop.yaml
$P -m deep_lss.utils.config_check diff $D/bench_v12/convnext.yaml $D/bench_v12/convnext_nodrop.yaml
#   network.kwargs.dropout_rate 0.1 -> None              -> 1 key each
$P -m deep_lss.utils.config_check diff $D/bench_v12/staged_nodrop.yaml \
                                       $D/bench_v12/staged_nodrop_droppath.yaml
#   network.kwargs.drop_path_rate 0.0 -> 0.1             -> 1 key
$P -m deep_lss.utils.config_check diff $D/bench_v12/classic_nodrop.yaml \
                                       $D/bench_v12/staged_nodrop.yaml
#   9 keys, "MORE THAN ONE KNOB"                         -> EXPECTED, and the tool says so:
#   base_channels/pool_layers/conv_layers/residual_layers out, stage_widths/stage_blocks in,
#   plus the ConvNeXt block's inseparable three. It is the package contrast, declared as such.
$P -m deep_lss.utils.config_check check $D/bench_v12/*.yaml
#   exit 0; the 9 NOTES are the intended legacy-vs-staged key-set split
```

Both live `check` runs exit 0, as do the four parked `_deferred/` sets.

## Carried forward, untested

- **`mean_std`** — the best combined readout, excluded here for shared-core reasons above. If the
  paper wants it on combined only, that is a separate, argued decision.
- **`layer_scale_init`** — still never varied anywhere in `configs/`. Held fixed at 1.0e-6 in every
  ConvNeXt arm and absent from the classic arms, so it enters no contrast.
- **`body_dropout_rate`** (`SpatialDropout1D` in the trunk) — the conv-appropriate dropout, and the
  one regularizer aimed at *misspecification* rather than FoM. Scored 0.943 on FoM, which is the
  wrong observable for it: judge it on Q2 posterior bias. Not in this round.
- **The maps-only evaluation.** Every arm here carries the Cls branch.
