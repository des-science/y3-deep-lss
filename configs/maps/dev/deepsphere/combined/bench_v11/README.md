# bench_v11 — COMBINED, the SUBTRACTIVE round

**Status (2026-08-28): COMPLETE. All 5 arms ran to budget; the round settled its own question.**
Jobs 3200020-3200030, 5 x 2-job `afterany` chains, all COMPLETED 0:0, all 79 200 / 79 200 s consumed.

## RESULT — adopt `simple_mean_std`; delete ConvNeXt, DropPath and attention

The pre-registered rule in `simple_mean_std.yaml` was `>= 0.951 -> ADOPT`. It measured **0.970**.

**Paired FoM (`run_comparison`, 1000 mocks, final checkpoints, seed floor 0.049):**

| arm | steps as run | it/s | vs `simple` | vs `bench_v8_mean_std` |
|---|---|---|---|---|
| `bench_v8_mean_std` (champion) | 260 000 | 3.32 | **1.104** | 1.000 |
| **`simple_mean_std`** | 235 400 | 3.02 | **1.069** | **0.970 =** |
| `noattn` | 273 800 | 3.55 | 1.054 | 0.950 |
| `nodroppath` | 249 300 | 3.18 | 1.039 = | 0.942 |
| `convnext` | 278 000 | 3.52 | 1.025 = | 0.926 |
| `simple` (floor) | 235 500 | 3.00 | 1.000 | 0.906 |

**The readout is the whole gain.** `simple -> simple_mean_std` is the cleanest contrast the
programme has produced: one knob, and the two arms landed within **100 steps** of each other
(235 500 vs 235 400) under the wall budget, so it is equal-wall *and* equal-samples. It is worth
**+6.9%**. The champion's remaining margin over it — 1.104 / 1.069 = **1.033** — is inside the floor.
Three blocks, a wash.

**ConvNeXt alone is nothing, once its own confound is paid back.** The header declared ~17% extra
steps at fixed wall; it realised **1.18x** (278 000 vs 235 500) and the declared "a win under ~2% is
not evidence for the block" rule therefore stands as written. At the measured +7.3%-per-2x budget
elasticity that 1.18x is worth ~+1.75%, so of `convnext`'s raw 1.025 the **block contributes ~+0.7%**.

**`simple_mean_std` ties the champion from behind.** It is the *slowest-but-one* arm and trained
**9.5% fewer steps** than the champion (235 400 vs 260 000, ~ -1.05% of FoM). Correcting for that,
the tie is ~0.980 — the wash is more comfortable than the raw number, not less, which is the safe
direction for a decision to *remove* machinery.

**`nodroppath` is uninterpretable as designed, and stayed that way.** Two knobs (DropPath off *and*
attention every:1), and the every:1 change made it the slowest arm at 3.18 it/s -> 4% fewer steps
than the champion. Its 0.942 is a loss outside the floor, but it is not a DropPath measurement and
must never be quoted as one.

**`noattn` at 0.950 sits one thousandth outside the 0.049 floor** — treat it as the boundary case it
is, not as a resolved 5% cost of removing attention. It does not matter for the decision: it is
*both* worse and more complex than `simple_mean_std`, which settles the round on its own.

**Q2 robustness (`run_diagnostics robustness`): every row a wash, no disasters.** MEAN 0.229-0.290
across all six runs, `source_clustering` MAX 0.333-0.526, every row inside the floor and marked `=`.
`simple_mean_std` MEAN 0.261 vs the champion's 0.265. Removing the three blocks costs no robustness.

**Q3 coverage (`run_diagnostics coverage`, paired vs the champion): gate PASSED.** `simple_mean_std`
HPD delta **-0.0057 [-0.0162, +0.0053]**, CI spans 0. SBC shows the cohort-wide mild rejection on
Om/s8/w0 and clean nuisances that *every* v18 run including the transformer and the Cls baselines
shows — a property of the mock set, not the architecture. No pathology from the 512 -> 1024 readout.

**Cost.** `simple_mean_std` is **18.47 M** trainable params vs the champion's **20.54 M** — 10% fewer,
with the ConvNeXt block, stochastic depth and both attention blocks gone.

**Not yet done:** the without-Cls evaluation, and the lensing/clustering transplant.


**The question.** Every previous round asked *what can we add*. This one asks **what can we delete**:
what is the simplest GCNN that is still meaningfully better than `bench_v7_simple`? The combined
probe is the paper's flagship, so it is the probe that decides. Every arm carries the Cls branch.

**Why subtractive, and why now.** The last three rounds closed the additive question:

- **The trunk is finished.** Six arms across bench_v8/v9 added machinery to it — injection convs, a
  split pool, a seam nonlinearity, two U-net schedules — and **all six lost** (0.836–0.933).
- **The readout is the only lever that ever paid, and it has a peak, not a slope.**
  flatten→mean **+32.1%**, mean→mean_std **+5.3%**, mean_std→moments **−12.2%**.
- **The champion's other three blocks have never been ablated on v18.** Their only one-knob numbers
  are v17 and sit *entirely inside* today's 0.049 floor: block 1.024, DropPath-on-ConvNeXt 1.050,
  attention 1.005–1.056. bench_v7's README says in as many words that its `full`-vs-`simple` contrast
  moves three knobs at once and is not evidence about any one of them. Every arm of bench_v8, v9 and
  v10 then inherited all three unexamined.

So the champion carries three blocks that nothing on this dataset has justified. This round removes
them, one contrast at a time.

**Not in scope, deliberately.** Closing the transformer's 12.3% combined lead (`bench_v7_transformer`
1.123 vs the champion's 1.053). The GCNN is the chosen compression *because* it is robust — every
GCNN ≤0.27σ on the systematics mocks against t2_cls's 1.12σ on source clustering — and the FoM gap is
a Q1/Q2 trade-off the paper reports rather than averages. No additive arm in this round.

## The arms

Anchor: **`bench_v8_mean_std`** (the programme champion, on disk, 79 200 s — the *same* wall as every
arm here, so every comparison is equal-wall by construction).

| arm | block | DropPath | attention | readout | jobs |
|---|---|---|---|---|---|
| `bench_v8_mean_std` **(on disk — the reference)** | convnext | 0.1 | every 2 | mean_std | — |
| `bench_v7_full` **(on disk — free)** | convnext | 0.1 | every 2 | mean | — |
| **`simple.yaml`** — the floor | classic | 0.0 | none | mean | 2 |
| **`simple_mean_std.yaml`** — the candidate | classic | 0.0 | none | **mean_std** | 2 |
| **`convnext.yaml`** — the block test | **convnext** | 0.0 | none | mean | 2 |
| **`noattn.yaml`** | convnext | 0.1 | **none** | mean_std | 2 |
| **`nodroppath.yaml`** | convnext | **0.0** | **every 1** | mean_std | 2 |

### The contrasts, and which are clean

| contrast | knob | clean? |
|---|---|---|
| `bench_v8_mean_std` → `noattn` | interleaved attention | ✅ one knob |
| `simple` → `simple_mean_std` | `map_pool` mean → mean_std | ✅ one knob |
| `simple` → `convnext` | `residual_block_type` classic → convnext | ✅ one knob — **but see the step confound** |
| `bench_v7_full` → `bench_v8_mean_std` | `map_pool`, at the *full* operating point | ✅ free, already scored (+5.3%) |
| `simple_mean_std` → `noattn` | the ConvNeXt **+** DropPath package | ✅ as a **package** — see below |
| `simple` → `bench_v8_mean_std` | all four blocks at once | ✅ the whole-stack number, equal wall |
| `bench_v8_mean_std` → `nodroppath` | DropPath **and** attention density | ❌ **two knobs** — see below |

**`simple_mean_std` → `noattn` is a package, not a confound.** They differ by `residual_block_type`
*and* `drop_path_rate` — but DropPath's sign is **block-dependent** (+5.0% on ConvNeXt, **0.946 on
classic**), so the two are not separable in principle and bench_v6 found the champion's entire gain
was the block×DropPath *interaction*: block + budget alone bought 0.998, a wash. Testing "ConvNeXt
with its DropPath" against "classic without" is the right granularity. It prints as 4 keys because
`mlp_ratio` and `layer_scale_init` are ConvNeXt-only kwargs `resnet.py` never reads on the classic
branch — inert, verified at [resnet.py:280-292](../../../../deep_lss/nets/encoders/maps/gcnn/resnet.py#L280-L292).

**`simple` → `convnext` is the first isolated block measurement in the programme, and its confound
runs the other way from every other arm's.** DropPath is pinned at 0.0 and attention is off on *both*
sides, so this is the ConvNeXt block alone — something no run on any dataset has ever measured. The
programme's only one-knob block number is v17's **1.024**, inside today's floor, and bench_v6's reading
was that the block does nothing alone (block + budget bought **0.998**) and pays only through its
interaction with DropPath. But ConvNeXt is *faster* than the classic block at base 64 (~3.52 vs ~3.00
it/s), so at fixed wall this arm buys **~17% more steps** — worth about **+1.7% of FoM before the block
does anything**. Therefore: **a convnext win under ~2% is not evidence for the block** (that is exactly
the trap bench_v6 fell into), while **a convnext loss is unambiguous and strong** — it lost despite 17%
more steps. Quote the realised step counts next to the ratio.

**`nodroppath` carries two knobs and cannot be attributed.** On 2026-08-27 the round adopted
"attention after every layer, that is cleaner" as a recipe decision, so this arm is
`residual_attention_every: 1` (5 interleaved blocks, not 2) *as well as* `drop_path_rate: 0.0`. A
dedicated one-knob `attn1` arm was offered and declined in favour of holding the round to 8 jobs.
Consequence, stated rather than corrected: **a wash is still actionable** (both knobs point at
"simpler and cheaper is not worse", and a tie goes to the simpler net), but **a loss is
uninterpretable** — DropPath, or 5 attention blocks being worse than 2, or the steps the extra blocks
cost. Do not quote this arm as a DropPath measurement. `noattn` is unaffected: its
`residual_attention_every: 1` is inert because `residual_attention` is null.

### `simple.yaml` is a re-run, and that is the point

`bench_v7_simple` is on disk but trained **230 000 steps over ~23.8 h** under fixed `n_steps`, against
the 79 200 s = 22.0 h every arm here gets — ~8% more wall, worth roughly +0.8% of FoM at the measured
~+7%-per-2×-budget lever. Small, but it points the **wrong way**: the floor is what every arm must
clear, so a handicapped-upward floor makes the machinery look more necessary than it is, and this
round exists to delete machinery. Two jobs remove the confound instead of arguing around it in every
row of the results table. The architecture is untouched, key for key.

### `simple_mean_std.yaml` is the arm that can settle the round outright

It is `bench_v7/simple.yaml` plus the only lever that has ever paid, and nothing else. If it ties the
champion, **three blocks leave the recipe in one step** and the paper's architecture becomes a plain
ResNet GCNN with a two-moment pooled readout. The other three arms exist to answer the case where it
does not.

## The rate spread, which is a real handicap and runs one way

Equal wall is equal samples only if the rates match, and here they do not:

| arm | expected it/s | steps vs champion at equal wall |
|---|---|---|
| `bench_v8_mean_std` (ref) | ~3.25 | — |
| `simple`, `simple_mean_std` | ~3.00 | **~10% fewer** (classic is *slower* than ConvNeXt at base 64) |
| `convnext` | **~3.52** | **~8% more** than the champion, **~17% more than `simple`** — the arm's own confound |
| `noattn` | ~3.35 | ~3% more |
| `nodroppath` | ~3.3 | ~flat (no DropPath ≈ +4.8%, 2→5 attention blocks ≈ −4.8%; a coincidence, not a design) |

The classic arms are handicapped by ~10% of steps — i.e. **against** the round's preferred conclusion,
so a wash from `simple_mean_std` is conservative. **Read the realised counts from each run's
`throughput.json` before quoting any ratio**; if a rate is off these estimates by more than a few
percent, the arm carries a handicap that is not its knob.

## Verified, not eyeballed

```bash
P=/users/athomsen/dlss/repos/y3-deep-lss/.venv/bin/python3
D=configs/maps/dev/deepsphere/combined
$P -m deep_lss.utils.config_check diff $D/bench_v7/simple.yaml   $D/bench_v11/simple.yaml
#   training.{n_steps -> auto, wall_budget_seconds, job_budget_seconds}   -> 3 keys = the sizing MODE
$P -m deep_lss.utils.config_check diff $D/bench_v11/simple.yaml  $D/bench_v11/simple_mean_std.yaml
#   network.map_pool  'mean' -> 'mean_std'                                -> 1 key
$P -m deep_lss.utils.config_check diff $D/bench_v11/simple.yaml  $D/bench_v11/convnext.yaml
#   residual_block_type + mlp_ratio + layer_scale_init                    -> 3 keys, 1 knob (see below)
$P -m deep_lss.utils.config_check diff $D/bench_v8/mean_std.yaml $D/bench_v11/noattn.yaml
#   the 6-key residual_attention subtree -> null, + the inert _every      -> 1 knob
$P -m deep_lss.utils.config_check diff $D/bench_v8/mean_std.yaml $D/bench_v11/nodroppath.yaml
#   drop_path_rate 0.1 -> 0.0, residual_attention_every 2 -> 1            -> 2 knobs, declared
$P -m deep_lss.utils.config_check check $D/bench_v11/*.yaml
#   exit 0; the NOTES are the intended classic-vs-convnext key-set split
```

## Pre-launch checks

1. **Each arm built its own knob.** Diff the `ResNetSummaryNetwork:` line across the four training
   logs. `noattn` must show no attention blocks; `nodroppath` must show **five**, not two.
2. **The budget engaged.** `Wall-clock training budget: <total> s total, <spent> s already spent…` —
   `already spent` must be 0 in job 1 and **non-zero** in job 2 of every chain.
3. **Fresh lineages, every arm.** `RUN_NUM=1`, own `MODEL_DIR`. Three of the four change weight
   shapes (readout width or attention blocks) and `expect_partial()` will **not** raise on a
   mismatch — it loads the shared layers and silently leaves the rest. `simple.yaml` needs it too:
   restoring a fixed-`n_steps` checkpoint into a wall-budget run resumes a cosine parameterised in
   *steps* under a schedule now parameterised in *seconds*, and nothing raises on that either.
4. **Commit the tree before launch.** Job 2 re-imports the working tree hours later; an uncommitted
   change between jobs silently changes the architecture mid-chain.

## Launching

Via the **`submit` skill** — no new submission script. `MAX_RUNS=2` (combined convention, 79 200 s
total), `MODEL_DIR` mirroring the basename with a `bench_v11_` prefix: `bench_v11_simple`,
`bench_v11_simple_mean_std`, `bench_v11_convnext`, `bench_v11_noattn`, `bench_v11_nodroppath`.

## Scoring

**`compare-runs` (`deep_lss.apps.tuning.run_comparison`), paired FoM, every arm against
`bench_v8_mean_std`, highest evaluated checkpoint.** Never by eye.

Q1 floor **0.049** — a ratio inside `1.000 ± 0.049` prints `=` and is a wash, and by
[[feedback_conservative_decision_floors]] **a tie goes to the simpler network**. That rule is what
makes this round cheap: the null result is the actionable one.

**Q2 is a gate, not an afterthought.** bench_v10 adopted none of its winners precisely because
posterior bias on the contamination mocks was never measured. Gate on
`source_clustering_{gatti,in_place}` via `run_diagnostics.py` — **not** the worst-overall number, since
`dmo` biases every architecture by +0.58 to +0.97σ regardless of compression and always wins a max.
At the 0.24σ floor most GCNN-vs-GCNN Q2 differences are washes, so this catches disasters rather than
ranking arms. DES FoM is unsigned and ranks nothing.

## Carried forward, untested

- **The fourth cell of the block × readout square is NOT in this round.** With `convnext.yaml` added,
  three of the four (block, readout) combinations at DropPath 0 / no attention are covered:

  | | `mean` | `mean_std` |
  |---|---|---|
  | **classic** | `simple` | `simple_mean_std` |
  | **convnext** | `convnext` | *missing* |

  The missing cell — ConvNeXt, DropPath 0, no attention, `mean_std` — is both the interaction test
  (does the block still pay once the readout carries the second moment?) and a plausible shipping
  architecture in its own right: the merge of the round's two most likely survivors. It is 2 jobs
  whenever wanted, and is the obvious follow-up if `convnext` clears its ~2% step confound.
- **`layer_scale_init` has never been varied anywhere in `configs/`** — two constants, `1.0e-6` on the
  ConvNeXt block and `1.0e-4` on the attention block, both inherited upstream defaults (ConvNeXt /
  CaiT). It is a *trainable* per-channel gate on the residual branch, so the small value is a
  near-identity init, not an operating scale. Held fixed in every arm that uses it and absent from the
  classic arms, so it does not enter any contrast here. Same "never checked" status the optimizer had
  before `bench_v8/lr_3e4`.
- **`n_neighbors: 60`** — settled by bench_v10, not re-opened: k=20 wins both single probes but loses
  6% on combined, and combined is the flagship.
- **The maps/Cls dimensionality balance** — `mean_std` doubles the map readout to 1024 against a
  pinned 512-d Cls embedding. Every mean_std arm here carries it identically, so it cannot affect any
  ratio against the champion. `mean_wide` remains the uncut control.
- **`_deferred/`** — `flatten.yaml` / `mean.yaml`, the maps-only readout contrast, shelved when the
  round was scoped to maps+Cls. Its header records the one open consequence: the three prod
  `maps.yaml` files were switched to `map_pool: mean` ahead of the evidence, and this pair was what
  would have validated that switch.

## Next, if `simple_mean_std` survives

Transplant **only the surviving architecture** to lensing and clustering, anchored on
`bench_v10_mean_std_2x`, which already exists on both probes at the same 2× wall. That is 4 jobs, not
a round.

Then evaluate the winner **without the Cls branch** — every arm here carries it, so the maps-only
number is a separate measurement, not a re-read of these runs.

## FOLLOW-UP LAUNCHED 2026-08-28 — `convnext_mean_std`, the missing cell

The user's stated preference is to **use ConvNeXt**. The round as scored could not say which
ConvNeXt recipe that should mean, because every ConvNeXt run on disk carries DropPath, attention,
or both. So the block x readout square was closed:

|  | mean readout | mean_std readout |
|---|---|---|
| **classic** | `simple` 1.000 | `simple_mean_std` **1.069** |
| **convnext, bare** | `convnext` 1.025 **=** | **`convnext_mean_std` — RUNNING** |
| convnext + DropPath | — | `noattn` 1.054 |
| convnext + attention | — | `nodroppath` 1.039 (two knobs) |
| convnext + DropPath + attention | `v2` 1.074 | `bench_v8_mean_std` **1.104** |

**Jobs:** combined 3211923 -> 3211924 (2-job `afterany` chain, 79 200 s); lensing 3211925 and
clustering 3211926 (1 job each, 39 600 s, equal wall with `bench_v10_mean_std_1x` and `v2`).

**Prediction on the record before the run:** ~1.02-1.05 vs `simple`, i.e. NOT beating
`simple_mean_std`. Every partial strip-down of the stack lands at or below plain classic, and the
isolated block at `mean` readout was a wash. If that holds, **ConvNeXt only pays as the full
package** and the recipe to adopt is `bench_v8_mean_std` / `bench_v10_mean_std_1x`, machinery
included. Above ~1.12 would overturn it and make bare ConvNeXt the best simple net in the programme.
