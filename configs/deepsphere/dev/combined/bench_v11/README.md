# bench_v11 — COMBINED, the SUBTRACTIVE round

**Status (2026-08-27): configs written and verified, NOTHING LAUNCHED.** 5 arms, 10 jobs, combined only.

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
D=configs/deepsphere/dev/combined
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
