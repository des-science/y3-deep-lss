# bench_v7 — results (v18/default, maps)

Nine map-level runs across three probes and three architectures, plus a four-probe Cls baseline.
All ranked with `deep_lss/apps/tuning/run_comparison.py`, paired FoM(Ωm, S8), **1000 mocks**, bootstrap CI
over mocks only. **Seed floor 1.5%** — treat `|ratio − 1| < 0.015` as a wash regardless of the CI.

Scored 2026-08-11. Round definition and per-lever rationale: `README.md` in this directory and
`configs/transformer/combined/bench_v7/README.md`.

## The two headline results

1. **Within the GCNN, the elaborate encoder barely earns its keep.** `full` (ConvNeXt + DropPath +
   attention) over `simple` (classic block): +2.6% lensing, +8.5% clustering, +4.1% combined. Only
   clustering is comfortably clear of the seed floor.
2. **On combined, the transformer wins by 12.3% over the best GCNN** — on 82% of mocks, using *less*
   training wall. That is ~3× any within-GCNN effect in the round, and it is the result the round was
   not designed to look for.

The transformer's margin **tracks how much cross-probe fusion the task requires**: a wash on lensing,
tied on clustering, decisive on combined. Consistent with where the architectures differ (the nested
transformer does content-dependent mixing at nside 512/256; the GCNN's first real graph convolution
runs at nside 64 and the probes meet in a pointwise Dense). Suggestive, not established — this round
varied architecture, not the fusion mechanism. It is the premise of bench_v8.

## Per-probe ranking vs that probe's `simple` anchor

Anchor = `bench_v7_simple`, the bench_v6 anchor recipe (classic block, mean-pool readout, no
attention). All of these passed the **strict** comparability gate.

### Lensing — everything is a wash

| arm | architecture | steps | train wall | ratio | 95% CI | win% | vali_total |
|---|---|---|---|---|---|---|---|
| `bench_v7_full` | GCNN · ConvNeXt + attn | 130 000 | 10.28 h | **1.026** | [1.019, 1.033] | 61% | −2.797 |
| `bench_v7_transformer` | nested transformer | 180 000 | 10.35 h | **1.014** `=` | [1.003, 1.030] | 54% | −2.799 |
| `bench_v7_simple` | GCNN · classic | 110 000 | 9.86 h | 1.000 | — | — | −2.637 |

### Clustering — the one probe where the stack pays

| arm | architecture | steps | train wall | ratio | 95% CI | win% | vali_total |
|---|---|---|---|---|---|---|---|
| `bench_v7_transformer` | nested transformer | 260 000 | 9.99 h | **1.091** | [1.078, 1.105] | 70% | −9.017 |
| `bench_v7_full` | GCNN · ConvNeXt + attn | 130 000 | 10.16 h | **1.085** | [1.074, 1.099] | 71% | −9.188 |
| `bench_v7_simple` | GCNN · classic | 110 000 | 9.68 h | 1.000 | — | — | −8.835 |

The top two are **0.6% apart — read them as tied.** Clustering is also the probe that needed
`base_channels: 128` to reach the pinned 512 trunk, so this round cannot separate the
block-and-attention stack from that wider stem.

### Combined — the transformer pulls clear

| arm | architecture | steps | train wall | ratio | 95% CI | win% | vali_total |
|---|---|---|---|---|---|---|---|
| `bench_v7_transformer` | nested transformer | 340 000 | 22.95 h | **1.153** | [1.144, 1.163] | 87% | −10.741 |
| `bench_v7_full` | GCNN · ConvNeXt + attn | 250 000 | 23.81 h | **1.041** | [1.032, 1.048] | 64% | −10.153 |
| `bench_v7_simple` | GCNN · classic | 230 000 | 22.21 h | 1.000 | — | — | −10.024 |

Head-to-head: **`transformer ÷ full` = 1.123 [1.108, 1.134]**, 82% of mocks — while using 0.86 h
*less* training wall, so it is not a budget artifact in either direction.

## vs the Cls (two-point) baseline — every network wins

Reference `cls/<probe>/bench_v7`. **The requirement that the networks outperform the Cls is met by all
nine arms**; the narrowest margin is clustering `simple` at 1.261 on 80% of mocks.

| probe | arm | ratio vs Cls | 95% CI | win% |
|---|---|---|---|---|
| lensing | `bench_v7_full` | **2.375** | [2.300, 2.463] | 99% |
| lensing | `bench_v7_simple` | 2.331 | [2.257, 2.429] | 99% |
| lensing | `bench_v7_transformer` | 2.251 | [2.190, 2.342] | 96% |
| clustering | `bench_v7_full` | **1.395** | [1.355, 1.423] | 87% |
| clustering | `bench_v7_transformer` | 1.372 | [1.343, 1.404] | 88% |
| clustering | `bench_v7_simple` | 1.261 | [1.228, 1.289] | 80% |
| combined | `bench_v7_transformer` | **1.558** | [1.526, 1.589] | 96% |
| combined | `bench_v7_full` | 1.377 | [1.352, 1.407] | 89% |
| combined | `bench_v7_simple` | 1.356 | [1.323, 1.379] | 90% |

Note the ordering flips between probes: on **lensing** the GCNN arms beat the transformer against the
two-point baseline, on **combined** the transformer is far ahead. Same pattern as the anchor ranking.

### The Cls baseline was re-run on purpose

`cls/<probe>/bench_v7` (job **3056843**, 2026-08-11, four probes in parallel on one node, 21 min) is
the same recipe as the earlier `cls/<probe>/v1` — pure `cls/cls_training.sh` defaults, MLP, 1e6 steps.
It was re-run because `v1`'s **inference** ran at 10:54 on 2026-08-06 while the canonical
mock-ordering fix `68210dd` was committed at 13:33 the same day, and the documented rule is that the
cutoff is the commit, not the date. A fresh run removes the question.

**It turned out `v1` was fine** — the fresh baseline reproduces it to within 1% everywhere (lensing
full 2.398 → 2.375, clustering full 1.392 → 1.395, combined full 1.380 → 1.377, combined transformer
1.557 → 1.558). The fix was authored 2026-07-28 and only committed on 08-06, so `v1`'s inference
already ran against it from the working tree. **Both pair 1000/1000 with no `--intersect`.** Keep the
commit-not-date rule anyway; here it produced a false alarm rather than a missed one, which is the
right direction for that kind of check to fail.

`cls/2x2pt/bench_v7` also exists and is scored; it has no map-level counterpart in this round.

## Cross-probe sanity gate — passes, and had never been checked

Carried as an open item since bench_v4: combined must be at least as informative as either single
probe.

| comparison | `full` arm | `simple` arm | `transformer` arm |
|---|---|---|---|
| combined ÷ lensing | 2.061 [2.013, 2.130] | 2.081 [2.029, 2.137] | **2.487** [2.411, 2.556] |
| combined ÷ clustering | 4.356 [4.224, 4.484] | 4.704 [4.591, 4.869] | **4.938** [4.786, 5.097] |
| lensing ÷ clustering | 2.072 [1.976, 2.169] | 2.309 [2.209, 2.386] | 1.962 [1.890, 2.032] |

Passes in every arm, most strongly for the transformer. Lensing carries ~2× the information of
clustering throughout.

**These legs required `--no_strict`, and so did every Cls comparison above.** In both cases the gate
objected to exactly one field and nothing else — `probe` for the cross-probe legs, `input_modality`
(`maps+cls` vs `cls`) for the Cls legs — while head type, θ standardization, summary-dim factor, loss
and split all matched, and the **mock-identity gate stayed active and passed** on an identical
1000-mock set. The ratios are on one scale; the gate was blocking the intended treatment axis, which
is the deliberate case its docstring sanctions. The per-probe anchor tables passed strict.

## Caveats

**Wall-matched, not step-matched — and that is the fair axis here.** Every arm got roughly the same
training wall (9.7–10.4 h single, 22.2–23.8 h combined) and its step budget was *sized to fill it*.
Faster architectures therefore take more steps, and at a fixed compute budget that is a real
advantage, not a confound. Judged at equal *steps* instead, lensing's 1.026 would shrink into the
noise while clustering's 8–9% would survive. A residual 4–7% wall advantage to `full` does remain,
worth a few tenths of a percent.

**Informativeness only.** No robustness and no estimator validity has been checked on **any** of the
nine runs — no posterior bias on the systematics mocks, no SBC/TARP/HPD, no PPC on real data. A good
FoM ratio substitutes for none of them, and on misspecified data the DES FoM is unsigned. This matters
directly for bench_v8: the GCNN is preferred *because* it is more robust (v17: every GCNN ≤0.27σ vs
`t2_cls` at 0.44σ, driven by source clustering at 1.12σ), and **that comparison has never been made on
v18**, where source clustering moved into the shape noise. The premise of the follow-up round is
itself unverified.

**Two combined configs no longer describe their runs.** `bench_v7/full.yaml` and `simple.yaml` carry
`!! PROVENANCE !!` blocks: the runs trained 250 k and 230 k steps, the files now specify 200 k and
220 k for a fresh chain re-sized on measured v18 rates. Node-to-node rate variation on combined
reached **24%** between two jobs of the same config, so those budgets are sized on the minimum
observed rate. The `long` arm of bench_v8 inherits that rule over four jobs.

**Three of the nine lost their eval stage to a mid-flight repo change.** The msfm `data/`
reorganization (`cf3f04f`, 2026-08-07) renamed four files that each run's own `configs.yaml` pins as
absolute paths at submission time. Both combined GCNN arms and the combined transformer failed on it
*after* training successfully; all snapshots were patched and the eval+inference re-run (jobs 3055891,
3055892, 3056404). No training was lost.

**The combined transformer spent five days in a directory called `default`.** It was launched without
`MODEL_DIR`, so it took the config basename. It escaped the 2026-08-06 cleanup that renamed the two
single-probe transformer runs because its eval had died the same day — so it had no `preds_*.h5` and
never appeared in an output inventory. The wrong name and the missing outputs hid each other. Renamed
to `bench_v7_transformer` on 2026-08-11. Audit run *dirs*, not just outputs.

## Run inventory

All under `/iopsstor/scratch/cscs/athomsen/deep_lss/runs/v18/default/`, mirrored byte-identically on
`/capstor/store/cscs/swissai/a0158/athomsen/deep_lss/runs/`.

| | lensing | clustering | combined | 2x2pt |
|---|---|---|---|---|
| `maps/<probe>/bench_v7_simple` | ✅ | ✅ | ✅ | — |
| `maps/<probe>/bench_v7_full` | ✅ | ✅ | ✅ | — |
| `maps/<probe>/bench_v7_transformer` | ✅ | ✅ | ✅ | — |
| `cls/<probe>/bench_v7` | ✅ | ✅ | ✅ | ✅ |

Configs: `configs/deepsphere/<probe>/bench_v7/{full,simple}.yaml`,
`configs/transformer/<probe>/bench_v7/transformer.yaml` (renamed from `default.yaml` on 2026-08-11),
and `cls_training.sh` defaults for the Cls arms.

## Follow-up

`configs/deepsphere/combined/bench_v8/` — `long` (2× budget, robustness-neutral, known +7.3% lever)
and `injection_conv` (one graph conv at nside 256 on the fused stream, targeting the fusion gap
above). Both one knob off `bench_v7/full.yaml`. See that round's README, including the measured-dead
downsampling levers re-scored from bench_v4 and the robustness gate `injection_conv` requires.
