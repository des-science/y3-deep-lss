# bench_v6 `_deferred/` — shelved, not deleted

Excluded from the submit glob (`bench_v6/*.yaml` does not recurse), same convention as `bench_v5/_deferred/`.

All three were part of the first draft of bench_v6 and were deferred in the 2026-07-26 rewrite, when the
1000-mock paired re-analysis (`dev/notes/bench_v5_paired_reanalysis_2026-07-26.md`) restored the BUDGET lever
(+7.3 % per doubling, 73 % of mocks) that the round had been designed to exclude. Budget arms displaced them.
None was deferred because it is a bad idea; each is a refinement whose answer is worth less than a 2×12 h slot
at this point in the search.

**Their headers still quote the old 16-mock numbers** (+15.6 % width, +12.7 % DropPath, +7.9 % attention,
−0.5 % ConvNeXt). Those magnitudes are inflated — the re-derived values are +8.4 / +8.5 / +5.6 / +2.4 % — and
the "2× budget bought ZERO signed gain" line every one of them repeats is **sign-reversed**. Rewrite the header
before reviving any of these.

| config | axis | why deferred | revive when |
|---|---|---|---|
| `attn.yaml` | global attention WITHOUT DropPath | Attention is already attributable in the live round as `droppath_attn.yaml / droppath.yaml`. This arm answers the second-order question "does attention pay on its own?", which changes no decision: nothing in the round would ship attention without DropPath. | Only if DropPath fails to transfer to base 64 (`droppath.yaml` ≤ `default.yaml`) — then attention needs its own clean base. |
| `droppath_strong.yaml` | DropPath dose 0.2 | A dose-response needs a point past the optimum to bound the effect, which is sound — but the dose question is premature while the 0.1 → base-64 *transfer* is itself untested. Running 0.2 alongside 0.1 spends a slot bracketing an optimum that may not exist at this width. | After `droppath.yaml` confirms transfer. It becomes MORE interesting once budget is restored: longer runs raise the overfitting pressure a stronger regularizer would relieve, so pair the revival with a long arm rather than a 2×12 h one. |
| `droppath_deep.yaml` | residual_layers 5 → 8 × DropPath interaction | Lowest expected value in the round. bench_v4 already killed depth alone (`deep_trunk`: strong on grid mocks, collapsed on DES), and this was the most expensive config *and* the one most at risk of not fitting its wall. With budget re-established as a real lever, the same wall-clock buys more steps on a known-good geometry instead of more blocks on an unknown one. | Only if `droppath.yaml` shows DropPath's gain *growing* with capacity at base 64, which would be the first positive evidence for the depth × DropPath premise. |
