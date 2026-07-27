# Two-node training benchmark — clustering maps+cls transformer (2026-07-19)

**Question:** the v17 unified recipe (global batch 80, 250k steps) runs at 7.34 it/s on one node
(4 GPUs, `MirroredStrategy`) = 9h28m training + ~32 min eval/inference tail (t2_cls job 2781360,
total 9:59:42). Does it fit a single 12 h job on **two** nodes?

**Answer: yes, comfortably.** Horovod on 2 nodes runs the recipe-preserving configuration
(8 replicas × local batch 10 = global 80) at **16.0 it/s pure step rate** — projected
**~5.7 h training + tail ≈ 6¼ h total**, vs 10 h on one node. MultiWorkerMirroredStrategy also
works (13.5 it/s, ~7 h total) but is slower and has an input-norm caveat (below).

## Measured numbers

Benchmark: `submissions/clariden/benchmark/benchmark_2node.sh` +
`configs/transformer/clustering/bench_2node/{ctrl_b20,b10,b20}.yaml` — real `run_training.py`
with `--pasc_throughput` (wall time of steps 200→1200), `vali_every: Null`, no wandb, no
eval/inference. Jobs 2795154 (ctrl) and 2795238/40/41/42 (fixed re-run wave).

| variant | topology | global batch | ex/s | it/s | vs ctrl |
|---|---|---|---|---|---|
| ctrl mirrored (production geometry) | 1 node, 1 task × 4 GPU | 80 | 688 | 8.61 | 1.00× |
| MWMS 1-node | 1 node, 4 task × 1 GPU | 80 | 792 | 9.90 | 1.15× |
| MWMS 2-node b10 | 2 nodes, 8 task × 1 GPU | 80 | 1081 | 13.5 | 1.57× |
| MWMS 2-node b20 | 2 nodes, 8 task × 1 GPU | 160 | 1518 | 9.5 | 2.21× ex/s |
| **Horovod 2-node b10** | 2 nodes, 8 task × 1 GPU | 80 | **1279** | **16.0** | **1.86×** |

Pure ctrl rate 8.61 it/s vs the 7.34 it/s production average ⇒ vali/ckpt/startup overhead
≈ +1.4 h absolute on the production run; projections below carry that same absolute overhead.

- **Horovod b10**: 250k / 16.0 = 4.34 h pure → **≈ 5.7 h training**, +0.5 h tail ≈ **6¼ h total**.
- **MWMS b10**: 250k / 13.5 = 5.14 h pure → ≈ 6.5 h training, ≈ 7 h total.
- Weak scaling is nearly perfect (MWMS 1→2 nodes at fixed local batch 20: 792→1518 ex/s = 96 %);
  the recipe-preserving b10 loses to small-per-replica-batch GPU efficiency, not to the network.
  Horovod's fused allreduce recovers most of that (+18 % over MWMS at b10).
- Free single-node observation: 4 tasks × 1 GPU beats 1 task × 4 GPU by ~15 % (792 vs 688 ex/s)
  — four independent input pipelines instead of one. A 1-node Horovod run would likely cut the
  9.5 h production training to ~8 h with zero recipe change.

## Recipe equivalence of 8×b10

`local_batch_size` is per replica. VMIM loss is a per-sample NLL (mean is linear), gradients are
averaged across replicas, the transformer uses LayerNorm only ⇒ 8 × b10 is gradient-equivalent to
the production 4 × b20 at global batch 80. Only the data order / shuffle-buffer layout differs.

## Launch geometry (Clariden)

4 tasks/node × 1 GPU/task, `--gpu-bind=single:1` (there is no in-code GPU pinning);
**horovod needs `srun --mpi=pmix`** (container OpenMPI; without it: `OPAL ERROR: Unreachable in
pmix3x_client.c`, job 2795158). The Perlmutter-era `SlurmClusterResolver` + handcrafted
`TF_CONFIG` machinery in `deep_lss/utils/distribute/tensorflow.py` works unmodified on Clariden
node names.

## Bugs found & fixed (both committed to the working tree, uncommitted)

1. `run_training.py` build-trace: under MWMS the eager `network(...)` trace runs in the
   `/job:localhost` context while in-scope variables live on `/job:worker/.../GPU:0`, which the
   XLA `jit_compile_body` cannot bridge → now routed through `strategy.run` for MWMS (both
   transformer branches). Mirrored/horovod paths untouched.
2. `nets/layers/maps/smoothing.py` kernel cache: all multi-worker tasks raced on
   `smoothing/ind_coo*.npy` in the shared run dir (concurrent `np.save` on Lustre → EIO, killed
   the first 2-node jobs 2795156/57) → per-task temp file + atomic `os.replace`.

## Open item before production 2-node use

- **MWMS input-norm divergence**: the `input_norm` adapt block in `run_training.py` broadcasts
  stats from rank 0 only on the Horovod path; under MWMS every worker computes its own stats from
  its own (unseeded) adapt dataset → slightly different normalization constants per worker.
  Harmless for throughput, wrong for training. Use Horovod, or add a broadcast/seed for MWMS.
- Productionizing = a `local_batch_size: 10` copy of the probe config + a `training.sh` variant
  with the 2-node geometry above, `STRATEGY=horovod`, `--mpi=pmix`. The eval/inference tail is
  unchanged (single node / single GPU steps).
