# Nested HEALPix Local-Window Transformer

Notes on the `nested_transformer` network (`config name: nested_transformer`):
a Swin-style hierarchical local-window transformer that operates on the HEALPix
NEST quadtree. Maps-only and maps+Cls variants exist.

Source files:
- `deep_lss/nets/nested_transfomer.py` — the engine (`NestedHierarchicalLocalWindowTransformer`)
- `deep_lss/nets/healpix_transformer.py` — thin HEALPix wrapper (`(B,P,C)` ↔ nested tensor)
- `deep_lss/nets/transformer_networks.py` — tokenizer + smoothing front-end + maps/maps+Cls models
- `deep_lss/apps/run_training.py:567-641` — config → network wiring

Reference geometry used throughout: **nside=512, token_nside=16** ⇒ `L=5` nested
levels, `4^5 = 1024` pixels/token, **N=448 occupied DES-Y3 footprint tokens**
(partial sky — *not* the full-sky `12·16² = 3072`), padding-free.

**As trained by `submissions/clariden/training.sh`** (the production run, `ARCH=transformer`,
`MODEL=t4_cls_dropout`): the maps+Cls variant, config
`configs/transformer/lensing/maps+cls.yaml`. That config is *not* the base-16 reference
baseline used for the tensor-flow walkthrough below — it runs `base_embed_dim=32`,
`growth=double` (final dim 1024), `global_blocks=4`, `local_blocks_per_level=1`, `num_heads=4`,
`window_levels=3`, and — notably — `layerscale_init: null` (**LayerScale off**; stability comes
from the fp32 LayerNorm + cosine warmup + `clip_by_global_norm=1.0` instead). It adds
`map_feature_dim=512`, a 16-bin `asinh_per_feature` Cls branch with a `[512,512,512,512]`
embedding MLP, `input_norm: true`, `jit_compile_body: true`, `precision: bfloat16`, and 0.1
head/embedding dropout. Trains for 120k steps at `local_batch_size=20` on 4×A100 under
`mirrored`.

---

## 1. Pipeline, step by step

`HealpixTransformerNetwork` (`transformer_networks.py:109`) is three stages:

### 1.1 Smoothing
`HealpySmoothing` — the *same* Gaussian front-end the DeepSphere GCNNs use. In
`(B, P, C)`, out `(B, P, C)`, where `P` = footprint pixels (458752 at nside 512)
and `C = in_channels = n_z_bins` (4 tomographic bins for Y3 lensing).

### 1.2 Tokenizer
`HealpixNestedTokenizer` (`transformer_networks.py:42`) — a **non-trainable**
`tf.gather`. It regroups the flat footprint into superpixel-major,
NEST-contiguous blocks: one **top-level token per occupied `token_nside`
superpixel**, each holding its `4^L` fine children in NEST order
(`L = order(nside) − order(token_nside)`). Output `(B, N·4^L, C)`.

- `N` = number of *occupied* superpixels (448 for the DES footprint at
  token_nside=16), built by `np.unique(superpix)`.
- Missing children (only when `token_nside <` the msfm footprint padding
  `n_side_down`) are zero-filled via an appended zero pixel row. **Padding-free**
  ⇔ `token_nside ≥ n_side_down`; the tokenizer's WAR log reports the count.

Emits a warning line on build, e.g.:
```
HealpixNestedTokenizer: nside=512, token_nside=16, levels=5, pixels/token=1024,
tokens(N)=448, num_pixels=458752, zero-padded slots=0 (padding-free)
```

### 1.3 Transformer
`NestedHierarchicalLocalWindowTransformer` (`nested_transfomer.py:329`).
`batch_flat_to_nested` reshapes `(B, N·4^L, C)` → `(B, C, N, 4, 4, …, 4)`
(L fours). Then:

1. **Input projection** `Dense: C → base_embed_dim` (`:414`), per fine pixel.
2. **L hierarchical stages** (`:522`). Each stage =
   `local_blocks_per_level` × `NestedLocalWindowBlock` (`:159`) followed by one
   `NestedPatchMerge4` (`:257`):
   - `NestedLocalWindowBlock`: local self-attention over a window of
     `4^min(window_levels, levels_remaining)` sibling tokens. It flattens the last
     few nested dims into a sequence, runs a standard `TransformerBlock`, and
     reshapes back — **no 2D-image reshape**, the NEST quadtree locality is kept.
   - `NestedPatchMerge4`: concatenates the 4 children of the last nested dim
     (`4·in_dim`) → `LayerNorm` → `Dense → out_dim`. Resolution coarsens ×4, width
     grows per `make_channel_dims`.
3. After L merges: `(B, N, final_dim)`. **`global_blocks` × `TransformerBlock`**
   (`:531`) = full all-to-all self-attention over the N tokens. **This is the only
   globally-mixing stage** (an N×N attention matrix per head).
4. **Mean-pool over N** → `(B, final_dim)` → `LayerNorm` → `Dense` head →
   `(B, num_outputs)` (`:534-542`). For maps-only this vector *is* the summary
   statistic; `num_outputs = n_output` is set by the loss, not the config.

Mental model: **Swin / hierarchical ViT on the HEALPix NEST quadtree** — windowed
local attention + patch-merging at fine scales, one global attention at the coarse
scale.

### Tensor flow (reference geometry, growth=double, base 16)
`channel_dims = [16, 32, 64, 128, 256, 512]` (6 entries; final 512).

```
(B,448,4,4,4,4,4,16)  input proj
 level0: local(win min(3,5)=3 → 64-tok) dim16 ; merge → (B,448,4,4,4,4,32)
 level1: local(win min(3,4)=3 → 64-tok) dim32 ; merge → (B,448,4,4,4,64)
 level2: local(win min(3,3)=3 → 64-tok) dim64 ; merge → (B,448,4,4,128)
 level3: local(win min(3,2)=2 → 16-tok) dim128; merge → (B,448,4,256)
 level4: local(win min(3,1)=1 →  4-tok) dim256; merge → (B,448,512)
 global: 2 × TransformerBlock over 448 tokens at dim 512
 mean-pool → (B,512) → head → (B,num_outputs)
```
Note `window_levels=3` only fully applies at the first three stages; later stages
are capped by the nested dims that remain.

---

## 2. Hyperparameters — params / compute / expressivity

`make_channel_dims(base, L, growth)` (`nested_transfomer.py:6`) builds the width
ladder. With L=5:
- `constant`: `[b,b,b,b,b,b]`
- `double`:   `[b,2b,4b,8b,16b,32b]`
- `full`:     `[b,4b,16b,64b,256b,1024b]`  ← final dim explodes; avoid at token16
- `"128"`:    `[b,b+128,b+256,b+384,b+512,b+640]`  (additive)

| Param | Params | Compute | Expressivity |
|---|---|---|---|
| `base_embed_dim` | quadratic (every stage ∝ width²) | quadratic | primary capacity knob; sets the whole ladder |
| `growth` | sets `final_dim` → dominates total params via global stage | dominates global FLOPs (∝ final_dim²) | coarse-vs-fine capacity allocation |
| `num_heads` | ~none | ~none | attention granularity; **must divide every channel_dim** |
| `window_levels` | ~none (shared block) | strong — attention seq-len `= 4^min(window_levels, levels_left)`, cost ∝ that² at the finest, most-populated stage | local receptive field within a token |
| `local_blocks_per_level` | linear | linear but cheap (small windows) | depth of local mixing |
| `global_blocks` | linear in the *most expensive* block (dim=final_dim) | linear in the dominant cost | depth of global mixing over N tokens |
| `mlp_ratio` | linear (~⅔ of each block) | linear | per-token nonlinearity |

### Rough param budget (reference baseline, base 16 / double / heads 4 / win 3 / local 1 / global 2)
Per `TransformerBlock` ≈ `12·dim²` (attention `4·dim²` + MLP `2·mlp_ratio·dim²`);
per `NestedPatchMerge4` ≈ `8·in_dim²`.
- local blocks (dims 16…256): `12·(16²+32²+64²+128²+256²)` ≈ **1.05M**
- merges (ins 16…256):        `8·(…)`                      ≈ **0.70M**
- global blocks (dim 512, ×2): `2·12·512²`                 ≈ **6.3M**
- **total ≈ 8M**, dominated by the global stage.

So `growth`, `base_embed_dim`, and `global_blocks` are the big param/compute
levers; `window_levels` and `local_blocks_per_level` buy expressivity cheaply.
For scale: `wide` (base 32) ≈ 4× baseline; `deep` (local 2 / global 4) ≈ 2×.

### Compute
- **Global attention** ≈ `O(N²·final_dim)` (attention matrix) + `O(N·final_dim²)`
  (projections). At N=448, dim 512 both terms are ~1e8; ∝ final_dim² as width grows.
- **Local attention** is dominated by the finest stage (458752 fine tokens);
  `window_levels` sets the `seq_len²` factor there.

---

## 3. Constraints (all enforced in code)

1. `nside > token_nside`, both powers of two
   (`healpix_transformer.py:13`, `transformer_networks.py:60`).
2. **`num_heads` divides every `channel_dim`** (`nested_transfomer.py:406`). With
   `double`/`full`/`constant` it suffices that `num_heads | base_embed_dim`; with
   `"128"` you also need `num_heads | 128`.
3. `window_levels ≥ 1` (silently capped at L), `global_blocks ≥ 1`,
   `local_blocks_per_level ≥ 0` (0 = merges only).
4. `dim·mlp_ratio` integer.
5. **`token_nside ≥ n_side_down` to stay padding-free.** token_nside=16 is the
   padding-free floor for this footprint; coarser (e.g. 8) introduces zero-padded
   partial superpixels.
6. **YAML gotcha:** `make_channel_dims` compares the *string* `"128"`, so the
   additive growth must be written `growth: "128"` — an unquoted YAML `128` parses
   as an int and raises `ValueError`.

### Reasonable values (nside=512, token_nside=16)
- `token_nside` = 16 (coarsest padding-free ⇒ fewest global tokens, deepest
  hierarchy; finer 32/64 increase N and global cost)
- `base_embed_dim` ∈ 8–64
- `growth` = `double` (default); `constant` = cheap floor; `full` explodes
  final_dim to `1024·b` at L=5 — avoid at token16
- `num_heads` 4–8
- `window_levels` 2–4 (≤ L=5)
- `local_blocks_per_level` 1–2
- `global_blocks` 2–4

---

## 4. maps+Cls variant

`TransformerMapsPlusCLSNetwork` (`transformer_networks.py:151`) mirrors
`MapsPlusCLSNetwork`: the transformer outputs a `map_feature_dim` vector
(`LayerNorm`'d), the Cls branch is `ClsBinningAndTransformLayer → LayerNorm →
embedding MLP`, and the two are concatenated into a dense regression head. Selected
when the pipeline returns Cls (`return_cls`); config adds `map_feature_dim`,
`cls_n_bins`, `cls_embedding_layers`, `fused_head_layers`, etc.
(`run_training.py:577-611`).

---

## 5. Debug configs

`configs/transformer/lensing/debug/` (all pinned to nside=512 / token_nside=16,
padding-free, N=448):

| Config | Varies vs. baseline | Final dim |
|---|---|---|
| `maps.yaml` | baseline (base 16, double, heads 4, win 3, local 1, global 2) | 512 |
| `maps_tiny.yaml` | smoke-test: base 8, win 2, 1 local / 1 global, mlp 2, 2000 steps | 256 |
| `maps_constant.yaml` | `growth=constant`, base 64 | 64 |
| `maps_growth128.yaml` | `growth="128"` additive | 656 |
| `maps_window1.yaml` | `window_levels=1` (4-sibling local) | 512 |
| `maps_window_full.yaml` | `window_levels=5` (full-superpixel local) | 512 |
| `maps_deep.yaml` | local 2 / global 4 | 512 |
| `maps_wide.yaml` | base 32, heads 8 | 1024 |
| `maps_coarse_tokens.yaml` | token_nside=8 (⚠ below padding-free floor; L=6) | 1024 |
| `maps_cosine_ema.yaml` | optimization only: cosine LR + EMA | 512 |

⚠ The `maps_wide.yaml`/`maps_coarse_tokens.yaml` header comments quote the
full-sky `N = 12·16² = 3072`; the real occupied count is **448**.

---

## 6. Relation to the other `deep_lss.nets` architectures

The nested transformer is one of a **family of learned-summary networks** in
`deep_lss/nets/`, all with the same job: map a HEALPix map `(B, P, C)` to a low-dim
summary `(B, num_outputs)` for the VMIM loss. They share two things and differ in the
**spatial-mixing core**:

- **Shared front-end:** every one of them optionally prepends `HealpySmoothing`
  (the DeepSphere Gaussian smoother) — the transformer uses the *identical* layer
  (`_build_fp32_smoothing`), so the maps entering the mixing core are preprocessed the
  same way across all architectures.
- **Shared back-end:** all end in a small regression head to `out_features`
  (`get_regression_head` for the layer-list nets; a plain `LayerNorm → Dense` here).

The registry lives in `deep_lss/nets/__init__.py`: the four `NETWORKS` entries are built
as DeepSphere `HealpyGCNN` layer-lists, whereas `nested_transformer` is the sole member of
`TRANSFORMER_NETWORKS` — a pre-assembled `tf.keras.Model` passed to `BaseModel` with
`n_side=None` (no GCNN graph is constructed).

### Comparison table

| Network (`__init__` key) | File / class | Spatial-mixing core | Uses sphere graph? | Rotation-equivariant? | Downsampling | Relation to nested transformer |
|---|---|---|---|---|---|---|
| **DeepSphere ResNet** (`resnet`) | `resnet.py` / `ResNetLayers` | Chebyshev **graph** convolutions + residual blocks (Janis' KiDS-1000 fiducial net) | **yes** (PyGSP graph) | approx. (graph conv) | `HealpyPseudoConv` (p=1, ×½ nside) | the incumbent baseline the transformer competes against (`ARCH=deepsphere` in `training.sh`) |
| **Vision Transformer** (`vision_transformer`) | `transformer.py` / `ViTLayers` → `Healpy_ViT` | **flat** ViT: one patch scale (superpixel = `4^p` block), global MHA over all patches | no (NEST order only) | no | optional `HealpyPseudoConv` pre-patching | **direct ancestor.** Same "NEST superpixel = token" idea; the nested transformer generalizes it from a *single* patch scale + global-only attention into a *hierarchy* of windowed-local stages + one global stage |
| **Graph Transformer** (`graph_transformer`) | `transformer.py` / `GTLayers` → `Healpy_Transformer` | graph attention after graph downsampling | **yes** | approx. | `HealpyPseudoConv` | attention-based like the transformer, but attends over graph neighborhoods rather than NEST quadtree windows |
| **1D CNN** (`one_d_conv`) | `one_d_conv.py` / `OneDConvLayers` | 1D conv over the NEST-flattened pixel sequence + residual blocks | no (NEST order only) | no | `HealpyPseudoConv` | shares the "structure enters only through NEST ordering, no graph" philosophy; conv kernels vs. windowed attention for local mixing |
| **Maps+Cls fusion** | `maps_plus_cls_network.py` / `MapsPlusCLSNetwork` | `HealpyGCNN` map branch ⊕ binned-Cls branch → fused head | yes (map branch) | approx. | GCNN | the transformer's `TransformerMapsPlusCLSNetwork` (§4) is a **direct clone** of this two-branch design with the GCNN map branch swapped for the nested transformer |
| **Nested transformer** (`nested_transformer`) | `nested_transfomer.py` / `transformer_networks.py` | Swin-style hierarchical **local-window attention on the NEST quadtree** + one global stage | no (NEST order only) | no | learned `NestedPatchMerge4` (concat-4 + Dense) | — |

### Lineage and borrowings

- **Closest relative: `ViTLayers` / `Healpy_ViT`.** Both tokenize by treating a
  `token_nside` superpixel as a token via NEST ordering, and neither uses the DeepSphere
  graph (spatial structure enters *only* through NEST contiguity → **not** rotation
  equivariant, unlike the GCNN nets). The nested transformer is essentially "`Healpy_ViT`
  made hierarchical": instead of collapsing each superpixel to one token and running
  global attention, it keeps the `4^L` fine children and interleaves **windowed local
  attention** with **patch merges**, running global attention only once at the coarsest
  level.
- **Patch merging ≈ learned `HealpyPseudoConv`.** `NestedPatchMerge4` coarsens resolution
  by exactly the same ×4 (½ nside) factor as the `HealpyPseudoConv` downsampling used by
  the ResNet / GT / 1D-CNN nets, but replaces average-pooling with a learned
  `concat(4 children) → LayerNorm → Dense`. So the transformer's stage structure mirrors
  the "downsample-and-widen" ladder of the graph nets, only with attention instead of
  graph convs between merges.
- **maps+Cls twin.** `TransformerMapsPlusCLSNetwork` reuses `MapsPlusCLSNetwork`'s exact
  recipe — independent `LayerNorm` per branch, a shared `ClsBinningAndTransformLayer` Cls
  branch, concat → regression head — with the only change being the map encoder.
- **External references:** Swin Transformer (Liu et al. 2021, arXiv:2103.14030) for the
  hierarchical local-window + patch-merge design; ViT (Dosovitskiy et al. 2020,
  arXiv:2010.11929) for the underlying tokens-as-patches encoder (also cited in
  `ViTLayers`); LayerScale (Touvron et al. 2021, arXiv:2103.17239) for the residual-branch
  scaling in `TransformerBlock` (though **disabled** in the as-trained config).

### Why it exists (vs. the GCNN baseline)

The DeepSphere `ResNetLayers` is the established fiducial encoder; the nested transformer
is the attention-based challenger. It trades the graph's approximate rotation equivariance
for (i) content-dependent long-range mixing via the global attention stage and (ii) a
much cheaper, XLA-clean forward pass — smoothing aside, the whole tokenizer→transformer
body is `tf.gather` + dense/attention/reshape ops with **no sparse graph Laplacian**, so
it jit-compiles into few fused kernels (`jit_compile_body`), which the graph nets cannot.
