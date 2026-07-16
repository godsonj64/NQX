# Migration from NanoQuant

Existing NanoQuant commands continue to work. The `nanoquant` and `dbf` ADMM
types are retained. To enable the enhanced path, set:

```text
--admm_type nqx
```

The balanced profile is the default and writes `scale_mid` for each quantized
linear layer. When loading a checkpoint, its NanoQuant-X configuration must be
available so the model creates that parameter before state loading.

For strict compatibility with the original two-scale layout, use:

```text
--admm_type nqx --nqx_rank_scale false
```

## Version 0.3 behavior

The production NQX path now optimizes the BF16 scale values stored by
`NanoQuantLinear`; disable this only for an ablation with:

```text
--nqx_storage_aware false
```

The portable reference path reclaims unused lanes in the final uint32 rank
word. For example, requested rank 24 becomes effective rank 32 at the same
strict packed cost. Artifacts record both `requested_rank` and
`effective_rank`. Use `--no-reclaim-padding` to reproduce the 0.2 rank exactly.

Portable `QuantizedMatrix.matmul()` now lazily prepares scale-fused factors.
Pass `prepared=False` for the former zero-cache, one-shot behavior, or call
`clear_runtime_cache()` after manually mutating factor arrays.

Legacy pickle checkpoints are no longer loaded through an automatic unsafe
fallback. A trusted legacy checkpoint should be loaded once in an isolated
environment and resaved as SafeTensors.
