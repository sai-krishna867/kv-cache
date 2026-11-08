# Adding a cache method or selection policy

## A selection policy (no runner changes)

Selective precision separates *which tokens* from *how they are stored*. A selector is a function in `src/kvstudy/methods/selectors.py`:

```python
def select_my_policy(view: CacheView, cfg) -> list[np.ndarray]:
    """Per layer, a bool mask of shape (kv_heads, tokens); True = high precision."""
```

- `view.keys` / `view.values` are the per-layer context K/V, shape `(1, kv_heads, tokens, head_dim)`, post-RoPE.
- Select exactly `n_selected(view.n_tokens, cfg.high_fraction)` tokens per layer and head; `select_tokens` rejects anything else, which keeps memory accounting exact and padding-free.
- Register it in `SELECTORS` and add its name to `config.SELECTORS`. If it needs a parameter, add an optional field to `SelectiveMethodConfig` and require it only for that selector in `_selector_fields`.

Storage, memory accounting, format verification, the protected window and real packed storage (`convert_real`) then apply unchanged. Add a test in `tests/test_selective_mlx.py` that checks what the policy selects.

## A new method kind

1. A config model in `config.py` with `kind: Literal["<kind>"]`, added to `MethodConfig`; extend `max_bits` if the protected-window check needs it.
2. A class in `methods/<kind>.py` with:
   - `apply(cache, arch) -> MethodReport`: convert the branch in place (simulated), verify the format, return estimated bytes, a label and any method facts for the record's metadata;
   - `estimate_bytes(cache, arch)`: estimated bytes of the cache as it stands, protected window included;
   - `convert_real(cache, arch)`: packed storage for `systems_real` (see `methods/packed.py`).
3. Register it in `METHODS` in `methods/base.py`.
4. Tests: accounting against the formula and `mx.quantize` buffer sizes, branch isolation, format checks, and simulation-vs-real agreement (`tests/test_packed_mlx.py`).

Methods never see which benchmark is running, and benchmark adapters never touch the cache.
