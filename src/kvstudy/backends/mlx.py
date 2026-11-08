"""MLX / MLX-LM backend: model loading, prompt rendering, prefill, branching, generation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError

# Fixed system prompt from the experiment design; identical for every run.
SYSTEM_PROMPT = (
    "You are a precise assistant. Answer using only the provided context. "
    "If the answer is a short fact, reply with the fact only, without explanation."
)

PREFILL_STEP_SIZE = 2048


@dataclass(frozen=True)
class Arch:
    num_layers: int
    num_kv_heads: int
    head_dim: int


@dataclass
class LoadedModel:
    model: Any
    tokenizer: Any  # mlx_lm TokenizerWrapper
    snapshot_path: Path
    revision: str
    arch: Arch


@dataclass(frozen=True)
class RenderedPrompt:
    token_ids: list[int]
    boundary: int  # ids[:boundary] is the context prefill, ids[boundary:] the question
    boundary_straddled: bool  # a token crossed the char boundary and went to the question side
    text: str


def local_snapshot(model_cfg: Any) -> Path | None:
    """Path of the pinned snapshot in the local HF cache, or None. Never downloads."""
    try:
        return Path(snapshot_download(model_cfg.id, revision=model_cfg.revision, local_files_only=True))
    except LocalEntryNotFoundError:
        return None


def read_arch(config: dict[str, Any]) -> Arch:
    num_heads = config["num_attention_heads"]
    return Arch(
        num_layers=config["num_hidden_layers"],
        num_kv_heads=config.get("num_key_value_heads", num_heads),
        head_dim=config.get("head_dim") or config["hidden_size"] // num_heads,
    )


def load_model(model_cfg: Any) -> LoadedModel:
    from mlx.utils import tree_flatten
    from mlx_lm import load

    path = local_snapshot(model_cfg)
    if path is None:
        raise FileNotFoundError(
            f"{model_cfg.id}@{model_cfg.revision} is not in the local HF cache; "
            f"download it with: hf download {model_cfg.id} --revision {model_cfg.revision}"
        )
    model, tokenizer = load(str(path))
    dtypes = {v.dtype for _, v in tree_flatten(model.parameters())}
    if dtypes != {mx.bfloat16}:
        raise ValueError(f"expected all weights in bfloat16, found {sorted(map(str, dtypes))}")
    with open(path / "config.json", encoding="utf-8") as f:
        arch = read_arch(json.load(f))
    return LoadedModel(model=model, tokenizer=tokenizer, snapshot_path=path, revision=model_cfg.revision, arch=arch)


def render_prompt(
    tokenizer: Any, system_prompt: str, user_text: str, question_offset: int, answer_prefix: str
) -> RenderedPrompt:
    """Render the full chat prompt once and locate the context/question token boundary.

    The whole prompt is tokenized in one pass; prefilling ids[:boundary] then
    ids[boundary:] therefore reproduces exactly the full-prompt tokenization.
    """
    hf_tok = tokenizer._tokenizer
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_text}]
    text = hf_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    if text.count(user_text) != 1:
        raise ValueError("user text does not appear exactly once in the rendered prompt")
    text += answer_prefix
    boundary_char = text.index(user_text) + question_offset

    enc = hf_tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]
    boundary = next(i for i, (start, _) in enumerate(offsets) if start >= boundary_char)
    straddled = boundary > 0 and offsets[boundary - 1][1] > boundary_char
    if straddled:
        boundary -= 1
    if boundary == 0 or boundary == len(ids):
        raise ValueError("context/question boundary falls at the edge of the prompt")
    return RenderedPrompt(token_ids=ids, boundary=boundary, boundary_straddled=straddled, text=text)


def prefill(model: Any, ids: list[int], step: int = PREFILL_STEP_SIZE) -> list[Any]:
    """Run ids through the model in chunks and return the populated cache."""
    from mlx_lm.models.cache import make_prompt_cache

    cache = make_prompt_cache(model)
    tokens = mx.array(ids)
    for start in range(0, len(ids), step):
        model(tokens[None, start : start + step], cache=cache)
        mx.eval([c.state for c in cache])
    return cache


def branch_cache(cache: list[Any]) -> list[Any]:
    """Independent copy of a KV cache; updates to the branch never reach the source."""
    from mlx_lm.models.cache import KVCache

    branch = []
    for c in cache:
        if type(c) is not KVCache:
            raise TypeError(f"branching supports KVCache only, got {type(c).__name__}")
        b = KVCache()
        b.keys = mx.contiguous(c.keys[..., : c.offset, :])
        b.values = mx.contiguous(c.values[..., : c.offset, :])
        b.offset = c.offset
        branch.append(b)
    mx.eval([b.state for b in branch])
    return branch


def cache_nbytes(cache: list[Any]) -> int:
    """Bytes of key/value data held up to each layer's offset."""
    total = 0
    for c in cache:
        k, v = c.keys[..., : c.offset, :], c.values[..., : c.offset, :]
        total += k.size * k.itemsize + v.size * v.itemsize
    return total


def generate(model: Any, tokenizer: Any, question_ids: list[int], cache: list[Any], max_new_tokens: int) -> tuple[str, int]:
    """Greedy decode after processing question_ids through cache (updated in place)."""
    from mlx_lm.generate import generate_step

    eos = set(tokenizer.eos_token_ids)
    out: list[int] = []
    for token, _ in generate_step(
        mx.array(question_ids),
        model,
        max_tokens=max_new_tokens,
        prompt_cache=cache,
        prefill_step_size=PREFILL_STEP_SIZE,
    ):
        token = int(token)
        if token in eos:
            break
        out.append(token)
    return tokenizer.decode(out), len(out)


def score_continuation(model: Any, cache: list[Any], continuation_ids: list[int], step: int = 256) -> tuple[float, int]:
    """Teacher-force continuation_ids through cache and return (summed NLL in nats, tokens scored).

    Feeds ids[:-1] and scores ids[1:]: each scored token is predicted from a position
    processed through `cache`. ids[0], whose prediction comes from the context
    prefill, is not scored. Log-softmax runs in float32, in chunks of `step`.
    """
    ids = mx.array(continuation_ids)
    total = 0.0
    for start in range(0, len(continuation_ids) - 1, step):
        stop = min(start + step, len(continuation_ids) - 1)
        logits = model(ids[None, start:stop], cache=cache)[0].astype(mx.float32)
        targets = ids[start + 1 : stop + 1]
        nll = mx.logsumexp(logits, axis=-1) - mx.take_along_axis(logits, targets[:, None], axis=-1)[:, 0]
        total += float(nll.sum())
    return total, len(continuation_ids) - 1
