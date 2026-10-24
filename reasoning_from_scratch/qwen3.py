# Adapted from https://github.com/rasbt/reasoning-from-scratch (Apache-2.0).
# Copyright (c) Sebastian Raschka.

"""Download Qwen3 0.6B files, load the tokenizer, and build the model."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import torch
import torch.nn as nn

_FILES = {
    "base": {
        "model": "qwen3-0.6B-base.pth",
        "tokenizer": "tokenizer-base.json",
    },
    "reasoning": {
        "model": "qwen3-0.6B-reasoning.pth",
        "tokenizer": "tokenizer-reasoning.json",
    },
}
_REPO = "rasbt/qwen3-from-scratch"
_BACKUP_ROOT = "https://f001.backblazeb2.com/file/reasoning-from-scratch/qwen3-0.6B"
_CHUNK = 1024 * 1024


class Qwen3Tokenizer:
    """BPE tokenizer for the Qwen3 0.6B base and reasoning checkpoints."""

    _SPECIALS = [
        "<|endoftext|>",
        "<|im_start|>",
        "<|im_end|>",
        "<|object_ref_start|>",
        "<|object_ref_end|>",
        "<|box_start|>",
        "<|box_end|>",
        "<|quad_start|>",
        "<|quad_end|>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<|vision_pad|>",
        "<|image_pad|>",
        "<|video_pad|>",
    ]
    _SPLIT_RE = re.compile(r"(<\|[^>]+?\|>)")

    def __init__(
        self,
        tokenizer_file_path: str | Path = "tokenizer-base.json",
        apply_chat_template: bool = False,
        add_generation_prompt: bool = False,
        add_thinking: bool = False,
    ) -> None:
        """Load a tokenizer JSON produced by ``download_qwen3_small``."""
        from tokenizers import Tokenizer

        self.apply_chat_template = apply_chat_template
        self.add_generation_prompt = add_generation_prompt
        self.add_thinking = add_thinking

        tok_path = Path(tokenizer_file_path)
        if not tok_path.is_file():
            raise FileNotFoundError(f"Tokenizer file '{tok_path}' not found.")

        self._tok = Tokenizer.from_file(str(tok_path))
        self._special_to_id = {token: self._tok.token_to_id(token) for token in self._SPECIALS}

        self.pad_token = "<|endoftext|>"
        self.pad_token_id = self._special_to_id.get(self.pad_token)

        # Match HF behavior: chat model -> <|im_end|>, base model -> <|endoftext|>
        filename = tok_path.name.lower()
        if "base" in filename and "reasoning" not in filename:
            self.eos_token = "<|endoftext|>"
        else:
            self.eos_token = "<|im_end|>"
        self.eos_token_id = self._special_to_id.get(self.eos_token)

    def encode(self, prompt: str, chat_wrapped: bool | None = None) -> list[int]:
        """Turn ``prompt`` into token ids, keeping special tokens intact."""
        if chat_wrapped is None:
            chat_wrapped = self.apply_chat_template

        stripped = prompt.strip()
        if stripped in self._special_to_id and "\n" not in stripped:
            return [self._special_to_id[stripped]]

        if chat_wrapped:
            prompt = self._wrap_chat(prompt)

        ids: list[int] = []
        for part in filter(None, self._SPLIT_RE.split(prompt)):
            if part in self._special_to_id:
                ids.append(self._special_to_id[part])
            else:
                ids.extend(self._tok.encode(part).ids)
        return ids

    def decode(self, token_ids: list[int]) -> str:
        """Turn token ids back into text, including special tokens."""
        return self._tok.decode(token_ids, skip_special_tokens=False)

    def _wrap_chat(self, user_msg: str) -> str:
        """Wrap a user message in the Qwen chat template."""
        text = f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        if self.add_generation_prompt:
            text += "<|im_start|>assistant"
            if self.add_thinking:
                text += "\n"
            else:
                text += "\n<think>\n\n</think>\n\n"
        return text


def download_qwen3_small(
    kind: str = "base",
    tokenizer_only: bool = False,
    out_dir: str | Path = ".",
) -> None:
    """Download the Qwen3 0.6B checkpoint and its tokenizer.

    Args:
        kind: ``"base"`` or ``"reasoning"``.
        tokenizer_only: When True, skip the model weights.
        out_dir: Folder where the files are written.
    """
    if kind not in _FILES:
        raise ValueError("kind must be 'base' or 'reasoning'")

    keys = ("tokenizer",) if tokenizer_only else ("model", "tokenizer")
    for key in keys:
        filename = _FILES[kind][key]
        primary = f"https://huggingface.co/{_REPO}/resolve/main/{filename}"
        _download_file(primary, out_dir, backup_url=f"{_BACKUP_ROOT}/{filename}")


def _download_file(url: str, out_dir: str | Path, backup_url: str | None = None) -> Path:
    """Save ``url`` under ``out_dir``, falling back to ``backup_url`` on failure."""
    folder = Path(out_dir)
    folder.mkdir(parents=True, exist_ok=True)
    filename = Path(url).name
    dest = folder / filename

    errors: list[str] = []
    for candidate in (url, backup_url):
        if candidate is None:
            continue
        try:
            _fetch(candidate, dest)
            return dest
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            errors.append(f"{candidate}: {type(exc).__name__}: {exc}")
            if candidate == url and backup_url:
                print(f"Primary URL failed. Trying backup URL ({backup_url})...")

    raise RuntimeError(f"Failed to download {filename}.\n" + "\n".join(errors))


def _fetch(url: str, dest: Path) -> None:
    """Stream ``url`` to ``dest``, skipping the transfer when the file is complete."""
    request = Request(url, headers={"User-Agent": "small-llm-advanced"})
    with urlopen(request, timeout=30) as response:
        size_remote = int(response.headers.get("Content-Length") or 0)
        if dest.is_file() and size_remote and dest.stat().st_size == size_remote:
            print(f"{dest} already up-to-date")
            return

        downloaded = 0
        partial = dest.with_suffix(dest.suffix + ".part")
        with partial.open("wb") as handle:
            while chunk := response.read(_CHUNK):
                handle.write(chunk)
                downloaded += len(chunk)
                if size_remote:
                    pct = downloaded * 100 // size_remote
                    print(
                        f"\r{dest.name}: {pct:3d}% "
                        f"({downloaded // _CHUNK} MiB / {size_remote // _CHUNK} MiB)",
                        end="",
                        flush=True,
                    )
        if size_remote:
            print()
        partial.replace(dest)


# 0.6 billion parameters
QWEN_CONFIG_06_B = {
    "vocab_size": 151_936,
    "context_length": 40_960,
    "emb_dim": 1024,
    "n_heads": 16,
    "n_layers": 28,
    "hidden_dim": 3072,
    "head_dim": 128,
    "qk_norm": True,
    "n_kv_groups": 8,
    "rope_base": 1_000_000.0,
    "dtype": torch.bfloat16,
}


class Qwen3Model(nn.Module):
    """Qwen3 decoder used by the 0.6B base and reasoning checkpoints."""

    def __init__(self, cfg: dict) -> None:
        """Build the embedding, transformer stack, and output head from ``cfg``."""
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])
        self.trf_blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = RMSNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])

        head_dim = cfg["emb_dim"] // cfg["n_heads"] if cfg["head_dim"] is None else cfg["head_dim"]
        cos, sin = compute_rope_params(
            head_dim=head_dim,
            theta_base=cfg["rope_base"],
            context_length=cfg["context_length"],
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.cfg = cfg
        self.current_pos = 0

    def forward(
        self,
        in_idx: torch.Tensor,
        cache: object | None = None,
    ) -> torch.Tensor:
        """Return next-token logits for ``in_idx``. ``cache`` is optional KV state."""
        x = self.tok_emb(in_idx)
        num_tokens = x.shape[1]
        pos_start = self.current_pos if cache is not None else 0
        pos_end = pos_start + num_tokens
        if pos_end > self.cfg["context_length"]:
            raise ValueError(
                f"Sequence length {pos_end} exceeds the model's context length "
                f"of {self.cfg['context_length']} tokens."
            )

        if cache is not None:
            self.current_pos = pos_end
            mask = torch.triu(
                torch.ones(pos_end, pos_end, device=x.device, dtype=torch.bool),
                diagonal=1,
            )[pos_start:pos_end, :pos_end]
        else:
            mask = torch.triu(
                torch.ones(num_tokens, num_tokens, device=x.device, dtype=torch.bool),
                diagonal=1,
            )
        mask = mask[None, None, :, :]

        for i, block in enumerate(self.trf_blocks):
            blk_cache = cache.get(i) if cache else None
            x, new_blk_cache = block(x, mask, self.cos, self.sin, start_pos=pos_start, cache=blk_cache)
            if cache is not None:
                cache.update(i, new_blk_cache)

        x = self.final_norm(x)
        return self.out_head(x.to(self.cfg["dtype"]))

    def reset_kv_cache(self) -> None:
        """Restart cached decoding at position 0."""
        self.current_pos = 0


class KVCache:
    """Per-layer key and value cache passed to ``Qwen3Model``."""

    def __init__(self, n_layers: int) -> None:
        """Allocate one empty slot per transformer layer."""
        self.cache: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * n_layers

    def get(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Return the cached keys and values for one layer."""
        return self.cache[layer_idx]

    def update(self, layer_idx: int, value: tuple[torch.Tensor, torch.Tensor]) -> None:
        """Store the updated keys and values for one layer."""
        self.cache[layer_idx] = value

    def get_all(self) -> list[tuple[torch.Tensor, torch.Tensor] | None]:
        """Return the cache for every layer."""
        return self.cache

    def reset(self) -> None:
        """Drop every cached key and value."""
        for i in range(len(self.cache)):
            self.cache[i] = None


class TransformerBlock(nn.Module):
    """One Qwen3 layer: grouped-query attention, then a SwiGLU feed-forward."""

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.att = GroupedQueryAttention(
            d_in=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            head_dim=cfg["head_dim"],
            num_kv_groups=cfg["n_kv_groups"],
            qk_norm=cfg["qk_norm"],
            dtype=cfg["dtype"],
        )
        self.ff = FeedForward(cfg)
        self.norm1 = RMSNorm(cfg["emb_dim"], eps=1e-6)
        self.norm2 = RMSNorm(cfg["emb_dim"], eps=1e-6)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        start_pos: int = 0,
        cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Apply the residual attention and feed-forward blocks."""
        shortcut = x
        x = self.norm1(x)
        x, next_cache = self.att(x, mask, cos, sin, start_pos=start_pos, cache=cache)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        return x + shortcut, next_cache


class FeedForward(nn.Module):
    """SwiGLU feed-forward block."""

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.fc1 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc2 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc3 = nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"], bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the gated linear unit and project back to ``emb_dim``."""
        return self.fc3(nn.functional.silu(self.fc1(x)) * self.fc2(x))


class GroupedQueryAttention(nn.Module):
    """Grouped-query attention with optional QK RMSNorm and rotary embeddings."""

    def __init__(
        self,
        d_in: int,
        num_heads: int,
        num_kv_groups: int,
        head_dim: int | None = None,
        qk_norm: bool = False,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if num_heads % num_kv_groups != 0:
            raise ValueError("num_heads must be divisible by num_kv_groups")

        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups

        if head_dim is None:
            if d_in % num_heads != 0:
                raise ValueError("`d_in` must be divisible by `num_heads` if `head_dim` is not set")
            head_dim = d_in // num_heads

        self.head_dim = head_dim
        self.d_out = num_heads * head_dim

        self.W_query = nn.Linear(d_in, self.d_out, bias=False, dtype=dtype)
        self.W_key = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)
        self.W_value = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)
        self.out_proj = nn.Linear(self.d_out, d_in, bias=False, dtype=dtype)

        if qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=1e-6)
            self.k_norm = RMSNorm(head_dim, eps=1e-6)
        else:
            self.q_norm = None
            self.k_norm = None

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        start_pos: int = 0,
        cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Attend over ``x`` and return the projected output plus the updated cache."""
        batch_size, num_tokens, _ = x.shape

        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        queries = queries.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys_new = keys.view(batch_size, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        values_new = values.view(batch_size, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)

        if self.q_norm:
            queries = self.q_norm(queries)
        if self.k_norm:
            keys_new = self.k_norm(keys_new)

        queries = apply_rope(queries, cos, sin, offset=start_pos)
        keys_new = apply_rope(keys_new, cos, sin, offset=start_pos)

        if cache is not None:
            prev_k, prev_v = cache
            keys = torch.cat([prev_k, keys_new], dim=2)
            values = torch.cat([prev_v, values_new], dim=2)
        else:
            keys, values = keys_new, values_new
        next_cache = (keys, values)

        keys = keys.repeat_interleave(self.group_size, dim=1)
        values = values.repeat_interleave(self.group_size, dim=1)

        attn_scores = queries @ keys.transpose(2, 3)
        attn_scores = attn_scores.masked_fill(mask, -torch.inf)
        attn_weights = torch.softmax(attn_scores / self.head_dim**0.5, dim=-1)
        context = (attn_weights @ values).transpose(1, 2).reshape(batch_size, num_tokens, self.d_out)
        return self.out_proj(context), next_cache


def compute_rope_params(
    head_dim: int,
    theta_base: float = 10_000,
    context_length: int = 4096,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute the split-halves RoPE cosine and sine tables."""
    if head_dim % 2 != 0:
        raise ValueError("Embedding dimension must be even")

    inv_freq = 1.0 / (
        theta_base ** (torch.arange(0, head_dim, 2, dtype=dtype)[: (head_dim // 2)].float() / head_dim)
    )
    positions = torch.arange(context_length, dtype=dtype)
    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0)
    angles = torch.cat([angles, angles], dim=1)
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, offset: int = 0) -> torch.Tensor:
    """Rotate ``x`` with the split-halves RoPE layout used by Qwen3."""
    _batch_size, _num_heads, seq_len, head_dim = x.shape
    if head_dim % 2 != 0:
        raise ValueError("Head dimension must be even")

    x1 = x[..., : head_dim // 2]
    x2 = x[..., head_dim // 2 :]
    cos = cos[offset : offset + seq_len, :].unsqueeze(0).unsqueeze(0)
    sin = sin[offset : offset + seq_len, :].unsqueeze(0).unsqueeze(0)
    rotated = torch.cat((-x2, x1), dim=-1)
    return ((x * cos) + (rotated * sin)).to(dtype=x.dtype)


class RMSNorm(nn.Module):
    """Root-mean-square normalization. Qwen3 computes the variance in float32."""

    def __init__(self, emb_dim: int, eps: float = 1e-6, bias: bool = False, qwen3_compatible: bool = True) -> None:
        super().__init__()
        self.eps = eps
        self.qwen3_compatible = qwen3_compatible
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize the last dimension and apply the learned scale."""
        input_dtype = x.dtype
        if self.qwen3_compatible:
            x = x.to(torch.float32)

        variance = x.pow(2).mean(dim=-1, keepdim=True)
        norm_x = x * torch.rsqrt(variance + self.eps) * self.scale
        if self.shift is not None:
            norm_x = norm_x + self.shift
        return norm_x.to(input_dtype)
