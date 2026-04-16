"""
GPT model: Long-Context extension of gpt.py

Changes vs baseline gpt.py
---------------------------
1. GPTConfig gains five RoPE extension fields:
     rope_base, rope_scaling_type, rope_scaling_factor,
     rope_original_max_seq_len, rope_dynamic_cache

2. _precompute_rotary_embeddings() routes through _get_rope_positions():
     - "none"   : unchanged (exact same as baseline)
     - "linear" : position interpolation — pos / factor
     - "yarn"   : frequency-aware per-dim scaling of inv_freq

3. _maybe_grow_rope_cache(): dynamically doubles the cos/sin cache when
     T0 + T exceeds the current cache length (needed for long-generation).

4. forward(): checks T0 + T (not just T) and calls _maybe_grow_rope_cache
     when rope_dynamic_cache is True.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0, COMPUTE_DTYPE
from nanochat.optim import MuonAdamW, DistMuonAdamW
from nanochat.flash_attention import flash_attn


@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"
    # RoPE extension
    rope_base: float = 100000.0
    rope_scaling_type: str = "none"         # none | linear | yarn
    rope_scaling_factor: float = 1.0        # e.g. 2.0 → 2× context
    rope_original_max_seq_len: int = 2048   # pre-training sequence length
    rope_dynamic_cache: bool = True         # grow cos/sin buffer on-the-fly


def norm(x):
    return F.rms_norm(x, (x.size(-1),))

class Linear(nn.Linear):
    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype))


def has_ve(layer_idx, n_layer):
    return layer_idx % 2 == (n_layer - 1) % 2

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)

class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 12
        self.ve_gate = Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        q = q * 1.2
        k = k * 1.2

        if kv_cache is None:
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        else:
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            y = flash_attn.flash_attn_with_kvcache(
                q, k_cache, v_cache, k=k, v=v,
                cache_seqlens=kv_cache.cache_seqlens, causal=True, window_size=window_size,
            )
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)

        y = y.contiguous().view(B, T, -1)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc   = Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        return self.c_proj(F.relu(self.c_fc(x)).square())


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp  = MLP(config)

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h":   nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        })
        self.lm_head        = Linear(config.n_embd, padded_vocab_size, bias=False)
        self.resid_lambdas  = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas     = nn.Parameter(torch.zeros(config.n_layer))
        self.smear_gate     = Linear(24, 1, bias=False)
        self.smear_lambda   = nn.Parameter(torch.zeros(1))
        self.backout_lambda = nn.Parameter(0.2 * torch.ones(1))
        head_dim = config.n_embd // config.n_head
        kv_dim   = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(padded_vocab_size, kv_dim)
            for i in range(config.n_layer) if has_ve(i, config.n_layer)
        })
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.8)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s * 0.4, s * 0.4)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
        n_layer = self.config.n_layer
        for i in range(n_layer):
            self.resid_lambdas.data[i] = 1.15 - (0.10 * i / max(n_layer - 1, 1))
        for i in range(n_layer):
            self.x0_lambdas.data[i] = 0.20 - (0.15 * i / max(n_layer - 1, 1))
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.uniform_(block.attn.ve_gate.weight, 0.0, 0.02)
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        if COMPUTE_DTYPE != torch.float16:
            self.transformer.wte.to(dtype=COMPUTE_DTYPE)
            for ve in self.value_embeds.values():
                ve.to(dtype=COMPUTE_DTYPE)

    # ------------------------------------------------------------------
    # RoPE helpers
    # ------------------------------------------------------------------

    def _get_rope_positions(self, seq_len, device):
        """Return scaled positions (and optional inv_freq modifier) for RoPE."""
        pos = torch.arange(seq_len, dtype=torch.float32, device=device)
        typ    = self.config.rope_scaling_type
        factor = self.config.rope_scaling_factor
        if typ == "none" or factor == 1.0:
            return pos, None
        if typ == "linear":
            # Position interpolation: compress positions back into trained range
            return pos / factor, None
        if typ == "yarn":
            # YaRN modifies inv_freq, not positions
            return pos, self._yarn_inv_freq_modifier
        raise ValueError(f"Unknown rope_scaling_type: {typ!r}. Choose: none | linear | yarn")

    def _yarn_inv_freq_modifier(self, inv_freq):
        """YaRN: frequency-aware per-dim scaling of inv_freq.

        High-freq dims (short wavelength < orig_len/4): unchanged — good local structure.
        Low-freq dims  (long  wavelength > orig_len/1): scaled like linear interpolation.
        Mid-range: smooth ramp between the two.

        Ref: Peng et al., "YaRN: Efficient Context Window Extension of LLMs" (2023).
        """
        factor   = self.config.rope_scaling_factor
        orig_len = float(self.config.rope_original_max_seq_len)
        # Default YaRN thresholds
        low_freq_factor  = 1.0   # wavelen > orig_len       → fully scaled
        high_freq_factor = 4.0   # wavelen < orig_len / 4   → not scaled
        low_freq_wavelen  = orig_len / low_freq_factor
        high_freq_wavelen = orig_len / high_freq_factor
        wavelen = 2 * torch.pi / inv_freq
        # ramp: 0 at high_freq_wavelen, 1 at low_freq_wavelen
        ramp = ((wavelen - high_freq_wavelen) / (low_freq_wavelen - high_freq_wavelen)).clamp(0.0, 1.0)
        # per-dim scale: 1.0 (no change) → 1/factor (full linear) as ramp goes 0 → 1
        scale = 1.0 - ramp * (1.0 - 1.0 / factor)
        return inv_freq * scale

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=None, device=None):
        if base is None:
            base = self.config.rope_base
        if device is None:
            device = self.transformer.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        pos, inv_freq_fn = self._get_rope_positions(seq_len, device)
        if inv_freq_fn is not None:
            inv_freq = inv_freq_fn(inv_freq)
        freqs = torch.outer(pos, inv_freq)
        cos, sin = freqs.cos().to(COMPUTE_DTYPE), freqs.sin().to(COMPUTE_DTYPE)
        return cos[None, :, None, :], sin[None, :, None, :]

    @torch.no_grad()
    def _maybe_grow_rope_cache(self, needed_seq_len):
        """Grow cos/sin buffer to cover `needed_seq_len`, doubling until large enough."""
        if needed_seq_len <= self.cos.size(1):
            return
        new_len = self.cos.size(1)
        while new_len < needed_seq_len:
            new_len *= 2
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(new_len, head_dim, device=self.cos.device)
        self.cos = cos
        self.sin = sin
        self.rotary_seq_len = new_len

    # ------------------------------------------------------------------

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern), f"Invalid window_pattern: {pattern}"
        long_window  = config.sequence_len
        short_window = -(-long_window // 4 // 128) * 128
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
        window_sizes = [char_to_window[pattern[i % len(pattern)]] for i in range(config.n_layer)]
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def get_device(self):
        return self.transformer.wte.weight.device

    def estimate_flops(self):
        nparams = sum(p.numel() for p in self.parameters())
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        nparams_exclude = (
            self.transformer.wte.weight.numel() + value_embeds_numel +
            self.resid_lambdas.numel() + self.x0_lambdas.numel() +
            self.smear_gate.weight.numel() + self.smear_lambda.numel() + self.backout_lambda.numel()
        )
        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        attn_flops = sum(12 * h * q * (t if w[0] < 0 else min(w[0], t)) for w in self.window_sizes)
        return 6 * (nparams - nparams_exclude) + attn_flops

    def num_scaling_params(self):
        wte                  = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds         = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head              = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = (self.resid_lambdas.numel() + self.x0_lambdas.numel() +
                   self.smear_gate.weight.numel() + self.smear_lambda.numel() + self.backout_lambda.numel())
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        assert total == sum(p.numel() for p in self.parameters()), "Parameter count mismatch"
        return {'wte': wte, 'value_embeds': value_embeds, 'lm_head': lm_head,
                'transformer_matrices': transformer_matrices, 'scalars': scalars, 'total': total}

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, scalar_lr=0.5):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()
        matrix_params       = list(self.transformer.h.parameters())
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params    = list(self.transformer.wte.parameters())
        lm_head_params      = list(self.lm_head.parameters())
        resid_params        = [self.resid_lambdas]
        x0_params           = [self.x0_lambdas]
        smear_params        = [self.smear_gate.weight, self.smear_lambda, self.backout_lambda]
        assert len(list(self.parameters())) == (
            len(matrix_params) + len(embedding_params) + len(lm_head_params) +
            len(value_embeds_params) + len(resid_params) + len(x0_params) + len(smear_params)
        )
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")
        param_groups = [
            dict(kind='adamw', params=lm_head_params,      lr=unembedding_lr * dmodel_lr_scale, betas=(0.8, 0.96),  eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=embedding_params,    lr=embedding_lr   * dmodel_lr_scale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr   * dmodel_lr_scale * 0.5, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=resid_params,   lr=scalar_lr * 0.01, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.05),
            dict(kind='adamw', params=x0_params,      lr=scalar_lr,        betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=smear_params,   lr=0.2,              betas=(0.8, 0.95),  eps=1e-10, weight_decay=0.0),
        ]
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=weight_decay,
            ))
        Factory = DistMuonAdamW if ddp else MuonAdamW
        optimizer = Factory(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        B, T = idx.size()
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        needed = T0 + T

        # Grow or assert the RoPE cache covers [T0, T0+T)
        if self.config.rope_dynamic_cache:
            self._maybe_grow_rope_cache(needed)
        else:
            assert needed <= self.cos.size(1), (
                f"Sequence position {needed} exceeds RoPE cache {self.cos.size(1)}. "
                f"Set rope_dynamic_cache=True or increase sequence_len."
            )
        assert idx.device == self.cos.device
        assert self.cos.dtype == COMPUTE_DTYPE

        cos_sin = self.cos[:, T0:needed], self.sin[:, T0:needed]

        x = self.transformer.wte(idx).to(COMPUTE_DTYPE)
        x = norm(x)

        if kv_cache is None:
            assert T > 1
            gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
            x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
        else:
            x_pre_smear = kv_cache.prev_embedding
            kv_cache.prev_embedding = x[:, -1:, :]
            if T > 1:
                gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
                x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
            elif x_pre_smear is not None:
                gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, :, :24]))
                x = x + gate * x_pre_smear

        x0 = x
        n_layer = self.config.n_layer
        backout_layer = n_layer // 2
        x_backout = None
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx).to(x.dtype) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[i], kv_cache)
            if i == backout_layer:
                x_backout = x
        if x_backout is not None:
            x = x - self.backout_lambda.to(x.dtype) * x_backout
        x = norm(x)

        softcap = 15
        logits = self.lm_head(x)
        logits = logits[..., :self.config.vocab_size].float()
        logits = softcap * torch.tanh(logits / softcap)

        if targets is not None:
            return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                   ignore_index=-1, reduction=loss_reduction)
        return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        for _ in range(max_tokens):
            logits = self.forward(ids)[:, -1, :]
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                next_ids = torch.multinomial(F.softmax(logits / temperature, dim=-1), 1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            yield next_ids.item()
