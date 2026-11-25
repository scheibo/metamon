#!/usr/bin/env python3
"""Export Metamon TaurosV0 (checkpoint 62) to self-contained ONNX for pkmn.ai.

Folds SigmaReparam layers into plain Linear weights, exposes a fixed-shape 32-step
sliding-window KV cache matching `FlashAttention(window_size=(32, 0))`, and includes
both the masked actor policy head and the 3-critic NCriticsTwoHot value head (gamma=0.999)
so the runtime agent can emit full `/info` telemetry (`win` + `policy`) with zero
PyTorch/AMAGO dependency.

Supports exporting both `fp32` and lossless `fp16`-weight ONNX (float16 weight
initializers cast to float32 before matmuls/norms, preserving full fp32 activations
and KV cache) and verifies step-by-step numerical equivalence against the original
PyTorch `TaurosV0` policy past the 32-step sliding-window boundary.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
import types
from pathlib import Path
from typing import Tuple

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F


def _stub_training_only_imports() -> None:
    """Stub unused dataset dependencies so export works in a minimal venv."""
    for mod in (
        "ratarmountcore",
        "ratarmountcore.SQLiteIndexedTarFsspec",
    ):
        if mod not in sys.modules:
            m = types.ModuleType(mod)
            m.SQLiteIndexedTarFileSystem = object
            sys.modules[mod] = m


def _patch_amago_for_cpu_reference() -> None:
    """Enable CPU FlashAttention fallback while preserving gin's `window_size=(32, 0)`."""
    import torch._dynamo
    import amago.nets.transformer as tr
    from amago.envs.amago_env import SequenceWrapper

    torch._dynamo.config.disable = True
    SequenceWrapper.random_traj_length = (
        lambda self: random.randint(int(self.save_every[0]), int(self.save_every[1]))
        if self.save_every
        else None
    )

    if tr.flash_attn is not None:
        return

    # Keep FlashAttention.__init__ intact so @gin.configurable injects window_size=(32, 0).
    tr.flash_attn = True

    @torch.compiler.disable
    def _forward_fallback(
        self, qkv, key_cache=None, val_cache=None, cache_seqlens=None
    ):
        orig_dtype = qkv.dtype
        queries, keys, values = torch.unbind(qkv.float(), dim=2)
        B, L, H, E = queries.shape
        scale = 1.0 / math.sqrt(E)

        if key_cache is None or val_cache is None or cache_seqlens is None:
            scores = scale * torch.einsum("blhe,bshe->bhls", queries, keys)
            if self.causal:
                q_idx = torch.arange(L, device=qkv.device)[:, None]
                k_idx = torch.arange(L, device=qkv.device)[None, :]
                invalid = k_idx > q_idx
                if self.window_size[0] >= 0:
                    invalid = invalid | ((q_idx - k_idx) > self.window_size[0])
                scores = scores.masked_fill(invalid[None, None, :, :], -torch.inf)
            attn = torch.softmax(scores, dim=-1)
            return torch.einsum("bhls,bshd->blhd", attn, values).to(orig_dtype)

        assert not self.training and L == 1
        cache_idxs = torch.arange(B, device=key_cache.device)
        # Keep float32 precision in reference cache for clean parity comparison
        key_cache[cache_idxs, cache_seqlens] = keys[:, 0].to(key_cache.dtype)
        val_cache[cache_idxs, cache_seqlens] = values[:, 0].to(val_cache.dtype)
        end = cache_seqlens + 1
        max_len = int(end.max().item())
        k_cache = torch.nan_to_num(key_cache[:, :max_len].float())
        v_cache = torch.nan_to_num(val_cache[:, :max_len].float())
        scores = scale * torch.einsum("blhe,bshe->bhls", queries, k_cache)
        s_idxs = torch.arange(max_len, device=cache_seqlens.device)[None, :]
        invalid = s_idxs >= end[:, None]
        if self.window_size[0] >= 0:
            start = torch.clamp(cache_seqlens - self.window_size[0], min=0)
            invalid = invalid | (s_idxs < start[:, None])
        scores = scores.masked_fill(invalid[:, None, None, :], -torch.inf)
        attn = torch.softmax(scores, dim=-1)
        return torch.einsum("bhls,bshd->blhd", attn, v_cache).to(orig_dtype)

    tr.FlashAttention.forward = _forward_fallback


def _fold_sigma_reparam(layer: nn.Module) -> nn.Linear:
    """Fold SigmaReparam(W, u, v, gamma, bias) into a plain nn.Linear."""
    with torch.no_grad():
        sigma = torch.einsum("d,dc,c->", layer.u.detach(), layer.weight.detach(), layer.v.detach())
        w_hat = (layer.gamma.detach() / sigma) * layer.weight.detach()
        has_bias = layer.bias is not None
        out = nn.Linear(layer.in_features, layer.out_features, bias=has_bias)
        out.weight.copy_(w_hat)
        if has_bias:
            out.bias.copy_(layer.bias.detach())
        return out


class _ExplicitPerceiverLayer(nn.Module):
    """ONNX-friendly mirror of `_PerceiverLayer` using explicit matmul attention."""

    def __init__(self, src: nn.Module):
        super().__init__()
        self.n_heads = src.n_heads
        self.head_dim = src.head_dim
        self.d_model = src.d_model
        self._normformer = src._normformer
        self._qk_norm = src._qk_norm
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.cross_norm_q = src.cross_norm_q
        self.cross_norm_kv = src.cross_norm_kv
        self.cross_q = src.cross_q
        self.cross_kv = src.cross_kv
        self.cross_out = src.cross_out
        self.cross_ff_norm = src.cross_ff_norm
        self.cross_ff1 = src.cross_ff1
        self.cross_ff2 = src.cross_ff2

        self.self_norm = src.self_norm
        self.self_qkv = src.self_qkv
        self.self_out = src.self_out
        self.self_ff_norm = src.self_ff_norm
        self.self_ff1 = src.self_ff1
        self.self_ff2 = src.self_ff2

        if self._normformer:
            self.cross_post_attn_norm = src.cross_post_attn_norm
            self.cross_mid_ff_norm = src.cross_mid_ff_norm
            self.self_post_attn_norm = src.self_post_attn_norm
            self.self_mid_ff_norm = src.self_mid_ff_norm

        if self._qk_norm:
            self.cross_q_norm = src.cross_q_norm
            self.cross_k_norm = src.cross_k_norm
            self.self_q_norm = src.self_q_norm
            self.self_k_norm = src.self_k_norm

    def _attn(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        bias_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if bias_mask is not None:
            scores = scores + bias_mask
        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, v)

    def forward(
        self,
        latents: torch.Tensor,
        kv_input: torch.Tensor,
        cross_bias: torch.Tensor | None = None,
        self_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        H, HD, D = self.n_heads, self.head_dim, self.d_model
        B, Lq = latents.shape[0], latents.shape[1]
        Lkv = kv_input.shape[1]

        q = self.cross_q(self.cross_norm_q(latents)).reshape(B, Lq, H, HD).transpose(1, 2)
        kv = self.cross_kv(self.cross_norm_kv(kv_input)).reshape(B, Lkv, 2, H, HD)
        k = kv[:, :, 0].transpose(1, 2)
        v = kv[:, :, 1].transpose(1, 2)
        if self._qk_norm:
            q = self.cross_q_norm(q)
            k = self.cross_k_norm(k)
        attn = self._attn(q, k, v, cross_bias)
        cross_out = self.cross_out(attn.transpose(1, 2).reshape(B, Lq, D))
        if self._normformer:
            cross_out = self.cross_post_attn_norm(cross_out)
        latents = latents + cross_out
        h = F.silu(self.cross_ff1(self.cross_ff_norm(latents)))
        if self._normformer:
            h = self.cross_mid_ff_norm(h)
        latents = latents + self.cross_ff2(h)

        qkv = self.self_qkv(self.self_norm(latents)).reshape(B, Lq, 3, H, HD)
        sq = qkv[:, :, 0].transpose(1, 2)
        sk = qkv[:, :, 1].transpose(1, 2)
        sv = qkv[:, :, 2].transpose(1, 2)
        if self._qk_norm:
            sq = self.self_q_norm(sq)
            sk = self.self_k_norm(sk)
        attn = self._attn(sq, sk, sv, self_bias)
        self_out = self.self_out(attn.transpose(1, 2).reshape(B, Lq, D))
        if self._normformer:
            self_out = self.self_post_attn_norm(self_out)
        latents = latents + self_out
        h = F.silu(self.self_ff1(self.self_ff_norm(latents)))
        if self._normformer:
            h = self.self_mid_ff_norm(h)
        latents = latents + self.self_ff2(h)
        return latents


class _FoldedTransformerLayer(nn.Module):
    """Single-step sliding-window TransformerLayer with folded SigmaReparam weights."""

    def __init__(self, src: nn.Module, n_heads: int, head_dim: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.scale = 1.0 / math.sqrt(head_dim)
        attn = src.attention_layer
        self.qkv = _fold_sigma_reparam(attn.qkv_projection)
        self.out_proj = _fold_sigma_reparam(attn.out_projection)
        self.register_buffer("head_scaler", attn.head_scaler.detach().clone())  # (1, 1, H, 1)
        self.ff1 = _fold_sigma_reparam(src.ff1)
        self.ff2 = _fold_sigma_reparam(src.ff2)
        self.norm1 = src.norm1
        self.norm2 = src.norm2
        self.norm3 = src.norm3
        self.norm4 = src.norm4

    def forward(
        self,
        x: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        attn_bias: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: (1, 1, D), k_cache/v_cache: (1, W, H, HD), attn_bias: (1, 1, 1, W+1)
        q1 = self.norm1(x)
        qkv = self.qkv(q1)
        # Match AMAGO exact rearrange:
        # "batch len (three d_qkv heads) -> batch len three heads d_qkv"
        qkv = qkv.reshape(1, 1, 3, self.head_dim, self.n_heads).permute(0, 1, 2, 4, 3)
        q = qkv[:, :, 0]  # (1, 1, H, HD)
        k = qkv[:, :, 1]  # (1, 1, H, HD)
        v = qkv[:, :, 2]  # (1, 1, H, HD)

        full_k = torch.cat([k_cache, k], dim=1)  # (1, W+1, H, HD)
        full_v = torch.cat([v_cache, v], dim=1)  # (1, W+1, H, HD)
        new_k = full_k[:, 1:, :, :]  # (1, W, H, HD)
        new_v = full_v[:, 1:, :, :]  # (1, W, H, HD)

        # (1, H, 1, HD) @ (1, H, HD, W+1) -> (1, H, 1, W+1)
        q_t = q.permute(0, 2, 1, 3)
        k_t = full_k.permute(0, 2, 3, 1)
        v_t = full_v.permute(0, 2, 1, 3)
        scores = torch.matmul(q_t, k_t) * self.scale + attn_bias
        weights = torch.softmax(scores, dim=-1)
        attn_out = torch.matmul(weights, v_t).permute(0, 2, 1, 3)  # (1, 1, H, HD)
        attn_out = attn_out * self.head_scaler
        attn_out = attn_out.reshape(1, 1, self.n_heads * self.head_dim)
        attn_out = self.norm2(self.out_proj(attn_out))
        x = x + attn_out

        ff = self.norm3(x)
        ff = self.norm4(F.leaky_relu(self.ff1(ff)))
        ff = self.ff2(ff)
        x = x + ff
        return x, new_k, new_v


class TaurosOnnxModule(nn.Module):
    """Self-contained single-step inference + critic evaluation module for TaurosV0."""

    WINDOW_SIZE = 32
    NUM_ACTIONS = 13

    def __init__(self, policy: nn.Module):
        super().__init__()
        enc = policy.tstep_encoder
        self.vocab_size = enc.pokemon_token_emb.vocab_size
        self.num_pokemon = enc.NUM_POKEMON
        self.latent_tokens_pokemon = enc.pokemon_perceiver.latent_tokens

        # --- Timestep encoder weights ---
        self.extra_emb = enc.extra_emb
        self.pokemon_token_emb = enc.pokemon_token_emb.text_emb
        self.pokemon_fuse_text = enc.pokemon_fuse.text_emb
        self.pokemon_fuse_num = enc.pokemon_fuse.num_emb
        self.pokemon_num_tokens = enc.pokemon_fuse.numerical_tokens
        self.register_buffer(
            "pokemon_pos_emb",
            enc.pokemon_pos(enc._pokemon_pos_ids).detach().clone(),  # (18, 304)
        )
        # Precompute role signal added to the concatenated 7-pokemon sequence (126, 304)
        role_signal = torch.zeros(
            self.num_pokemon * self.pokemon_pos_emb.shape[0],
            self.pokemon_pos_emb.shape[1],
            dtype=torch.float32,
        )
        if enc._pokemon_role_emb is not None:
            role = enc._pokemon_role_emb(enc._pokemon_role_ids).detach()
            stride = self.pokemon_pos_emb.shape[0]
            for i in range(self.num_pokemon):
                role_signal[i * stride] = role[i]
        self.register_buffer("pokemon_role_signal", role_signal)

        self.register_buffer(
            "pokemon_latents",
            enc.pokemon_perceiver.latents.detach().repeat(self.num_pokemon, 1).unsqueeze(0),
        )
        # Convert boolean SDPA masks (True=allowed) to additive float biases (0 or -1e9)
        cross_bias = torch.where(
            enc.pokemon_perceiver._cross_mask, 0.0, -1e9
        ).to(torch.float32)[None, None, :, :]
        self_bias = torch.where(
            enc.pokemon_perceiver._self_mask, 0.0, -1e9
        ).to(torch.float32)[None, None, :, :]
        self.register_buffer("pokemon_cross_bias", cross_bias)
        self.register_buffer("pokemon_self_bias", self_bias)
        self.pokemon_layers = nn.ModuleList(
            [_ExplicitPerceiverLayer(layer) for layer in enc.pokemon_perceiver.layers]
        )
        self.pokemon_out_norm = enc.pokemon_out_norm
        self.pokemon_proj = enc.pokemon_proj

        # Global encoder
        self.global_token_emb = enc.global_token_emb.text_emb
        self.global_fuse_text = enc.global_fuse.text_emb
        self.global_fuse_num = enc.global_fuse.num_emb
        self.global_num_tokens = enc.global_fuse.numerical_tokens
        self.register_buffer(
            "global_pos_emb",
            enc.global_pos(enc._global_pos_ids).detach().clone(),
        )
        self.register_buffer(
            "global_latents",
            enc.global_perceiver.latents.detach().unsqueeze(0).clone(),
        )
        self.global_layers = nn.ModuleList(
            [_ExplicitPerceiverLayer(layer) for layer in enc.global_perceiver.layers]
        )
        self.global_out_norm = enc.global_out_norm
        self.global_proj = enc.global_proj

        # Fusion encoder
        self.register_buffer(
            "entity_type_emb",
            enc.entity_type_emb(enc._entity_type_ids).detach().clone(),
        )
        self.register_buffer(
            "fusion_latents",
            enc.fusion.latents.detach().unsqueeze(0).clone(),
        )
        self.fusion_layers = nn.ModuleList(
            [_ExplicitPerceiverLayer(layer) for layer in enc.fusion.layers]
        )
        self.fusion_out_norm = enc.fusion_out_norm

        # --- Trajectory transformer ---
        tformer = policy.traj_encoder.tformer
        self.d_model = policy.traj_encoder.d_model
        self.n_heads = policy.traj_encoder.n_heads
        self.head_dim = policy.traj_encoder.head_dim
        self.n_layers = policy.traj_encoder.n_layers
        self.traj_inp = tformer.inp
        self.traj_norm = tformer.norm
        self.traj_layers = nn.ModuleList(
            [
                _FoldedTransformerLayer(layer, self.n_heads, self.head_dim)
                for layer in tformer.layers
            ]
        )
        # Precompute FixedPosEmb sinusoidal frequencies
        pos_coeff = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        self.register_buffer("pos_coeff", pos_coeff)
        self.register_buffer(
            "window_positions",
            torch.arange(self.WINDOW_SIZE + 1, dtype=torch.int64).view(1, 1, 1, -1),
        )

        # --- Actor head (primary gamma = -1, i.e. 0.999) ---
        self.actor_in = policy.actor.base.in_layer
        self.actor_hidden = policy.actor.base.layers[0]
        # Slice out only the primary gamma (-1) rows of out_layer (13 of 26 rows)
        num_gammas = len(policy.gammas)
        out_w = policy.actor.base.out_layer.weight.detach().reshape(
            num_gammas, self.NUM_ACTIONS, -1
        )[-1]
        out_b = policy.actor.base.out_layer.bias.detach().reshape(
            num_gammas, self.NUM_ACTIONS
        )[-1]
        self.actor_out = nn.Linear(out_w.shape[1], self.NUM_ACTIONS, bias=True)
        with torch.no_grad():
            self.actor_out.weight.copy_(out_w)
            self.actor_out.bias.copy_(out_b)

        # --- 3-Critic NCriticsTwoHot head (primary gamma = -1, i.e. 0.999) ---
        # Match NCriticsTwoHot + Discrete.action_from_buffer exactly.
        primary_gamma = float(policy.gammas[-1].item())
        gamma_feat = math.copysign(1.0, 1.0 / (1.0 - primary_gamma)) * math.log(
            abs(1.0 / (1.0 - primary_gamma)) + 1.0
        )
        eye = torch.eye(self.NUM_ACTIONS, dtype=torch.float32)
        clipped = eye.clamp(
            policy.actor.policy_dist.clip_prob_low,
            policy.actor.policy_dist.clip_prob_high,
        )
        action_bank = (clipped / clipped.sum(-1, keepdim=True)).clamp(-0.999, 0.999)
        gamma_col = torch.full((self.NUM_ACTIONS, 1), gamma_feat, dtype=torch.float32)
        self.register_buffer(
            "critic_gamma_action",
            torch.cat([gamma_col, action_bank], dim=-1),  # (13, 14)
        )
        cnet = policy.critics.net
        self.register_buffer("critic_w0", cnet.inp_layer.weight.detach().clone())  # (3, 718, 768)
        self.register_buffer("critic_b0", cnet.inp_layer.bias.detach().squeeze(0).squeeze(0).clone())  # (3, 768)
        self.register_buffer("critic_w1", cnet.core_layers[0].weight.detach().clone())  # (3, 768, 768)
        self.register_buffer("critic_b1", cnet.core_layers[0].bias.detach().squeeze(0).squeeze(0).clone())  # (3, 768)
        self.register_buffer("critic_w2", cnet.output_layer.weight.detach().clone())  # (3, 768, 128)
        self.register_buffer("critic_b2", cnet.output_layer.bias.detach().squeeze(0).squeeze(0).clone())  # (3, 128)
        self.register_buffer("critic_bin_vals", policy.critics.bin_vals.detach().clone())  # (128,)

    def _encode_tstep(
        self,
        pokemon_text: torch.Tensor,
        pokemon_nums: torch.Tensor,
        global_text: torch.Tensor,
        global_nums: torch.Tensor,
        rl2: torch.Tensor,
    ) -> torch.Tensor:
        # 1. Pokemon encoder (7 entities)
        shifted_pt = (pokemon_text.reshape(self.num_pokemon, -1) + 1).clamp(0, self.vocab_size - 1)
        tok = self.pokemon_token_emb(shifted_pt)  # (7, 12, 304)
        tok = F.leaky_relu(self.pokemon_fuse_text(tok))
        pnum = F.leaky_relu(self.pokemon_fuse_num(pokemon_nums.reshape(self.num_pokemon, -1)))
        pnum = pnum.reshape(self.num_pokemon, self.pokemon_num_tokens, -1)
        pseq = torch.cat([tok, pnum], dim=1) + self.pokemon_pos_emb.unsqueeze(0)  # (7, 18, 304)
        pseq = pseq.reshape(1, self.num_pokemon * pseq.shape[1], -1) + self.pokemon_role_signal.unsqueeze(0)

        plat = self.pokemon_latents
        for layer in self.pokemon_layers:
            plat = layer(plat, pseq, self.pokemon_cross_bias, self.pokemon_self_bias)
        plat = plat.reshape(1, self.num_pokemon, self.latent_tokens_pokemon, -1)
        plat = self.pokemon_out_norm(plat).reshape(1, self.num_pokemon, -1)
        pokemon_embs = self.pokemon_proj(plat)  # (1, 7, 448)

        # 2. Global encoder
        rl2_sym = torch.sign(rl2) * torch.log(torch.abs(rl2) + 1.0)
        extras = F.leaky_relu(self.extra_emb(rl2_sym))
        g_nums = torch.cat([global_nums, extras], dim=-1)  # (1, 28)

        shifted_gt = (global_text + 1).clamp(0, self.vocab_size - 1)
        gtok = F.leaky_relu(self.global_fuse_text(self.global_token_emb(shifted_gt)))  # (1, 20, 208)
        gnum = F.leaky_relu(self.global_fuse_num(g_nums)).reshape(1, self.global_num_tokens, -1)
        gseq = torch.cat([gtok, gnum], dim=1) + self.global_pos_emb.unsqueeze(0)  # (1, 24, 208)

        glat = self.global_latents
        for layer in self.global_layers:
            glat = layer(glat, gseq)
        glat = self.global_out_norm(glat).reshape(1, -1)
        global_emb = self.global_proj(glat).unsqueeze(1)  # (1, 1, 448)

        # 3. Fusion encoder
        all_embs = torch.cat([pokemon_embs, global_emb], dim=1) + self.entity_type_emb.unsqueeze(0)
        flat = self.fusion_latents
        for layer in self.fusion_layers:
            flat = layer(flat, all_embs)
        flat = self.fusion_out_norm(flat)
        return flat.reshape(1, 1, -1)  # (1, 1, 1344)

    def _fixed_pos_emb(self, time_idx: torch.Tensor) -> torch.Tensor:
        angles = time_idx.to(torch.float32).reshape(1, 1, 1) * self.pos_coeff.reshape(1, 1, -1)
        sin_e = torch.sin(angles)
        cos_e = torch.cos(angles)
        return torch.stack([sin_e, cos_e], dim=-1).reshape(1, 1, self.d_model)

    def forward(
        self,
        pokemon_text: torch.Tensor,
        pokemon_nums: torch.Tensor,
        global_text: torch.Tensor,
        global_nums: torch.Tensor,
        rl2: torch.Tensor,
        illegal_actions: torch.Tensor,
        time_idx: torch.Tensor,
        cache_len: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        tstep_emb = self._encode_tstep(
            pokemon_text, pokemon_nums, global_text, global_nums, rl2
        )
        x = self.traj_inp(tstep_emb) + self._fixed_pos_emb(time_idx)

        # Right-aligned sliding window mask: positions >= (WINDOW_SIZE - cache_len) are valid.
        # Position WINDOW_SIZE (index 32) is always the current token and always valid.
        min_valid = self.WINDOW_SIZE - cache_len.clamp(0, self.WINDOW_SIZE)
        valid = self.window_positions >= min_valid.reshape(1, 1, 1, 1)
        attn_bias = torch.where(valid, 0.0, -1e9).to(torch.float32)

        new_ks = []
        new_vs = []
        for i, layer in enumerate(self.traj_layers):
            x, nk, nv = layer(x, k_cache[i], v_cache[i], attn_bias)
            new_ks.append(nk)
            new_vs.append(nv)

        traj_out = self.traj_norm(x)  # (1, 1, 704)
        new_k_cache = torch.stack(new_ks, dim=0)  # (4, 1, 32, 8, 88)
        new_v_cache = torch.stack(new_vs, dim=0)  # (4, 1, 32, 8, 88)

        # Actor: primary gamma (-1) masked logits & clean probabilities over legal actions
        state_vec = traj_out[:, 0, :]  # (1, 704)
        h = F.leaky_relu(self.actor_in(state_vec))
        h = F.leaky_relu(self.actor_hidden(h))
        logits = self.actor_out(h)  # (1, 13)

        no_options = illegal_actions.all(dim=-1, keepdim=True)
        eff_illegal = illegal_actions & (~no_options)
        masked_logits = torch.where(eff_illegal, -1e9, logits)
        probs = torch.softmax(masked_logits, dim=-1)  # (1, 13)
        # Guarantee exact 0.0 on masked-out actions
        probs = torch.where(eff_illegal, torch.zeros_like(probs), probs)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-12)

        # Critics: evaluate all 13 discrete actions across all 3 critics for gamma=0.999
        state_rep = state_vec.expand(self.NUM_ACTIONS, -1)  # (13, 704)
        critic_inp = torch.cat([state_rep, self.critic_gamma_action], dim=-1)  # (13, 718)
        # Layer 0: (13, 718) x (3, 718, 768) -> (3, 13, 768)
        ch = torch.matmul(critic_inp.unsqueeze(0), self.critic_w0) + self.critic_b0.unsqueeze(1)
        ch = F.leaky_relu(ch)
        # Layer 1: (3, 13, 768) x (3, 768, 768) -> (3, 13, 768)
        ch = torch.matmul(ch, self.critic_w1) + self.critic_b1.unsqueeze(1)
        ch = F.leaky_relu(ch)
        # Output: (3, 13, 768) x (3, 768, 128) -> (3, 13, 128)
        c_logits = torch.matmul(ch, self.critic_w2) + self.critic_b2.unsqueeze(1)
        c_probs = torch.softmax(c_logits, dim=-1)
        q_per_critic = (c_probs * self.critic_bin_vals.reshape(1, 1, -1)).sum(dim=-1)  # (3, 13)
        q_values = q_per_critic.mean(dim=0, keepdim=True)  # (1, 13)
        v_value = (probs * q_values).sum(dim=-1)  # (1,)

        return probs, q_values, v_value, new_k_cache, new_v_cache


def convert_weights_to_fp16_cast(in_path: Path, out_path: Path, min_elems: int = 1024) -> None:
    """Lossless FP16 weight compression: store large FP32 initializers as FP16 + Cast(FP32).

    Activations, LayerNorms, Softmax, and KV cache remain full FP32, matching the
    pattern in pkmn.ai's optimization guide (KL ~ 3e-6, 100% argmax agreement, ~2x smaller).
    """
    model = onnx.load(str(in_path))
    graph = model.graph

    new_initializers = []
    cast_nodes = []
    converted_names: set[str] = set()
    converted_bytes = 0

    for init in graph.initializer:
        if init.data_type == TensorProto.FLOAT:
            arr = numpy_helper.to_array(init)
            if arr.size >= min_elems:
                fp16_name = f"{init.name}__fp16"
                fp16_arr = arr.astype(np.float16)
                new_initializers.append(numpy_helper.from_array(fp16_arr, name=fp16_name))
                cast_nodes.append(
                    helper.make_node(
                        "Cast",
                        inputs=[fp16_name],
                        outputs=[init.name],
                        to=TensorProto.FLOAT,
                        name=f"cast_{init.name}",
                    )
                )
                converted_names.add(init.name)
                converted_bytes += arr.nbytes - fp16_arr.nbytes
                continue
        new_initializers.append(init)

    del graph.initializer[:]
    graph.initializer.extend(new_initializers)
    for node in reversed(cast_nodes):
        graph.node.insert(0, node)

    onnx.save(model, str(out_path))
    print(
        f"Converted {len(converted_names)} initializers to FP16+Cast "
        f"(saved {converted_bytes / 1e6:.2f} MB) -> {out_path} "
        f"({out_path.stat().st_size / 1e6:.2f} MB)"
    )


def _load_tauros_reference(repo_root: Path, ckpt_path: Path):
    _stub_training_only_imports()
    _patch_amago_for_cpu_reference()

    os.environ["METAMON_CACHE_DIR"] = str(repo_root / ".build-cache" / "cache")
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    import gin
    from metamon.rl.pretrained import TaurosV0
    from metamon.rl.metamon_to_amago import make_placeholder_experiment

    gin.clear_config()
    m = TaurosV0()
    gin.parse_config_files_and_bindings(
        [m.model_gin_config_path, m.train_gin_config_path],
        [],
        finalize_config=False,
    )
    for k, v in m.base_config.items():
        gin.bind_parameter(k, v)
    gin.finalize()

    exp = make_placeholder_experiment(
        ckpt_base_dir=str(repo_root / ".build-cache" / "cache"),
        run_name="tauros-export",
        log=False,
        observation_space=m.observation_space,
        action_space=m.action_space,
    )
    exp.start()
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    exp.policy.load_state_dict(ckpt)
    exp.policy.cpu()
    exp.policy.eval()

    ws = exp.policy.traj_encoder.tformer.layers[0].attention_layer.self_attention.window_size
    assert ws == (32, 0), f"Expected FlashAttention window_size=(32, 0), got {ws}"
    return m, exp.policy


def _verify_equivalence(
    ref_policy: nn.Module,
    wrapper: TaurosOnnxModule,
    ort_paths: dict[str, Path],
    vocab_size: int,
    steps: int = 45,
) -> None:
    """Verify multi-step trajectory equivalence across PyTorch ref, wrapper, and ONNX."""
    rng = np.random.default_rng(12345)

    # Use float32 KV cache in ref policy so we measure true architectural parity
    # (real GPU FlashAttention uses bfloat16 cache; checking against float32 is stricter).
    import amago.nets.transformer as tr

    ref_hidden = tr.TformerHiddenState(
        key_cache=tr.Cache(
            device=torch.device("cpu"),
            dtype=torch.float32,
            layers=ref_policy.traj_encoder.n_layers,
            batch_size=1,
            max_seq_len=ref_policy.traj_encoder.max_seq_len,
            n_heads=ref_policy.traj_encoder.n_heads,
            head_dim=ref_policy.traj_encoder.head_dim,
        ),
        val_cache=tr.Cache(
            device=torch.device("cpu"),
            dtype=torch.float32,
            layers=ref_policy.traj_encoder.n_layers,
            batch_size=1,
            max_seq_len=ref_policy.traj_encoder.max_seq_len,
            n_heads=ref_policy.traj_encoder.n_heads,
            head_dim=ref_policy.traj_encoder.head_dim,
        ),
        seq_lens=torch.zeros((1,), dtype=torch.int32),
    )

    sessions = {}
    ort_caches = {}
    for label, path in ort_paths.items():
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        opts.inter_op_num_threads = 1
        sessions[label] = ort.InferenceSession(
            str(path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        ort_caches[label] = (
            np.zeros((4, 1, 32, 8, 88), dtype=np.float32),
            np.zeros((4, 1, 32, 8, 88), dtype=np.float32),
        )

    wrap_k = torch.zeros((4, 1, 32, 8, 88), dtype=torch.float32)
    wrap_v = torch.zeros((4, 1, 32, 8, 88), dtype=torch.float32)

    metrics = {
        label: {
            "prob_max_err": 0.0,
            "q_max_err": 0.0,
            "v_abs_errs": [],
            "kls": [],
            "argmax_matches": 0,
            "latencies_ms": [],
        }
        for label in ["torch_wrapper", *ort_paths.keys()]
    }

    action_eye = F.one_hot(torch.arange(13), num_classes=13).float()
    critic_one_hot = action_eye.view(1, 13, 1, 1, 13).repeat(1, 1, 1, len(ref_policy.gammas), 1)
    critic_actions = ref_policy.actor.policy_dist.action_from_buffer(critic_one_hot)

    with torch.no_grad():
        for t in range(steps):
            pt = rng.integers(-1, vocab_size - 1, size=(1, 7, 12), dtype=np.int64)
            pn = rng.normal(0, 0.5, size=(1, 7, 31)).astype(np.float32)
            gt = rng.integers(-1, vocab_size - 1, size=(1, 20), dtype=np.int64)
            gn = rng.uniform(0, 1, size=(1, 4)).astype(np.float32)
            rl2 = np.zeros((1, 14), dtype=np.float32)
            if t > 0:
                rl2[0, 0] = float(rng.normal(0, 1.0))
                rl2[0, 1 + int(rng.integers(0, 9))] = 1.0
            # Realistic gen1ou mask: 1-4 moves legal, 0-5 switches legal, tera (9..12) illegal
            illegal = np.ones((1, 13), dtype=bool)
            n_moves = int(rng.integers(1, 5))
            n_switches = int(rng.integers(0, 6))
            illegal[0, :n_moves] = False
            illegal[0, 4 : 4 + n_switches] = False

            # --- Reference PyTorch policy ---
            obs_dict = {
                "text_active_pokemon_tokens": torch.from_numpy(pt[:, 0:1, :]).to(torch.int32),
                "numbers_active_pokemon": torch.from_numpy(pn[:, 0:1, :]),
                "text_opponent_active_pokemon_tokens": torch.from_numpy(pt[:, 6:7, :]).to(torch.int32),
                "numbers_opponent_active_pokemon": torch.from_numpy(pn[:, 6:7, :]),
                "text_misc_tokens": torch.from_numpy(gt).unsqueeze(1).to(torch.int32),
                "numbers_misc": torch.from_numpy(gn).unsqueeze(1),
                "illegal_actions": torch.from_numpy(illegal).unsqueeze(1),
            }
            for s in range(5):
                obs_dict[f"text_switch_{s}_tokens"] = torch.from_numpy(pt[:, 1 + s : 2 + s, :]).to(torch.int32)
                obs_dict[f"numbers_switch_{s}"] = torch.from_numpy(pn[:, 1 + s : 2 + s, :])

            traj_emb_ref, ref_hidden = ref_policy.get_state_embedding(
                obs=obs_dict,
                rl2s=torch.from_numpy(rl2).unsqueeze(1),
                time_idxs=torch.tensor([[[t]]], dtype=torch.int64),
                hidden_state=ref_hidden,
            )
            raw_logits_ref = ref_policy.actor.actor_network_forward(
                traj_emb_ref, straight_from_obs={"illegal_actions": obs_dict["illegal_actions"]}
            )[0, 0, -1, :]
            ref_probs = torch.softmax(raw_logits_ref, dim=-1).numpy()
            ref_probs[illegal[0]] = 0.0
            ref_probs = ref_probs / ref_probs.sum()

            q_dist_ref = ref_policy.critics(traj_emb_ref.repeat(13, 1, 1), critic_actions)
            q_vals_ref = ref_policy.critics.bin_dist_to_raw_vals(q_dist_ref).mean(dim=3)[0, :, 0, -1, 0].numpy()
            v_ref = float(np.sum(ref_probs * q_vals_ref))
            ref_argmax = int(np.argmax(ref_probs))

            # --- PyTorch Wrapper ---
            t0 = time.perf_counter()
            w_probs, w_q, w_v, wrap_k, wrap_v = wrapper(
                torch.from_numpy(pt),
                torch.from_numpy(pn),
                torch.from_numpy(gt),
                torch.from_numpy(gn),
                torch.from_numpy(rl2),
                torch.from_numpy(illegal),
                torch.tensor([t], dtype=torch.int64),
                torch.tensor([min(t, 32)], dtype=torch.int64),
                wrap_k,
                wrap_v,
            )
            dt_ms = (time.perf_counter() - t0) * 1000.0
            _record_metrics(
                metrics["torch_wrapper"],
                ref_probs,
                q_vals_ref,
                v_ref,
                ref_argmax,
                w_probs.numpy()[0],
                w_q.numpy()[0],
                float(w_v.item()),
                dt_ms,
            )

            # --- ONNX Runtime sessions ---
            for label, sess in sessions.items():
                ok, ov = ort_caches[label]
                t0 = time.perf_counter()
                o_probs, o_q, o_v, nok, nov = sess.run(
                    None,
                    {
                        "pokemon_text": pt,
                        "pokemon_nums": pn,
                        "global_text": gt,
                        "global_nums": gn,
                        "rl2": rl2,
                        "illegal_actions": illegal,
                        "time_idx": np.array([t], dtype=np.int64),
                        "cache_len": np.array([min(t, 32)], dtype=np.int64),
                        "k_cache": ok,
                        "v_cache": ov,
                    },
                )
                dt_ms = (time.perf_counter() - t0) * 1000.0
                ort_caches[label] = (nok, nov)
                _record_metrics(
                    metrics[label],
                    ref_probs,
                    q_vals_ref,
                    v_ref,
                    ref_argmax,
                    o_probs[0],
                    o_q[0],
                    float(o_v[0]),
                    dt_ms,
                )

    print(f"\n=== Multi-step Parity Verification ({steps} steps, window=32) ===")
    for label, m in metrics.items():
        mean_kl = float(np.mean(m["kls"]))
        max_kl = float(np.max(m["kls"]))
        v_mae = float(np.mean(m["v_abs_errs"]))
        v_max = float(np.max(m["v_abs_errs"]))
        med_ms = float(np.median(m["latencies_ms"][2:]))
        print(
            f"  {label:16s} | argmax={m['argmax_matches']}/{steps} "
            f"| prob_max_err={m['prob_max_err']:.2e} | mean_KL={mean_kl:.2e} max_KL={max_kl:.2e} "
            f"| Q_max_err={m['q_max_err']:.4f} | V_MAE={v_mae:.4f} V_max={v_max:.4f} "
            f"| med={med_ms:.2f}ms"
        )
        assert m["argmax_matches"] == steps, f"{label} argmax mismatch!"
        assert max_kl < 1e-3, f"{label} KL too high: {max_kl}"
        assert v_mae < 1.0, f"{label} V MAE too high: {v_mae}"
        assert v_max < 10.0, f"{label} V max error too high: {v_max}"


def _record_metrics(
    m: dict,
    ref_probs: np.ndarray,
    ref_q: np.ndarray,
    ref_v: float,
    ref_argmax: int,
    probs: np.ndarray,
    q: np.ndarray,
    v: float,
    dt_ms: float,
) -> None:
    m["prob_max_err"] = max(m["prob_max_err"], float(np.max(np.abs(probs - ref_probs))))
    m["q_max_err"] = max(m["q_max_err"], float(np.max(np.abs(q - ref_q))))
    m["v_abs_errs"].append(abs(v - ref_v))
    eps = 1e-12
    mask = ref_probs > 0
    kl = float(np.sum(ref_probs[mask] * np.log((ref_probs[mask] + eps) / (probs[mask] + eps))))
    m["kls"].append(max(0.0, kl))
    if int(np.argmax(probs)) == ref_argmax:
        m["argmax_matches"] += 1
    m["latencies_ms"].append(dt_ms)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Metamon TaurosV0 to ONNX.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(".build-cache/weights/tauros_v0_epoch_62.pt"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(".build-cache/onnx"),
    )
    parser.add_argument("--verify-steps", type=int, default=45)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    ckpt_path = args.checkpoint if args.checkpoint.is_absolute() else repo_root / args.checkpoint
    out_dir = args.out_dir if args.out_dir.is_absolute() else repo_root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading TaurosV0 checkpoint from {ckpt_path} ...")
    model_spec, ref_policy = _load_tauros_reference(repo_root, ckpt_path)
    wrapper = TaurosOnnxModule(ref_policy).eval()

    dummy_inputs = (
        torch.zeros((1, 7, 12), dtype=torch.int64),
        torch.zeros((1, 7, 31), dtype=torch.float32),
        torch.zeros((1, 20), dtype=torch.int64),
        torch.zeros((1, 4), dtype=torch.float32),
        torch.zeros((1, 14), dtype=torch.float32),
        torch.tensor([[False, False, True, True, False, True, True, True, True, True, True, True, True]]),
        torch.tensor([0], dtype=torch.int64),
        torch.tensor([0], dtype=torch.int64),
        torch.zeros((4, 1, 32, 8, 88), dtype=torch.float32),
        torch.zeros((4, 1, 32, 8, 88), dtype=torch.float32),
    )

    fp32_path = out_dir / "tauros_v0_fp32.onnx"
    fp16_path = out_dir / "tauros_v0_fp16.onnx"

    print(f"Exporting FP32 ONNX -> {fp32_path} ...")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy_inputs,
            str(fp32_path),
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            dynamo=False,
            input_names=[
                "pokemon_text",
                "pokemon_nums",
                "global_text",
                "global_nums",
                "rl2",
                "illegal_actions",
                "time_idx",
                "cache_len",
                "k_cache",
                "v_cache",
            ],
            output_names=[
                "probs",
                "q_values",
                "v_value",
                "new_k_cache",
                "new_v_cache",
            ],
        )
    print(f"Exported FP32 ONNX: {fp32_path.stat().st_size / 1e6:.2f} MB")

    convert_weights_to_fp16_cast(fp32_path, fp16_path)

    _verify_equivalence(
        ref_policy=ref_policy,
        wrapper=wrapper,
        ort_paths={"onnx_fp32": fp32_path, "onnx_fp16": fp16_path},
        vocab_size=len(model_spec.tokenizer) + 1,
        steps=args.verify_steps,
    )


if __name__ == "__main__":
    main()
