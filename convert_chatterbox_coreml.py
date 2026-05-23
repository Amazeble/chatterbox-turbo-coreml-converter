#!/usr/bin/env python3
"""
Convert Chatterbox Turbo TTS models to CoreML format.

Produces three CoreML models:
  - T3Prefill.mlpackage  — GPT-2 prefill (GPU)
  - T3Decode.mlpackage   — GPT-2 single-step decode (GPU)
  - S3Encoder.mlpackage  — Conformer encoder (ANE)
  - S3UNet.mlpackage     — U-Net denoiser (ANE)

Plus vocoder weights and tokenizer files for the Swift package.

Usage:
    python convert_chatterbox_coreml.py --stage t3 --output-dir /tmp/chatterbox-coreml
    python convert_chatterbox_coreml.py --stage s3 --output-dir /tmp/chatterbox-coreml
    python convert_chatterbox_coreml.py --stage vocoder --output-dir /tmp/chatterbox-coreml
    python convert_chatterbox_coreml.py --stage all --output-dir /tmp/chatterbox-coreml [--validate]
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import coremltools as ct
from safetensors.torch import save_file as save_safetensors


# ---------------------------------------------------------------------------
# Constants matching Chatterbox Turbo's architecture
# ---------------------------------------------------------------------------
SPEECH_VOCAB_SIZE = 6563
SPEECH_STOP_TOKEN = 6562
SPEECH_START_TOKEN = 0
MEL_BINS = 80
SPEAKER_EMB_DIM = 256
CAMPP_EMB_DIM = 192
GPT2_HIDDEN = 1024
GPT2_HEADS = 16
GPT2_LAYERS = 24
GPT2_MAX_POS = 8196
GPT2_HEAD_DIM = GPT2_HIDDEN // GPT2_HEADS  # 64

# Text vocab size for GPT-2 tokenizer (not the speech vocab)
TEXT_VOCAB_SIZE = 50258


# ---------------------------------------------------------------------------
# Stateful KV cache + patched GPT-2 attention for ANE StateType decode
# ---------------------------------------------------------------------------


class SliceUpdateKeyValueCache:
    """Seq-first 2D KV cache with dim-0 slice writes for CoreML StateType.

    Layout: keyCache/valueCache are (max_seq, n_layers * n_heads * head_dim).
    Sequence dimension is dim 0 — the ONLY dimension CoreML runtime supports
    for dynamic slice_update on state tensors (confirmed via testing).

    Per-layer K/V are extracted by slicing the feature dimension, then reshaped
    to (1, n_heads, seq_len, head_dim) for attention.
    """

    def __init__(self, key_buffer, value_buffer, n_layers, n_heads, head_dim):
        # key/value_buffer: (max_seq, n_layers * n_heads * head_dim)
        self.key_cache = key_buffer
        self.value_cache = value_buffer
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.layer_size = n_heads * head_dim  # features per layer

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        """Write new K/V for one layer and return full cache for that layer.

        key_states:   (1, n_heads, seq_len, head_dim)
        value_states: (1, n_heads, seq_len, head_dim)
        """
        cache_position = cache_kwargs.get("cache_position")
        begin = cache_position[0]
        end = cache_position[-1] + 1
        seq_len = key_states.shape[2]

        # Flatten (1, n_heads, seq_len, head_dim) -> (seq_len, n_heads * head_dim)
        k_flat = key_states.squeeze(0).transpose(0, 1).reshape(seq_len, self.layer_size)
        v_flat = value_states.squeeze(0).transpose(0, 1).reshape(seq_len, self.layer_size)

        # Feature slice for this layer
        feat_start = layer_idx * self.layer_size
        feat_end = feat_start + self.layer_size

        # Write into 2D cache at dim 0 (seq positions)
        self.key_cache[begin:end, feat_start:feat_end] = k_flat
        self.value_cache[begin:end, feat_start:feat_end] = v_flat

        # Read back FULL cache for this layer (fixed shape — no dynamic slice on read).
        # Unfilled positions are zeros; with is_causal=False, attention to zero K/V
        # produces near-zero weights after softmax, so this is numerically safe.
        max_seq = self.key_cache.shape[0]
        k_out = self.key_cache[:, feat_start:feat_end]  # (max_seq, layer_size)
        v_out = self.value_cache[:, feat_start:feat_end]

        # Reshape to (1, n_heads, max_seq, head_dim)
        k_out = k_out.reshape(max_seq, self.n_heads, self.head_dim).transpose(0, 1).unsqueeze(0)
        v_out = v_out.reshape(max_seq, self.n_heads, self.head_dim).transpose(0, 1).unsqueeze(0)

        return k_out, v_out

    def get_seq_length(self, layer_idx=0):
        return 0

    def get_max_cache_shape(self):
        return None

    def get_mask_sizes(self, cache_position, layer_idx=0):
        """Return (kv_length, kv_offset) for causal mask creation."""
        kv_length = cache_position[-1].item() + 1
        return kv_length, 0


def patched_gpt2_attention_forward(
    self,
    hidden_states,
    past_key_values=None,
    cache_position=None,
    attention_mask=None,
    head_mask=None,
    encoder_hidden_states=None,
    encoder_attention_mask=None,
    output_attentions=False,
    **kwargs,
):
    """Simplified GPT2Attention.forward using SliceUpdateKeyValueCache.

    - Passes cache_position from kwargs to cache.update()
    - Always is_causal=False (GPT-2 causal bias is in model weights;
      avoids torch.jit.trace baking a bool constant)
    - Removes cross-attention, upcast_and_reorder, encoder paths
    """
    query_states, key_states, value_states = self.c_attn(hidden_states).split(
        self.split_size, dim=2
    )

    shape_kv = (*key_states.shape[:-1], -1, self.head_dim)
    key_states = key_states.view(shape_kv).transpose(1, 2)
    value_states = value_states.view(shape_kv).transpose(1, 2)

    shape_q = (*query_states.shape[:-1], -1, self.head_dim)
    query_states = query_states.view(shape_q).transpose(1, 2)

    if past_key_values is not None:
        key_states, value_states = past_key_values.update(
            key_states,
            value_states,
            self.layer_idx,
            {"cache_position": cache_position},
        )

    # Ensure matching dtypes (cache is FP16, projections may be FP32)
    key_states = key_states.to(query_states.dtype)
    value_states = value_states.to(query_states.dtype)

    # Read kv_mask from cache object (stored by T3StatefulWrapper.forward)
    kv_mask = getattr(past_key_values, 'kv_mask', None) if past_key_values is not None else None
    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=kv_mask,  # (1, 1, 1, 2048) — 0 valid, -1e9 unfilled
        is_causal=False,
        dropout_p=0.0,
    )

    # SDPA returns (batch, heads, seq, head_dim) — transpose to (batch, seq, heads, head_dim)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(*attn_output.shape[:-2], -1)  # (batch, seq, hidden)
    attn_output = self.c_proj(attn_output)
    attn_output = self.resid_dropout(attn_output)
    return attn_output, None


# ---------------------------------------------------------------------------
# Helper: load the PyTorch Chatterbox Turbo model
# ---------------------------------------------------------------------------
class ChatterboxModels:
    """Container for individually loaded Chatterbox sub-models."""
    def __init__(self, t3, s3gen, model_dir):
        self.t3 = t3
        self.s3gen = s3gen
        self.model_dir = model_dir


def load_pytorch_model(cache_dir=None):
    """Download and load the Chatterbox Turbo PyTorch model components."""
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    print("Downloading Chatterbox Turbo weights...")
    model_dir = Path(snapshot_download("ResembleAI/chatterbox-turbo"))
    print(f"  Model dir: {model_dir}")

    # Load T3 with turbo config from YAML, but override to GPT2_medium
    print("  Loading T3 (GPT-2 Medium turbo)...")
    from chatterbox.models.t3.t3 import T3, T3Config
    import yaml

    # Load the turbo config from YAML
    yaml_path = model_dir / "t3_turbo_v1.yaml"
    with open(yaml_path) as f:
        # Use full_load to handle !!python/tuple tags
        cfg_dict = yaml.full_load(f)

    # Override llama_config_name — weights are GPT-2 despite YAML
    cfg_dict["llama_config_name"] = "GPT2_medium"
    turbo_cfg = T3Config.__new__(T3Config)
    # T3Config uses a simple __init__, populate from dict
    for k, v in cfg_dict.items():
        setattr(turbo_cfg, k, v)

    t3 = T3(hp=turbo_cfg)
    t3_state = load_file(model_dir / "t3_turbo_v1.safetensors")
    t3.load_state_dict(t3_state)
    t3.to("cpu").eval()

    # Load S3Gen
    print("  Loading S3Gen...")
    from chatterbox.models.s3gen import S3Gen
    s3gen = S3Gen()
    s3gen.load_state_dict(load_file(model_dir / "s3gen.safetensors"), strict=False)
    s3gen.to("cpu").eval()

    print("  Models loaded successfully.")
    return ChatterboxModels(t3=t3, s3gen=s3gen, model_dir=model_dir)


# ===========================================================================
# T3 CONVERSION
# ===========================================================================


CONTEXT_SIZE = 2048  # Max sequence length for KV cache state


class T3StatefulWrapper(nn.Module):
    """Stateful T3 wrapper for CoreML: takes pre-computed embeddings, not token IDs.

    The embedding lookup and speaker conditioning happen in Swift.
    This model is JUST: transformer + speech_head + KV cache state.

    Swift caller handles:
    - Position 0: cond_enc.spkr_enc(speaker_emb) → (1, 1, 1024) conditioning embedding
    - Positions 1+: speech_emb[token_id] → (1, 1, 1024) token embedding
    """

    def __init__(self, t3_model, context_size=CONTEXT_SIZE):
        super().__init__()
        self.tfmr = t3_model.t3.tfmr
        self.speech_head = t3_model.t3.speech_head
        self.context_size = context_size

        # 2D seq-first cache: (max_seq, n_layers * n_heads * head_dim)
        feature_size = GPT2_LAYERS * GPT2_HEADS * GPT2_HEAD_DIM
        self.register_buffer("keyCache", torch.zeros(context_size, feature_size, dtype=torch.float16))
        self.register_buffer("valueCache", torch.zeros(context_size, feature_size, dtype=torch.float16))

    def forward(self, inputs_embeds, position_ids, cache_position, attention_mask):
        """
        Args:
            inputs_embeds:  (1, 1, 1024) float — pre-computed embedding from Swift
            position_ids:   (1, 1) int32 — GPT-2 wpe position
            cache_position: (1,) int32 — KV cache write position
            attention_mask: (1, 1, 1, 2048) float — 0 valid, -1e9 unfilled

        Returns:
            logits: (1, vocab_size) float16 — logits for next token
        """
        cache = SliceUpdateKeyValueCache(
            self.keyCache, self.valueCache,
            n_layers=GPT2_LAYERS, n_heads=GPT2_HEADS, head_dim=GPT2_HEAD_DIM
        )
        # Store mask on cache for patched attention to read
        cache.kv_mask = attention_mask

        outputs = self.tfmr(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
            cache_position=cache_position,
        )
        hidden = outputs.last_hidden_state

        logits = self.speech_head(hidden[:, -1:, :]).squeeze(1)
        return logits


def convert_t3(model, output_dir, validate=False):
    """Convert T3 to a single stateful CoreML model with StateType KV cache."""
    print("\n=== Converting T3 Stateful (GPT-2 + KV Cache) ===")

    t3_model = model
    t3_model.t3.eval()

    # Monkey-patch GPT2Attention to use SliceUpdateKeyValueCache
    from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
    original_forward = GPT2Attention.forward
    GPT2Attention.forward = patched_gpt2_attention_forward

    wrapper = T3StatefulWrapper(t3_model, context_size=CONTEXT_SIZE)
    wrapper.eval()

    # Export embedding weights for Swift-side lookup
    print("  Exporting embedding weights...")
    speech_emb_weights = t3_model.t3.speech_emb.weight.data.cpu().float()  # (6563, 1024)
    np.save(os.path.join(output_dir, "speech_emb.npy"), speech_emb_weights.numpy())
    print(f"    speech_emb: {speech_emb_weights.shape}")

    text_emb_weights = t3_model.t3.text_emb.weight.data.cpu().float()  # (50276, 1024)
    np.save(os.path.join(output_dir, "text_emb.npy"), text_emb_weights.numpy())
    print(f"    text_emb: {text_emb_weights.shape}")

    # Export conditioning linear weights (spkr_enc)
    spkr_enc = t3_model.t3.cond_enc.spkr_enc
    spkr_w = spkr_enc.weight.data.cpu().float()  # (1024, 256)
    spkr_b = spkr_enc.bias.data.cpu().float()    # (1024,)
    np.save(os.path.join(output_dir, "spkr_enc_weight.npy"), spkr_w.numpy())
    np.save(os.path.join(output_dir, "spkr_enc_bias.npy"), spkr_b.numpy())
    print(f"    spkr_enc: weight {spkr_w.shape}, bias {spkr_b.shape}")

    # Trace with float embedding input + attention mask
    example_embeds = torch.randn(1, 1, GPT2_HIDDEN)
    example_pos = torch.zeros(1, 1, dtype=torch.int32)
    example_cache_pos = torch.zeros(1, dtype=torch.int32)
    example_mask = torch.zeros(1, 1, 1, CONTEXT_SIZE, dtype=torch.float32)
    example_mask[:, :, :, 1:] = -1e9  # only position 0 valid in example

    print("  Tracing T3Stateful (inputs_embeds + mask)...")
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (example_embeds, example_pos, example_cache_pos, example_mask))

    # Restore original forward
    GPT2Attention.forward = original_forward

    # StateType for 2D seq-first KV cache
    feature_size = GPT2_LAYERS * GPT2_HEADS * GPT2_HEAD_DIM  # 24 * 16 * 64 = 24576
    cache_shape = (CONTEXT_SIZE, feature_size)
    states = [
        ct.StateType(
            wrapped_type=ct.TensorType(shape=cache_shape, dtype=np.float16),
            name="keyCache",
        ),
        ct.StateType(
            wrapped_type=ct.TensorType(shape=cache_shape, dtype=np.float16),
            name="valueCache",
        ),
    ]

    # Fixed decode shape: seq=1 token + 1 conditioning = 2 positions.
    # EnumeratedShapes and RangeDim both cause error -14 with stateful models.
    # Prefill is done token-by-token through the same fixed-shape model.
    print("  Converting with fixed decode shape (seq=1, pos=2)...")

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="inputs_embeds", shape=(1, 1, GPT2_HIDDEN), dtype=np.float32),
            ct.TensorType(name="position_ids", shape=(1, 1), dtype=np.int32),
            ct.TensorType(name="cache_position", shape=(1,), dtype=np.int32),
            ct.TensorType(name="attention_mask", shape=(1, 1, 1, CONTEXT_SIZE), dtype=np.float32),
        ],
        outputs=[ct.TensorType(name="logits", dtype=np.float16)],
        states=states,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.iOS18,
    )

    out_path = os.path.join(output_dir, "T3Stateful.mlpackage")
    mlmodel.save(out_path)
    print(f"  Saved: {out_path}")

    # Check for state ops in MIL program
    from coremltools.converters.mil.testing_utils import get_op_types_in_program
    ops = get_op_types_in_program(mlmodel._mil_program)
    has_state = "coreml_update_state" in ops
    print(f"  State ops present: {has_state}")
    if not has_state:
        print("  WARNING: No coreml_update_state — KV cache may not work!")

    if validate:
        validate_t3_stateful(t3_model, out_path)


def validate_t3_stateful(pytorch_model, model_path):
    """Validate stateful T3 CoreML output matches PyTorch."""
    print("\n  --- T3 Stateful Numerical Validation ---")

    # Save and reload to get CoreML framework backend (needed for make_state).
    # The convert() output uses coremltools internal backend which can't make_state().
    # CPU_ONLY avoids error -14 from ANE compilation of dynamic shapes.
    import tempfile, shutil
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = os.path.join(tmpdir, "T3Stateful.mlpackage")
        shutil.copytree(model_path, tmp_path)
        ml_model = ct.models.MLModel(tmp_path, compute_units=ct.ComputeUnit.CPU_ONLY)

    state = ml_model.make_state()

    # Token-by-token prefill (fixed shape model only accepts seq=1)
    # Each call processes 1 speech token + 1 conditioning token = 2 positions
    import time
    prefill_ids = np.random.randint(0, SPEECH_VOCAB_SIZE, (16,)).astype(np.int32)
    prefill_spk = np.random.randn(1, SPEAKER_EMB_DIM).astype(np.float32)
    zero_spk = np.zeros((1, SPEAKER_EMB_DIM), dtype=np.float32)

    # Load speech embedding table for lookups
    speech_emb_table = pytorch_model.t3.speech_emb.weight.data.cpu().numpy()  # (vocab, 1024)

    # Load spkr_enc weights for conditioning
    spkr_enc = pytorch_model.t3.cond_enc.spkr_enc
    spkr_w = spkr_enc.weight.data.cpu().numpy()  # (1024, 256)
    spkr_b = spkr_enc.bias.data.cpu().numpy()    # (1024,)

    # Prefill: position 0 = speaker conditioning, positions 1+ = speech tokens
    speaker_emb = np.random.randn(SPEAKER_EMB_DIM).astype(np.float32)
    cond_emb = (spkr_w @ speaker_emb + spkr_b).reshape(1, 1, GPT2_HIDDEN).astype(np.float32)

    t0 = time.time()
    # Position 0: conditioning
    ml_model.predict(
        {"inputs_embeds": cond_emb, "position_ids": np.array([[0]], dtype=np.int32),
         "cache_position": np.array([0], dtype=np.int32)},
        state=state,
    )
    # Positions 1-16: speech tokens
    for i in range(16):
        emb = speech_emb_table[prefill_ids[i]].reshape(1, 1, GPT2_HIDDEN).astype(np.float32)
        pos = np.array([[i + 1]], dtype=np.int32)
        cp = np.array([i + 1], dtype=np.int32)
        ml_model.predict(
            {"inputs_embeds": emb, "position_ids": pos, "cache_position": cp},
            state=state,
        )
    prefill_ms = (time.time() - t0) * 1000
    print(f"  Prefill (1 cond + 16 tokens): {prefill_ms:.0f}ms ({prefill_ms/17:.1f}ms/tok)")

    # Decode step 1
    decode_emb = speech_emb_table[42].reshape(1, 1, GPT2_HIDDEN).astype(np.float32)
    cm_d1 = ml_model.predict(
        {"inputs_embeds": decode_emb, "position_ids": np.array([[17]], dtype=np.int32),
         "cache_position": np.array([17], dtype=np.int32)},
        state=state,
    )

    # Decode step 2
    cm_d2 = ml_model.predict(
        {"inputs_embeds": decode_emb, "position_ids": np.array([[18]], dtype=np.int32),
         "cache_position": np.array([18], dtype=np.int32)},
        state=state,
    )

    diff = np.abs(cm_d1["logits"] - cm_d2["logits"]).max()
    print(f"  Decode step diff: {diff:.4f}")
    if diff > 0.001:
        print("  PASS: KV cache working across decode steps!")
    else:
        print("  WARNING: Decode outputs identical")


# ===========================================================================
# S3 CONVERSION (Encoder + UNet)
# ===========================================================================


class S3EncoderWrapper(nn.Module):
    """Wraps the S3 encoder path for CoreML tracing.

    Takes a single concatenated token sequence (prompt + speech), provides
    token_len internally, and returns mel-projected encoder output.
    The encoder components live under s3gen.flow.
    """

    def __init__(self, s3_flow):
        super().__init__()
        self.input_embedding = s3_flow.input_embedding
        self.encoder = s3_flow.encoder
        self.encoder_proj = s3_flow.encoder_proj

    def forward(self, all_tokens):
        """
        Args:
            all_tokens: (1, T) int32 - concatenated [prompt_tokens | speech_tokens]

        Returns:
            encoder_proj: (1, 80, T_enc) float32 - mel-projected, BCT format
        """
        T = all_tokens.size(1)
        token_len = torch.tensor([T], dtype=torch.long, device=all_tokens.device)
        mask = torch.ones(1, T, 1, device=all_tokens.device, dtype=torch.float32)

        x = self.input_embedding(all_tokens.long()) * mask
        h, _ = self.encoder(x, token_len)
        mu = self.encoder_proj(h)
        mu = mu.transpose(1, 2)  # (1, 80, T_enc)
        return mu


class S3UNetWrapper(nn.Module):
    """Wraps the U-Net denoiser (estimator) for CoreML tracing.

    Includes the spkEmbedAffineLayer (192 to projected dim) baked in.
    The decoder's estimator and affine layer live under s3gen.flow.
    """

    def __init__(self, s3_flow):
        super().__init__()
        self.estimator = s3_flow.decoder.estimator
        self.spk_affine = s3_flow.spk_embed_affine_layer

    def forward(self, x, mu, mask, t, spks, cond, r):
        """
        Args:
            x: (1, 80, T) float - noisy mel
            mu: (1, 80, T) float - target mel from encoder
            mask: (1, 1, T) float - validity mask
            t: (1,) float - timestep
            spks: (1, 192) float - raw CAMPPlus speaker embedding
            cond: (1, 80, T) float - conditioning mel
            r: (1,) float - meanflow ratio (unused by estimator, kept for API compat)

        Returns:
            velocity: (1, 80, T) float - predicted velocity
        """
        spks_proj = self.spk_affine(spks)
        return self.estimator(x, mask, mu, t, spks_proj, cond)


def convert_s3(model, output_dir, validate=False):
    """Convert S3Encoder and S3UNet to CoreML."""
    print("\n=== Converting S3Encoder (Conformer) ===")

    s3gen = model.s3gen
    s3gen.eval()
    s3_flow = s3gen.flow  # encoder/decoder live under flow

    # Monkey-patch view_as -> reshape (CoreML doesn't support view_as)
    _original_view_as = torch.Tensor.view_as
    torch.Tensor.view_as = lambda self, other: self.reshape(other.shape)

    # --- S3Encoder ---
    print("  Tracing S3Encoder...")
    encoder_wrapper = S3EncoderWrapper(s3_flow)
    encoder_wrapper.eval()

    # Single input: concatenated prompt + speech tokens
    example_tokens = torch.zeros(1, 70, dtype=torch.long)

    with torch.no_grad():
        traced_encoder = torch.jit.trace(encoder_wrapper, (example_tokens,))

    print("  Converting S3Encoder to CoreML...")
    encoder_inputs = [
        ct.TensorType(
            name="all_tokens",
            shape=ct.Shape(shape=(1, ct.RangeDim(lower_bound=1, upper_bound=2048, default=70))),
            dtype=np.int32,
        ),
    ]

    encoder_coreml = ct.convert(
        traced_encoder,
        inputs=encoder_inputs,
        outputs=[ct.TensorType(name="encoder_proj", dtype=np.float16)],
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.iOS18,
    )

    encoder_path = os.path.join(output_dir, "S3Encoder.mlpackage")
    encoder_coreml.save(encoder_path)
    print(f"  Saved: {encoder_path}")

    # --- S3UNet ---
    print("\n=== Converting S3UNet (Denoiser) ===")
    print("  Tracing S3UNet...")
    unet_wrapper = S3UNetWrapper(s3_flow)
    unet_wrapper.eval()

    T = 100  # example time steps
    example_x = torch.randn(1, MEL_BINS, T)
    example_mu = torch.randn(1, MEL_BINS, T)
    example_mask = torch.ones(1, 1, T)
    example_t = torch.tensor([0.5])
    example_spks = torch.randn(1, CAMPP_EMB_DIM)
    example_cond = torch.randn(1, MEL_BINS, T)
    example_r = torch.tensor([0.5])

    with torch.no_grad():
        traced_unet = torch.jit.trace(
            unet_wrapper,
            (example_x, example_mu, example_mask, example_t,
             example_spks, example_cond, example_r),
        )

    print("  Converting S3UNet to CoreML...")
    T_dim = ct.RangeDim(lower_bound=1, upper_bound=4096, default=100)
    unet_inputs = [
        ct.TensorType(name="x", shape=ct.Shape(shape=(1, MEL_BINS, T_dim)), dtype=np.float32),
        ct.TensorType(name="mu", shape=ct.Shape(shape=(1, MEL_BINS, T_dim)), dtype=np.float32),
        ct.TensorType(name="mask", shape=ct.Shape(shape=(1, 1, T_dim)), dtype=np.float32),
        ct.TensorType(name="t", shape=(1,), dtype=np.float32),
        ct.TensorType(name="spks", shape=(1, CAMPP_EMB_DIM), dtype=np.float32),
        ct.TensorType(name="cond", shape=ct.Shape(shape=(1, MEL_BINS, T_dim)), dtype=np.float32),
        ct.TensorType(name="r", shape=(1,), dtype=np.float32),
    ]

    unet_coreml = ct.convert(
        traced_unet,
        inputs=unet_inputs,
        outputs=[ct.TensorType(name="velocity", dtype=np.float16)],
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.iOS18,
    )

    unet_path = os.path.join(output_dir, "S3UNet.mlpackage")
    unet_coreml.save(unet_path)
    print(f"  Saved: {unet_path}")

    # Restore monkey-patched view_as
    torch.Tensor.view_as = _original_view_as

    if validate:
        validate_s3(s3_flow, encoder_path, unet_path)


def validate_s3(s3_flow, encoder_path, unet_path):
    """Validate S3 CoreML outputs match PyTorch."""
    print("\n  --- S3 Numerical Validation ---")

    encoder_ml = ct.models.MLModel(encoder_path)
    unet_ml = ct.models.MLModel(unet_path)

    # Test encoder
    test_tokens = torch.randint(0, SPEECH_VOCAB_SIZE, (1, 40), dtype=torch.long)

    encoder_wrapper = S3EncoderWrapper(s3_flow)
    with torch.no_grad():
        pt_mu = encoder_wrapper(test_tokens)

    cm_out = encoder_ml.predict({
        "all_tokens": test_tokens.int().numpy(),
    })
    cm_mu = torch.from_numpy(cm_out["encoder_proj"]).float()

    cos_sim = torch.nn.functional.cosine_similarity(
        pt_mu.flatten().unsqueeze(0),
        cm_mu.flatten().unsqueeze(0)
    ).item()
    print(f"  S3Encoder cosine similarity: {cos_sim:.6f}")
    if cos_sim >= 0.99:
        print("  PASS")
    else:
        print("  WARNING: cosine sim < 0.99")

    # Test UNet
    T = 40
    test_x = torch.randn(1, MEL_BINS, T)
    test_mu_in = torch.randn(1, MEL_BINS, T)
    test_mask = torch.ones(1, 1, T)
    test_t = torch.tensor([0.5])
    test_spks = torch.randn(1, CAMPP_EMB_DIM)
    test_cond = torch.randn(1, MEL_BINS, T)
    test_r = torch.tensor([0.5])

    unet_wrapper = S3UNetWrapper(s3_flow)
    unet_wrapper.eval()
    with torch.no_grad():
        pt_vel = unet_wrapper(test_x, test_mu_in, test_mask, test_t, test_spks, test_cond, test_r)

    cm_out = unet_ml.predict({
        "x": test_x.numpy(),
        "mu": test_mu_in.numpy(),
        "mask": test_mask.numpy(),
        "t": test_t.numpy(),
        "spks": test_spks.numpy(),
        "cond": test_cond.numpy(),
        "r": test_r.numpy(),
    })
    cm_vel = torch.from_numpy(cm_out["velocity"]).float()

    cos_sim = torch.nn.functional.cosine_similarity(
        pt_vel.flatten().unsqueeze(0),
        cm_vel.flatten().unsqueeze(0)
    ).item()
    print(f"  S3UNet cosine similarity: {cos_sim:.6f}")
    if cos_sim >= 0.99:
        print("  PASS")
    else:
        print("  WARNING: cosine sim < 0.99")


# ===========================================================================
# VOCODER WEIGHT EXTRACTION
# ===========================================================================

def extract_vocoder_weights(model, output_dir):
    """Extract HiFTGenerator weights to safetensors format."""
    print("\n=== Extracting Vocoder Weights ===")

    vocoder = model.s3gen.mel2wav
    state_dict = vocoder.state_dict()

    # Convert all weights to float32 contiguous tensors
    weights = {}
    for key, tensor in state_dict.items():
        weights[key] = tensor.contiguous().float()
        print(f"  {key}: {list(tensor.shape)}")

    output_path = os.path.join(output_dir, "hift_vocoder.safetensors")
    save_safetensors(weights, output_path)
    print(f"  Saved: {output_path} ({os.path.getsize(output_path) / 1e6:.1f} MB)")


# ===========================================================================
# TOKENIZER + CONFIG EXTRACTION
# ===========================================================================

def extract_tokenizer_and_config(model, output_dir):
    """Copy tokenizer files and create config.json."""
    print("\n=== Extracting Tokenizer + Config ===")

    # Find the cached model directory
    from huggingface_hub import snapshot_download
    model_dir = snapshot_download("ResembleAI/chatterbox-turbo")

    # Copy tokenizer files
    tokenizer_files = ["tokenizer.json", "vocab.json", "merges.txt"]
    for fname in tokenizer_files:
        src = os.path.join(model_dir, fname)
        if os.path.exists(src):
            dst = os.path.join(output_dir, fname)
            shutil.copy2(src, dst)
            print(f"  Copied: {fname}")
        else:
            print(f"  WARNING: {fname} not found in model directory")

    # Create config.json
    config = {
        "model_type": "chatterbox-turbo-coreml",
        "sample_rate": 24000,
        "speech_vocab_size": SPEECH_VOCAB_SIZE,
        "speech_stop_token": SPEECH_STOP_TOKEN,
        "speech_start_token": SPEECH_START_TOKEN,
        "mel_bins": MEL_BINS,
        "n_cfm_timesteps": 2,
        "speaker_emb_dim": SPEAKER_EMB_DIM,
        "campp_emb_dim": CAMPP_EMB_DIM,
        "gpt2_hidden": GPT2_HIDDEN,
        "gpt2_heads": GPT2_HEADS,
        "gpt2_layers": GPT2_LAYERS,
    }
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Saved: config.json")

    # Copy default conditioning if available
    for cond_file in ["default-conds.safetensors", "conds.safetensors"]:
        src = os.path.join(model_dir, cond_file)
        if os.path.exists(src):
            dst = os.path.join(output_dir, "default-conds.safetensors")
            shutil.copy2(src, dst)
            print(f"  Copied: {cond_file} as default-conds.safetensors")
            break


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Convert Chatterbox Turbo TTS to CoreML format"
    )
    parser.add_argument(
        "--stage",
        choices=["t3", "s3", "vocoder", "all"],
        required=True,
        help="Which stage to convert",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save converted models",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run numerical validation after conversion",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load the PyTorch model
    model = load_pytorch_model()

    if args.stage in ("t3", "all"):
        convert_t3(model, args.output_dir, validate=args.validate)

    if args.stage in ("s3", "all"):
        convert_s3(model, args.output_dir, validate=args.validate)

    if args.stage in ("vocoder", "all"):
        extract_vocoder_weights(model, args.output_dir)

    if args.stage == "all":
        extract_tokenizer_and_config(model, args.output_dir)

    print("\n=== Done ===")
    print(f"Output directory: {args.output_dir}")
    if args.stage == "all":
        print("Files:")
        for f in sorted(os.listdir(args.output_dir)):
            fpath = os.path.join(args.output_dir, f)
            if os.path.isdir(fpath):
                print(f"  {f}/")
            else:
                size = os.path.getsize(fpath)
                print(f"  {f} ({size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
