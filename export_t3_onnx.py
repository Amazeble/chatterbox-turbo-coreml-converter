#!/usr/bin/env python3
"""
Export Chatterbox T3 Turbo language model to ONNX format from safetensors weights.

This script loads t3_turbo_finetuned_merged.safetensors and exports:
  - language_model_single.onnx (GPT-2 single-step decoder with KV cache)

After export, applies ORT transformer optimization to fuse operations into
GroupQueryAttention nodes for optimal iOS/Windows deployment.

Usage:
    python export_t3_onnx.py \
        --weights ./t3_turbo_finetuned_merged.safetensors \
        --output-dir ./onnx_models \
        --optimize-graph

Expected output after optimization:
--- Node Type Distribution ---
  Add: 194
  MatMul: 145
  LayerNormalization: 49
  GroupQueryAttention: 24
  Gelu: 24
  Constant: 2
  Cast: 2
  Gather: 2
  ReduceSum: 1
  Sub: 1
  Shape: 1
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file


# ---------------------------------------------------------------------------
# Architecture Constants
# ---------------------------------------------------------------------------
GPT2_HIDDEN = 1024
GPT2_HEADS = 16
GPT2_LAYERS = 24
GPT2_HEAD_DIM = GPT2_HIDDEN // GPT2_HEADS  # 64
TEXT_VOCAB_SIZE = 50276
SPEECH_VOCAB_SIZE = 6563


# ---------------------------------------------------------------------------
# Language Model Wrapper - Re-implements attention with primitives
# ---------------------------------------------------------------------------
class _LanguageModelWrapper(nn.Module):
    """Single-step GPT-2 decode with explicit KV cache management.
    
    Avoids HF's Cache utility which breaks torch.onnx.export tracing.
    Uses scaled_dot_product_attention for efficient fused attention ops.
    
    Forward signature:
        (inputs_embeds, attention_mask, position_ids,
         k0, v0, k1, v1, ..., k23, v23)
    Outputs:
        (logits, present_k0, present_v0, ..., present_k23, present_v23)
    """

    def __init__(self, t3):
        super().__init__()
        tfmr = t3.tfmr
        self.speech_head = t3.speech_head  # Linear(1024, 6563)
        self.wpe = tfmr.wpe
        self.ln_f = tfmr.ln_f
        self.h = tfmr.h
        self.n_layer = len(tfmr.h)
        self.n_head = GPT2_HEADS
        self.head_dim = GPT2_HEAD_DIM
        self.hidden = GPT2_HIDDEN
        self.split_size = GPT2_HIDDEN

    def _block_forward(self, block, hidden_states, past_k, past_v, attn_mask_additive):
        # Self-attention
        residual = hidden_states
        h = block.ln_1(hidden_states)
        qkv = block.attn.c_attn(h)  # (1, seq, 3*hidden)
        q, k_new, v_new = qkv.split(self.split_size, dim=2)

        # Reshape: (1, seq, hidden) -> (1, heads, seq, head_dim)
        q = q.view(1, -1, self.n_head, self.head_dim).transpose(1, 2)
        k_new = k_new.view(1, -1, self.n_head, self.head_dim).transpose(1, 2)
        v_new = v_new.view(1, -1, self.n_head, self.head_dim).transpose(1, 2)

        # Append new key/value to past cache
        k_full = torch.cat([past_k, k_new], dim=2)
        v_full = torch.cat([past_v, v_new], dim=2)

        # Fused attention - will be optimized to GroupQueryAttention by ORT
        attn = torch.nn.functional.scaled_dot_product_attention(
            q, k_full, v_full,
            attn_mask=attn_mask_additive,
            is_causal=False,
            dropout_p=0.0,
        )
        attn = attn.transpose(1, 2).contiguous().view(1, -1, self.hidden)
        attn = block.attn.c_proj(attn)
        attn = block.attn.resid_dropout(attn)
        hidden_states = residual + attn

        # MLP
        residual = hidden_states
        h = block.ln_2(hidden_states)
        h = block.mlp(h)
        hidden_states = residual + h

        return hidden_states, k_full, v_full

    def forward(self, inputs_embeds, attention_mask, position_ids, *flat_past_kv):
        assert len(flat_past_kv) == GPT2_LAYERS * 2

        # Position embeddings
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        # Build additive 4D attention mask: (1, 1, 1, total_seq_len)
        attn_mask_4d = (
            (1.0 - attention_mask.to(hidden_states.dtype).view(1, 1, 1, -1)) * -1.0e9
        )

        present_kv = []
        for layer_idx in range(self.n_layer):
            past_k = flat_past_kv[2 * layer_idx]
            past_v = flat_past_kv[2 * layer_idx + 1]
            hidden_states, present_k, present_v = self._block_forward(
                self.h[layer_idx], hidden_states, past_k, past_v, attn_mask_4d
            )
            present_kv.append(present_k)
            present_kv.append(present_v)

        hidden_states = self.ln_f(hidden_states)
        logits = self.speech_head(hidden_states).squeeze(1)  # (1, 6563)
        return (logits, *present_kv)


def _lm_onnx_io_names():
    """Generate input/output names matching Swift OrtFastDecoder expectations."""
    inputs = ["inputs_embeds", "attention_mask", "position_ids"]
    for i in range(GPT2_LAYERS):
        inputs.extend([f"past_key_values.{i}.key", f"past_key_values.{i}.value"])
    outputs = ["logits"]
    for i in range(GPT2_LAYERS):
        outputs.extend([f"present.{i}.key", f"present.{i}.value"])
    return inputs, outputs


def _lm_onnx_dynamic_axes():
    """Define dynamic axes for variable sequence lengths."""
    axes = {
        "attention_mask": {1: "total_seq_len"},
    }
    for i in range(GPT2_LAYERS):
        axes[f"past_key_values.{i}.key"] = {2: "past_len"}
        axes[f"past_key_values.{i}.value"] = {2: "past_len"}
        axes[f"present.{i}.key"] = {2: "total_seq_len"}
        axes[f"present.{i}.value"] = {2: "total_seq_len"}
    return axes


def _load_t3_from_safetensors(weights_path):
    """Load T3 model from safetensors file."""
    from chatterbox.models.t3.t3 import T3, T3Config
    
    print(f"Loading T3 model from: {weights_path}")
    
    # Create model config matching GPT-2 Medium
    llama_configs = {"GPT2_medium": {
        "activation_function": "gelu_new",
        "architectures": ["GPT2LMHeadModel"],
        "attn_pdrop": 0.1,
        "bos_token_id": 50256,
        "embd_pdrop": 0.1,
        "eos_token_id": 50256,
        "initializer_range": 0.02,
        "layer_norm_epsilon": 1e-05,
        "model_type": "gpt2",
        "n_ctx": 8196,
        "n_embd": GPT2_HIDDEN,
        "hidden_size": GPT2_HIDDEN,
        "n_head": GPT2_HEADS,
        "n_layer": GPT2_LAYERS,
        "n_positions": 8196,
        "n_special": 0,
        "predict_special_tokens": True,
        "resid_pdrop": 0.1,
        "summary_activation": None,
        "summary_first_dropout": 0.1,
        "summary_proj_to_labels": True,
        "summary_type": "cls_index",
        "summary_use_proj": True,
        "vocab_size": TEXT_VOCAB_SIZE,
    }}
    
    # Patch into chatterbox if not present
    try:
        from chatterbox.models.t3 import llama_configs as lc_module
        if hasattr(lc_module, 'LLAMA_CONFIGS'):
            if "GPT2_medium" not in lc_module.LLAMA_CONFIGS:
                lc_module.LLAMA_CONFIGS["GPT2_medium"] = llama_configs["GPT2_medium"]
    except Exception:
        pass
    
    # Create T3 config
    turbo_cfg_dict = {
        "llama_config_name": "GPT2_medium",
        "speech_vocab_size": SPEECH_VOCAB_SIZE,
        "text_vocab_size": TEXT_VOCAB_SIZE,
        "speaker_emb_dim": 256,
        "campp_emb_dim": 192,
        "mel_bins": 80,
    }
    turbo_cfg = T3Config.__new__(T3Config)
    for k, v in turbo_cfg_dict.items():
        setattr(turbo_cfg, k, v)
    
    # Instantiate model and load weights
    t3 = T3(hp=turbo_cfg)
    state_dict = load_file(weights_path)
    
    # Filter and load state dict
    t3_state = {}
    for k, v in state_dict.items():
        if k.startswith("t3.") or k.startswith("tfmr."):
            new_key = k.replace("t3.", "") if k.startswith("t3.") else k
            t3_state[new_key] = v
    
    t3.load_state_dict(t3_state, strict=False)
    t3.to("cpu").train(False)
    
    print("  Model loaded successfully.")
    return t3


def _optimize_onnx_graph(onnx_path: str, model_type: str = "bert",
                          num_heads: int = GPT2_HEADS, 
                          hidden_size: int = GPT2_HIDDEN) -> None:
    """Run onnxruntime's graph optimizer to fuse ops into GroupQueryAttention.
    
    Applies operator fusion (LayerNorm, GELU, attention), constant folding,
    and ORT's L2 optimizations. This transforms standard attention patterns
    into Microsoft's GroupQueryAttention custom op for optimal performance.
    """
    try:
        from onnxruntime.transformers import optimizer
    except ImportError:
        print("  ERROR: onnxruntime-transformers not installed.")
        print("  Install with: pip install onnxruntime-transformers")
        sys.exit(1)
    
    print(f"  Optimizing graph (model_type={model_type!r}, full fusions)...")
    
    opt = optimizer.optimize_model(
        onnx_path,
        model_type=model_type,
        num_heads=num_heads,
        hidden_size=hidden_size,
        opt_level=1,
        only_onnxruntime=False,  # Enable transformer-specific fusions
    )
    
    # Save optimized model
    opt.save_model_to_file(onnx_path, use_external_data_format=True)
    
    new_size = os.path.getsize(onnx_path) / 1e6
    data_path = onnx_path + ".data"
    if os.path.exists(data_path):
        new_size += os.path.getsize(data_path) / 1e6
    
    print(f"  Optimized graph saved ({new_size:.1f} MB total)")
    
    # Print node distribution
    _print_node_distribution(onnx_path)


def _print_node_distribution(onnx_path: str):
    """Print the distribution of node types in the ONNX model."""
    try:
        import onnx
    except ImportError:
        print("  (Skipping node distribution - onnx not installed)")
        return
    
    model = onnx.load(onnx_path, load_external_data=False)
    
    node_counts = {}
    for node in model.graph.node:
        node_type = node.op_type
        node_counts[node_type] = node_counts.get(node_type, 0) + 1
    
    print("\n--- Node Type Distribution ---")
    for node_type, count in sorted(node_counts.items(), key=lambda x: -x[1]):
        print(f"  {node_type}: {count}")


def export_language_model_onnx(weights_path: str, output_dir: str, 
                                optimize_graph: bool = False):
    """Export T3 language model to ONNX format.
    
    Args:
        weights_path: Path to t3_turbo_finetuned_merged.safetensors
        output_dir: Directory to save ONNX model
        optimize_graph: Whether to apply ORT transformer optimization
    """
    print("\n=== Exporting language_model_single.onnx ===")
    
    # Create output directory
    onnx_dir = os.path.join(output_dir, "onnx")
    os.makedirs(onnx_dir, exist_ok=True)
    out_path = os.path.join(onnx_dir, "language_model_single.onnx")
    
    # Load model from safetensors
    t3 = _load_t3_from_safetensors(weights_path)
    
    # Wrap model for ONNX export
    wrapper = _LanguageModelWrapper(t3)
    wrapper.train(False)
    
    # Create dummy inputs for tracing
    past_len = 10
    embeds_t = torch.randn(1, 1, GPT2_HIDDEN, dtype=torch.float32) * 0.02
    mask_t = torch.ones(1, past_len + 1, dtype=torch.int64)
    pos_t = torch.tensor([[past_len]], dtype=torch.int64)
    flat_pkv_t = []
    for _ in range(GPT2_LAYERS):
        flat_pkv_t.append(torch.randn(1, GPT2_HEADS, past_len, GPT2_HEAD_DIM) * 0.1)
        flat_pkv_t.append(torch.randn(1, GPT2_HEADS, past_len, GPT2_HEAD_DIM) * 0.1)
    
    input_names, output_names = _lm_onnx_io_names()
    dynamic_axes = _lm_onnx_dynamic_axes()
    
    # Export to ONNX
    print(f"  Exporting to {out_path} (opset 17)...")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (embeds_t, mask_t, pos_t, *flat_pkv_t),
            out_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=17,
            do_constant_folding=True,
        )
    
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"  Wrote {size_mb:.1f} MB")
    
    # Apply optimization if requested
    if optimize_graph:
        _optimize_onnx_graph(out_path)
    
    print(f"\n  Export complete: {out_path}")
    if optimize_graph:
        print("  Graph optimized with GroupQueryAttention fusion.")


def main():
    parser = argparse.ArgumentParser(
        description="Export Chatterbox T3 Turbo to ONNX from safetensors weights"
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to t3_turbo_finetuned_merged.safetensors"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save ONNX model"
    )
    parser.add_argument(
        "--optimize-graph",
        action="store_true",
        help="Apply ORT transformer optimization (enables GroupQueryAttention)"
    )
    
    args = parser.parse_args()
    
    # Validate weights file exists
    if not os.path.exists(args.weights):
        print(f"ERROR: Weights file not found: {args.weights}")
        sys.exit(1)
    
    # Run export
    export_language_model_onnx(
        weights_path=args.weights,
        output_dir=args.output_dir,
        optimize_graph=args.optimize_graph
    )
    
    print("\n=== Done ===")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
