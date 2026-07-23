#!/usr/bin/env python3
"""
Export T3 language model to ONNX format following the exact logic from convert_chatterbox_coreml.py.

This script exports language_model_single.onnx - the GPT-2 single-step decoder with KV cache.
It uses the _LanguageModelWrapper pattern from the original script to avoid tracing issues.

Usage:
    python export_t3_onnx.py --weights t3_turbo_finetuned_merged.safetensors --output-dir ./onnx_models --optimize-graph
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

# Constants matching Chatterbox Turbo's architecture
GPT2_HIDDEN = 1024
GPT2_HEADS = 16
GPT2_LAYERS = 24
GPT2_MAX_POS = 8196
GPT2_HEAD_DIM = GPT2_HIDDEN // GPT2_HEADS  # 64
TEXT_VOCAB_SIZE = 50276


def _ensure_chatterbox_gpt2_config():
    """Patch chatterbox's llama_configs to include GPT2_medium config."""
    try:
        from chatterbox.models.t3 import llama_configs as _lc
    except ImportError:
        print("ERROR: 'chatterbox' library not found.")
        print("Please install it: pip install chatterbox-tts")
        sys.exit(1)
    
    if hasattr(_lc, "GPT2_medium"):
        return
    
    _GPT2_MEDIUM_CONFIG = {
        "activation_function": "gelu_new",
        "architectures": ["GPT2LMHeadModel"],
        "attn_pdrop": 0.1,
        "bos_token_id": 50256,
        "embd_pdrop": 0.1,
        "eos_token_id": 50256,
        "initializer_range": 0.02,
        "layer_norm_epsilon": 1e-05,
        "model_type": "gpt2",
        "n_ctx": GPT2_MAX_POS,
        "n_embd": GPT2_HIDDEN,
        "hidden_size": GPT2_HIDDEN,
        "n_head": GPT2_HEADS,
        "n_layer": GPT2_LAYERS,
        "n_positions": GPT2_MAX_POS,
        "n_special": 0,
        "predict_special_tokens": True,
        "resid_pdrop": 0.1,
        "summary_activation": None,
        "summary_first_dropout": 0.1,
        "summary_proj_to_labels": True,
        "summary_type": "cls_index",
        "summary_use_proj": True,
        "vocab_size": TEXT_VOCAB_SIZE,
    }
    _lc["GPT2_medium"] = _GPT2_MEDIUM_CONFIG


class _LanguageModelWrapper(nn.Module):
    """Stateful T3 wrapper for ONNX export.
    
    This is the EXACT wrapper from convert_chatterbox_coreml.py (lines 1027-1123).
    It re-implements GPT-2 attention using primitives to avoid tracing issues 
    with HuggingFace's Cache utility.
    
    Takes pre-computed embeddings (not token IDs) and manages per-layer KV cache 
    as separate inputs/outputs.
    """
    
    def __init__(self, t3_model):
        super().__init__()
        self.t3 = t3_model
        
        # Extract components from T3 model
        # T3 structure: t3.tfmr.h contains the 24 transformer blocks
        # t3.cond_enc.spkr_enc is the speaker encoder
        # t3.text_emb, t3.speech_emb are embeddings
        # t3.speech_head is the output head
        
        self.n_layer = GPT2_LAYERS
        self.n_head = GPT2_HEADS
        self.head_dim = GPT2_HEAD_DIM
        self.hidden = GPT2_HIDDEN
        
        # Register the transformer blocks
        self.h = t3_model.tfmr.h
        
        # Final layer norm
        self.ln_f = t3_model.tfmr.ln_f
        
        # Output head (speech prediction)
        self.speech_head = t3_model.speech_head
        
    def forward(self, inputs_embeds, attention_mask, position_ids, *past_key_values):
        """Forward pass with KV cache.
        
        Args:
            inputs_embeds: (batch, seq_len, hidden) - pre-computed embeddings
            attention_mask: (batch, total_seq_len) - attention mask
            position_ids: (batch, seq_len) - position indices
            past_key_values: 48 tensors (key, value for each of 24 layers)
                            Each: (batch, heads, past_len, head_dim)
        
        Returns:
            logits: (batch, seq_len, vocab_size) - speech token logits
            present_key_values: 48 tensors - updated KV cache
        """
        batch_size = inputs_embeds.shape[0]
        seq_len = inputs_embeds.shape[1]
        
        # Split past_key_values into list of (key, value) tuples
        pkv_list = []
        for i in range(self.n_layer):
            key = past_key_values[2 * i]
            value = past_key_values[2 * i + 1]
            pkv_list.append((key, value))
        
        hidden = inputs_embeds
        
        # Process each transformer block
        presents = []
        for i, block in enumerate(self.h):
            past_key, past_value = pkv_list[i]
            
            # Get block components
            # GPT2Block structure: ln_1, attn (c_attn, c_proj), ln_2, mlp (c_fc, c_proj)
            ln_1 = block.ln_1
            attn = block.attn
            ln_2 = block.ln_2
            mlp = block.mlp
            
            # Self-attention with manual implementation
            residual = hidden
            hidden = ln_1(hidden)
            
            # Compute Q, K, V
            query, key, value = attn.c_attn(hidden).split(self.hidden, dim=-1)
            
            # Reshape for multi-head attention
            query = query.view(batch_size, -1, self.n_head, self.head_dim).transpose(1, 2)
            key = key.view(batch_size, -1, self.n_head, self.head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, self.n_head, self.head_dim).transpose(1, 2)
            
            # Concatenate with past KV
            key = torch.cat([past_key, key], dim=-2)
            value = torch.cat([past_value, value], dim=-2)
            
            # Scaled dot-product attention
            attn_output = F.scaled_dot_product_attention(
                query, key, value, 
                attn_mask=None,
                dropout_p=0.0,
                is_causal=True
            )
            
            # Reshape back
            attn_output = attn_output.transpose(1, 2).contiguous().view(
                batch_size, -1, self.hidden
            )
            
            # Output projection
            hidden = residual + attn.c_proj(attn_output)
            
            # Store updated KV cache
            presents.append(key)
            presents.append(value)
            
            # MLP block
            residual = hidden
            hidden = ln_2(hidden)
            hidden = residual + mlp.c_proj(F.gelu(mlp.c_fc(hidden)))
        
        # Final layer norm
        hidden = self.ln_f(hidden)
        
        # Output head
        logits = self.speech_head(hidden)
        
        return (logits, *presents)


def _lm_onnx_io_names():
    """Generate input/output names matching the original script."""
    input_names = ["inputs_embeds", "attention_mask", "position_ids"]
    for i in range(GPT2_LAYERS):
        input_names.append(f"past_key_values.{i}.key")
        input_names.append(f"past_key_values.{i}.value")
    
    output_names = ["logits"]
    for i in range(GPT2_LAYERS):
        output_names.append(f"present.{i}.key")
        output_names.append(f"present.{i}.value")
    
    return input_names, output_names


def _lm_onnx_dynamic_axes():
    """Generate dynamic axes for ONNX export."""
    dynamic_axes = {
        "inputs_embeds": {1: "seq_len"},
        "attention_mask": {1: "total_seq_len"},
        "logits": {1: "seq_len"},
    }
    for i in range(GPT2_LAYERS):
        dynamic_axes[f"past_key_values.{i}.key"] = {2: "past_seq_len"}
        dynamic_axes[f"past_key_values.{i}.value"] = {2: "past_seq_len"}
        dynamic_axes[f"present.{i}.key"] = {2: "past_seq_len"}
        dynamic_axes[f"present.{i}.value"] = {2: "past_seq_len"}
    
    return dynamic_axes


def _load_t3_from_safetensors(weights_path):
    """Load T3 model from safetensors file following the original script's logic."""
    _ensure_chatterbox_gpt2_config()
    
    from chatterbox.models.t3.t3 import T3, T3Config
    import yaml
    
    # Load config from YAML (same as original script)
    # For finetuned models, we need to infer or use default config
    yaml_path = Path(weights_path).parent / "t3_turbo_v1.yaml"
    
    if yaml_path.exists():
        with open(yaml_path) as f:
            cfg_dict = yaml.full_load(f)
    else:
        # Create minimal config for finetuned model
        cfg_dict = {
            "llama_config_name": "GPT2_medium",
            "speaker_emb_dim": 256,
            "campp_emb_dim": 192,
            "encoder_type": "voice_encoder",
        }
    
    # Ensure GPT2_medium config
    cfg_dict["llama_config_name"] = "GPT2_medium"
    
    # Create T3Config object
    turbo_cfg = T3Config.__new__(T3Config)
    for k, v in cfg_dict.items():
        setattr(turbo_cfg, k, v)
    
    # Add missing attributes that might cause errors
    if not hasattr(turbo_cfg, 'speaker_embed_size'):
        turbo_cfg.speaker_embed_size = getattr(turbo_cfg, 'speaker_emb_dim', 256)
    
    # Create and load model
    t3 = T3(hp=turbo_cfg)
    t3_state = load_file(weights_path)
    t3.load_state_dict(t3_state)
    t3.to("cpu").train(False)
    
    return t3


def _optimize_onnx_graph(onnx_path, model_type="bert", num_heads=GPT2_HEADS, hidden_size=GPT2_HIDDEN):
    """Optimize ONNX graph using onnxruntime transformers optimizer (from original script)."""
    try:
        from onnxruntime.transformers import optimizer
        
        print(f"  Optimizing graph (model_type={model_type!r}, full fusions)...")
        opt = optimizer.optimize_model(
            onnx_path,
            model_type=model_type,
            num_heads=num_heads,
            hidden_size=hidden_size,
            opt_level=1,
            only_onnxruntime=False,
        )
        opt.save_model_to_file(onnx_path, use_external_data_format=True)
        
        new_size = os.path.getsize(onnx_path) / 1e6
        data_path = onnx_path + ".data"
        if os.path.exists(data_path):
            new_size += os.path.getsize(data_path) / 1e6
        print(f"  Optimized graph saved ({new_size:.1f} MB total)")
        
    except Exception as exc:
        print(f"  Optimization failed ({type(exc).__name__}): {str(exc)[:200]}")
        print("  Retrying with ORT-only optimizations...")
        try:
            from onnxruntime.transformers import optimizer
            opt = optimizer.optimize_model(
                onnx_path,
                model_type="bert",
                num_heads=num_heads,
                hidden_size=hidden_size,
                opt_level=1,
                only_onnxruntime=True,
            )
            opt.save_model_to_file(onnx_path, use_external_data_format=True)
            print("  ORT-only optimization successful")
        except Exception as exc2:
            print(f"  Graph optimization disabled — {type(exc2).__name__}: {str(exc2)[:200]}")


def count_node_types(onnx_path):
    """Count node types in ONNX model."""
    import onnx
    from collections import Counter
    
    # Load model (handle external data)
    model = onnx.load(onnx_path)
    
    # Count node types
    node_counts = Counter(node.op_type for node in model.graph.node)
    return node_counts


def print_node_distribution(node_counts):
    """Print node type distribution in the requested format."""
    print("\n--- Node Type Distribution ---")
    for op_type, count in sorted(node_counts.items()):
        print(f"  {op_type}: {count}")


def export_language_model_onnx(weights_path, output_dir, optimize_graph=False):
    """Export T3 language model to ONNX following the exact logic from convert_chatterbox_coreml.py."""
    print(f"\n=== Exporting language_model_single.onnx ===")
    print(f"Loading T3 model from: {weights_path}")
    
    # Load T3 model
    t3 = _load_t3_from_safetensors(weights_path)
    
    # Create wrapper
    wrapper = _LanguageModelWrapper(t3)
    wrapper.train(False)
    
    # Create output directory
    onnx_dir = os.path.join(output_dir, "onnx")
    os.makedirs(onnx_dir, exist_ok=True)
    out_path = os.path.join(onnx_dir, "language_model_single.onnx")
    
    # Prepare dummy inputs (matching original script)
    past_len = 10
    embeds_t = torch.randn(1, 1, GPT2_HIDDEN, dtype=torch.float32) * 0.02
    mask_t = torch.ones(1, past_len + 1, dtype=torch.int64)
    pos_t = torch.tensor([[past_len]], dtype=torch.int64)
    
    flat_pkv_t = []
    for _ in range(GPT2_LAYERS):
        flat_pkv_t.append(torch.randn(1, GPT2_HEADS, past_len, GPT2_HEAD_DIM) * 0.1)
        flat_pkv_t.append(torch.randn(1, GPT2_HEADS, past_len, GPT2_HEAD_DIM) * 0.1)
    
    # Get IO names and dynamic axes
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
    
    # Optimize if requested
    if optimize_graph:
        _optimize_onnx_graph(out_path)
    
    # Count and print node types
    node_counts = count_node_types(out_path)
    print_node_distribution(node_counts)
    
    print(f"\nONNX model saved to: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Export T3 language model to ONNX (following convert_chatterbox_coreml.py)"
    )
    parser.add_argument(
        "--weights", 
        type=str, 
        required=True,
        help="Path to the .safetensors file (e.g., t3_turbo_finetuned_merged.safetensors)"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="./onnx_models",
        help="Output directory for ONNX model"
    )
    parser.add_argument(
        "--optimize-graph", 
        action="store_true",
        help="Apply ONNX graph optimization (enables GroupQueryAttention fusion)"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.weights):
        print(f"ERROR: Weights file not found: {args.weights}")
        sys.exit(1)
    
    # Export model
    export_language_model_onnx(
        args.weights,
        args.output_dir,
        optimize_graph=args.optimize_graph
    )


if __name__ == "__main__":
    main()
