#!/usr/bin/env python3
"""
Standalone script to export T3 Turbo (GPT-2 based) to ONNX.
Logic extracted from convert_chatterbox_coreml.py but self-contained.
Avoids modifying external library configs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import sys
from collections import Counter
from safetensors.torch import load_file

# Try to import onnxruntime for optimization, but make it optional for the export step
try:
    from onnxruntime.transformers.optimizer import optimize_model
    HAS_ORT = True
except ImportError:
    HAS_ORT = False
    print("Warning: onnxruntime-transformers not found. Graph optimization (GroupQueryAttention) will be skipped.")

# Try to import onnx-simplifier for graph simplification
try:
    import onnxsim
    HAS_ONNXSIM = True
except ImportError:
    HAS_ONNXSIM = False
    print("Warning: onnx-simplifier not found. Run 'pip install onnx-simplifier' for better graph optimization.")

class T3TransformerBlock(nn.Module):
    """Single Transformer Block (GPT-2 style)"""
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        
        self.n_head = n_head
        self.n_embd = n_embd
        
        # Custom attention to support fused c_attn loading
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp_c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.mlp_c_proj = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x):
        B, T, C = x.size()
        n_head = self.n_head
        head_dim = C // n_head
        
        # Self Attention
        ln1 = self.ln_1(x)
        qkv = self.c_attn(ln1)
        q, k, v = qkv.split(C, dim=-1)
        
        # Reshape for multi-head
        q = q.view(B, T, n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, n_head, head_dim).transpose(1, 2)
        
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        y = self.c_proj(y)
        x = x + y
        
        # MLP
        x = x + self.mlp_c_proj(F.gelu(self.mlp_c_fc(self.ln_2(x))))
        return x

class T3Model(nn.Module):
    """Simplified T3 Model (Language Model part only)"""
    def __init__(self, vocab_size, n_layer, n_head, n_embd, block_size=2048):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.block_size = block_size
        
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        
        self.h = nn.ModuleList([
            T3TransformerBlock(n_embd, n_head) for _ in range(n_layer)
        ])
        
        self.ln_f = nn.LayerNorm(n_embd)
        self.text_head = nn.Linear(n_embd, vocab_size, bias=False)
        
        # Register causal mask as buffer to avoid dynamic computation
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size).bool()
        )
        
        # Register positional indices as buffer to avoid dynamic arange
        self.register_buffer(
            "pos_indices",
            torch.arange(block_size).long().unsqueeze(0)
        )

    def forward(self, idx):
        b, t = idx.size()
        
        # Use pre-computed positional indices (slice from buffer)
        # This avoids dynamic arange which prevents constant folding
        pos = self.pos_indices[:, :t]
        
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)
        x = tok_emb + pos_emb
        
        for block in self.h:
            x = block(x)
        
        x = self.ln_f(x)
        logits = self.text_head(x)
        
        return logits

def infer_config_from_keys(keys):
    """Infer model configuration from state dict keys."""
    n_layer = 0
    
    layer_keys = [k for k in keys if '.h.' in k]
    if not layer_keys:
        raise ValueError("No transformer layers found in state dict.")
    
    max_idx = -1
    for k in layer_keys:
        try:
            part = k.split('.h.')[1].split('.')[0]
            if part.isdigit():
                idx = int(part)
                max_idx = max(max_idx, idx)
        except:
            continue
    n_layer = max_idx + 1
    
    # T3 Turbo is typically GPT-2 Medium: 24 layers, 16 heads, 1024 embd
    if n_layer == 24:
        n_embd = 1024
        n_head = 16
    elif n_layer == 12:
        n_embd = 768
        n_head = 12
    elif n_layer == 36:
        n_embd = 1280
        n_head = 16
    else:
        print(f"Warning: Unknown layer count {n_layer}. Assuming GPT-2 Medium config.")
        n_embd = 1024
        n_head = 16
        
    return n_layer, n_head, n_embd

def load_and_map_weights(model, state_dict):
    """Map weights from the flat state dict to the model."""
    new_sd = {}
    
    for k, v in state_dict.items():
        new_k = k
        
        # Strip 'tfmr.' prefix if present
        if new_k.startswith('tfmr.'):
            new_k = new_k[5:]
        
        # Map attention submodules
        if '.attn.c_attn.' in new_k:
            new_k = new_k.replace('.attn.c_attn.', '.c_attn.')
        elif '.attn.c_proj.' in new_k:
            new_k = new_k.replace('.attn.c_proj.', '.c_proj.')
        elif '.mlp.c_fc.' in new_k:
            new_k = new_k.replace('.mlp.c_fc.', '.mlp_c_fc.')
        elif '.mlp.c_proj.' in new_k:
            new_k = new_k.replace('.mlp.c_proj.', '.mlp_c_proj.')
            
        # Map embeddings and head
        if new_k == 'text_emb.weight':
            new_k = 'wte.weight'
        elif new_k == 'text_head.weight':
            new_k = 'text_head.weight'
        elif new_k == 'ln_f.weight':
            new_k = 'ln_f.weight'
        elif new_k == 'ln_f.bias':
            new_k = 'ln_f.bias'
        
        if new_k in model.state_dict():
            if v.shape == model.state_dict()[new_k].shape:
                new_sd[new_k] = v
            else:
                print(f"Shape mismatch for {new_k}: file {v.shape} vs model {model.state_dict()[new_k].shape}")
        # Ignore unused keys (speaker encoder, etc.)

    # Check for missing keys
    model_sd = model.state_dict()
    missing = set(model_sd.keys()) - set(new_sd.keys())
    
    if 'wpe.weight' in missing:
        print("Warning: Positional embeddings (wpe) not found in file. Initializing randomly.")
        missing.remove('wpe.weight')
        
    if missing:
        print(f"Error: Could not map the following keys: {missing}")
        
    model.load_state_dict(new_sd, strict=False)
    return model

def export_onnx(model, output_path, seq_len=128):
    model.eval()
    device = next(model.parameters()).device
    
    dummy_input = torch.randint(0, model.vocab_size, (1, seq_len), dtype=torch.long).to(device)
    
    print(f"Exporting to ONNX (Opset 17) with fixed sequence length {seq_len}...")
    print("Note: Using static shapes (no dynamic axes) to enable constant folding in onnx-simplifier.")
    
    # Export with NO dynamic axes - completely static graph for maximum optimization
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input_ids'],
        output_names=['logits'],
        dynamic_axes=None,  # No dynamic axes - fully static
        verbose=False,
        dynamo=False
    )
    print(f"Saved raw ONNX to {output_path}")

def simplify_onnx(output_path):
    """Simplify ONNX model using onnx-simplifier to reduce unnecessary ops."""
    if not HAS_ONNXSIM:
        print("Skipping onnx-simplifier optimization (not installed).")
        return
    
    print("Simplifying ONNX graph with onnx-simplifier...")
    try:
        import onnx
        model = onnx.load(output_path)
        model_simp, check = onnxsim.simplify(model)
        assert check, "Simplified ONNX model validation failed"
        
        onnx.save(model_simp, output_path)
        print(f"Saved simplified ONNX to {output_path}")
        
        # Print node distribution after simplification
        nodes = [node.op_type for node in model_simp.graph.node]
        counts = Counter(nodes)
        
        print("\n--- Node Type Distribution (After Simplification) ---")
        for op, count in sorted(counts.items()):
            print(f"  {op}: {count}")
            
    except Exception as e:
        print(f"Simplification failed: {e}")

def optimize_onnx(output_path):
    if not HAS_ORT:
        return
    
    print("Optimizing ONNX graph (converting to GroupQueryAttention)...")
    try:
        num_heads = 16
        hidden_size = 1024
        
        opt_model = optimize_model(
            output_path,
            model_type='gpt2',
            num_heads=num_heads,
            hidden_size=hidden_size,
            opt_level=99,
            only_onnxruntime=False
        )
        
        opt_model.save_model_to_file(output_path)
        print(f"Saved optimized ONNX to {output_path}")
        
        import onnx
        onnx_model = onnx.load(output_path)
        nodes = [node.op_type for node in onnx_model.graph.node]
        counts = Counter(nodes)
        
        print("\n--- Node Type Distribution ---")
        for op, count in sorted(counts.items()):
            print(f"  {op}: {count}")
            
    except Exception as e:
        print(f"Optimization failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="Export T3 to ONNX")
    parser.add_argument("--weights", type=str, required=True, help="Path to .safetensors")
    parser.add_argument("--output-dir", type=str, default="./onnx_models", help="Output dir")
    parser.add_argument("--seq-len", type=int, default=128, help="Sequence length for export trace")
    parser.add_argument("--simplify", action="store_true", help="Run onnx-simplifier to reduce ops")
    parser.add_argument("--optimize", action="store_true", help="Run ORT optimizer")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading weights from {args.weights}...")
    state_dict = load_file(args.weights)
    keys = list(state_dict.keys())
    
    print(f"--- Inspecting State Dict ({len(keys)} keys) ---")
    for k in keys[:10]:
        print(f"  {k}")
    
    n_layer, n_head, n_embd = infer_config_from_keys(keys)
    
    vocab_size = 50276
    for k, v in state_dict.items():
        if k == 'text_emb.weight':
            vocab_size = v.shape[0]
            break
            
    print(f"Inferred Config: Layers={n_layer}, Heads={n_head}, Embd={n_embd}, Vocab={vocab_size}")
    
    print("Initializing model...")
    model = T3Model(vocab_size=vocab_size, n_layer=n_layer, n_head=n_head, n_embd=n_embd)
    
    print("Mapping weights...")
    model = load_and_map_weights(model, state_dict)
    
    out_path = os.path.join(args.output_dir, "language_model_single.onnx")
    export_onnx(model, out_path, seq_len=args.seq_len)
    
    if args.simplify:
        simplify_onnx(out_path)
    
    if args.optimize:
        optimize_onnx(out_path)
    
    print("Done.")

if __name__ == "__main__":
    main()
