#!/usr/bin/env python3
"""Generate two sample WAVs and open them in QuickTime for an A/B listen.

  out/sample_pytorch.wav  - upstream chatterbox PyTorch path (HF reference)
  out/sample_ours.wav     - same speech tokens + our converted Stage C ONNX

The PyTorch sample shows what the model SHOULD sound like. Our sample shows
what the converted conditional_decoder produces with the harmonic source
path bypassed (the Stage C tradeoff documented in README).
"""

import os
import sys
import warnings
import subprocess
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import torch
import onnxruntime as ort
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from convert_chatterbox_coreml import _ensure_chatterbox_gpt2_config

_ensure_chatterbox_gpt2_config()
from chatterbox.tts_turbo import ChatterboxTurboTTS

TEXT = (
    "Hello there. This is a test of the chatterbox turbo model "
    "running through our converted ONNX pipeline."
)

# --- load full TTS -------------------------------------------------------
print("Loading ChatterboxTurboTTS.from_pretrained('cpu')...")
tts = ChatterboxTurboTTS.from_pretrained("cpu")

# ChatterboxTurboTTS.generate calls t3.inference_turbo (not t3.inference).
# Hook that to capture the generated speech tokens.
captured = {}
_orig = tts.t3.inference_turbo

def _capture(*args, **kwargs):
    out = _orig(*args, **kwargs)
    if torch.is_tensor(out):
        captured["raw_speech_tokens"] = out.detach().cpu().clone()
    elif isinstance(out, (tuple, list)) and torch.is_tensor(out[0]):
        captured["raw_speech_tokens"] = out[0].detach().cpu().clone()
    return out

tts.t3.inference_turbo = _capture

# --- generate PyTorch reference -----------------------------------------
print(f"\nGenerating PyTorch reference for: {TEXT!r}")
with torch.inference_mode():
    audio_pt = tts.generate(text=TEXT)
audio_pt = audio_pt.squeeze().cpu().numpy().astype(np.float32)

out_dir = ROOT / "out"
out_dir.mkdir(parents=True, exist_ok=True)
pt_path = out_dir / "sample_pytorch.wav"
sf.write(str(pt_path), audio_pt, 24000)
print(f"  Wrote {pt_path}  ({audio_pt.shape[0]} samples = {audio_pt.shape[0]/24000:.2f}s)")

# --- prep inputs for our ONNX Stage C ------------------------------------
raw_tokens = captured["raw_speech_tokens"].squeeze().numpy().astype(np.int64)
print(f"\n  T3 produced {len(raw_tokens)} speech tokens")

# Filter to valid range for the decoder (0..6560) — Swift does this
valid = raw_tokens[(raw_tokens >= 0) & (raw_tokens <= 6560)]
print(f"  After filtering invalid: {len(valid)}")

conds = tts.conds
gen_cond = conds.gen

def _to_np(t):
    if torch.is_tensor(t):
        return t.detach().cpu().numpy()
    return np.asarray(t)

prompt_tokens = _to_np(gen_cond["prompt_token"]).astype(np.int64).reshape(-1)
prompt_feat = _to_np(gen_cond["prompt_feat"]).astype(np.float32)
camp_emb = _to_np(gen_cond["embedding"]).astype(np.float32).reshape(-1)
print(
    f"  prompt_tokens={prompt_tokens.shape} prompt_feat={prompt_feat.shape} "
    f"camp_emb={camp_emb.shape}"
)

all_tokens = np.concatenate([prompt_tokens, valid]).astype(np.int64).reshape(1, -1)

norm = float(np.linalg.norm(camp_emb))
camp_norm = (camp_emb / max(norm, 1e-12)).astype(np.float32).reshape(1, -1)

if prompt_feat.ndim == 2:
    spk_feat = prompt_feat.reshape(1, *prompt_feat.shape).astype(np.float32)
else:
    spk_feat = prompt_feat.astype(np.float32)

print(f"\n  ONNX inputs:")
print(f"    speech_tokens      {all_tokens.shape} {all_tokens.dtype}")
print(f"    speaker_embeddings {camp_norm.shape} {camp_norm.dtype}")
print(f"    speaker_features   {spk_feat.shape} {spk_feat.dtype}")

# --- run our converted ONNX ----------------------------------------------
our_path = out_dir / "onnx" / "conditional_decoder_single.onnx"
print(f"\n  Running {our_path.name}...")
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess = ort.InferenceSession(str(our_path), sess_options=so, providers=["CPUExecutionProvider"])
audio_ours = sess.run(["waveform"], {
    "speech_tokens": all_tokens,
    "speaker_embeddings": camp_norm,
    "speaker_features": spk_feat,
})[0].squeeze().astype(np.float32)

ours_path = out_dir / "sample_ours.wav"
sf.write(str(ours_path), audio_ours, 24000)
print(f"  Wrote {ours_path}  ({audio_ours.shape[0]} samples = {audio_ours.shape[0]/24000:.2f}s)")

# --- open both in QuickTime ----------------------------------------------
print("\nOpening both in QuickTime Player...")
subprocess.run(["open", "-a", "QuickTime Player", str(pt_path), str(ours_path)])
