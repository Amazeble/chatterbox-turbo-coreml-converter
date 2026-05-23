#!/usr/bin/env python3
"""Microbenchmark the converted v4 artifacts against the vanilla PyTorch path.

Times each stage of the pipeline in isolation:
  - PyTorch end-to-end (chatterbox.generate)
  - T3Prefill.mlpackage prefill (one call)
  - language_model_single.onnx decode (per-step latency, averaged)
  - conditional_decoder_single.onnx synthesize (one call)

The point is *not* to measure speedup vs PyTorch end-to-end (the converted
pipeline runs on different runtimes per stage). It's to show why we split
the pipeline the way we did: prefill is one big CoreML call, decode is a
tight per-step ONNX loop, conditional_decoder is one big ONNX call.

Run on M1 (or any Apple Silicon Mac). CoreML compute units chosen to match
what pooler-core does on iPhone (cpuAndGPU). Results vary with hardware /
thermal state; treat numbers as ballpark, not absolute.
"""

import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import torch
import onnxruntime as ort
import coremltools as ct

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from convert_chatterbox_coreml import (
    _ensure_chatterbox_gpt2_config,
    _make_fixture_lm_onnx,
    _make_fixture_prefill,
    _make_fixture_cond_decoder,
    GPT2_LAYERS,
)


def _bench(fn, name, runs=5, warmup=1):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    mean = sum(times) / len(times)
    print(f"  {name:40s}  mean={mean*1000:7.1f} ms  min={min(times)*1000:7.1f} ms  (n={runs})")
    return mean


def bench_pytorch_end_to_end():
    """Vanilla chatterbox PyTorch generate() — single-call wall time."""
    print("\n=== PyTorch end-to-end (chatterbox.ChatterboxTurboTTS.generate) ===")
    _ensure_chatterbox_gpt2_config()
    from chatterbox.tts_turbo import ChatterboxTurboTTS

    tts = ChatterboxTurboTTS.from_pretrained("cpu")
    text = "Hello there. This is a benchmark of the chatterbox turbo model."

    print("  Warmup (1)...")
    with torch.inference_mode():
        _ = tts.generate(text=text)

    print("  Timed (3)...")
    n_tokens_seen = []
    captured = {}
    _orig = tts.t3.inference_turbo

    def _capture(*a, **kw):
        out = _orig(*a, **kw)
        captured["tok_count"] = (out.shape[-1] if torch.is_tensor(out) else len(out))
        return out

    tts.t3.inference_turbo = _capture

    times = []
    for i in range(3):
        t0 = time.perf_counter()
        with torch.inference_mode():
            audio = tts.generate(text=text)
        dt = time.perf_counter() - t0
        n_samples = audio.shape[-1]
        n_tokens = captured.get("tok_count", 0)
        times.append((dt, n_tokens, n_samples))
        print(
            f"    run {i+1}: {dt*1000:7.1f} ms total | "
            f"{n_tokens} speech tokens | {n_samples} samples ({n_samples/24000:.2f}s audio)"
        )

    mean = sum(t for t, _, _ in times) / len(times)
    avg_tokens = sum(n for _, n, _ in times) / len(times)
    avg_samples = sum(s for _, _, s in times) / len(times)
    audio_sec = avg_samples / 24000
    rtf = mean / audio_sec
    print(f"  mean={mean*1000:.1f} ms | avg tokens={avg_tokens:.0f} | avg audio={audio_sec:.2f}s | RTF={rtf:.2f}x")
    return mean, avg_tokens, audio_sec


def bench_prefill_mlpackage():
    """T3Prefill.mlpackage — single prefill call latency."""
    print("\n=== T3Prefill.mlpackage (CoreML CPU+GPU) ===")
    pkg = ROOT / "out" / "T3Prefill.mlpackage"
    if not pkg.exists():
        print(f"  SKIP: {pkg} not found — run --stage prefill first")
        return None

    print(f"  Loading {pkg.name}...")
    ml = ct.models.MLModel(str(pkg), compute_units=ct.ComputeUnit.CPU_AND_GPU)
    text_t, cond_t, spkr_t, spch_t = _make_fixture_prefill(seed=0)
    inputs = {
        "text_tokens": text_t,
        "cond_speech_tokens": cond_t,
        "speaker_emb": spkr_t,
        "speech_tokens": spch_t,
    }
    print(
        f"  Fixture: T_text={text_t.shape[1]}, T_cond={cond_t.shape[1]}, T_speech={spch_t.shape[1]}"
    )
    return _bench(lambda: ml.predict(inputs), "prefill call", runs=5)


def bench_lm_onnx_decode():
    """language_model_single.onnx — per-step decode latency at growing KV length."""
    print("\n=== language_model_single.onnx (ONNX Runtime CPU) per-step decode ===")
    path = ROOT / "out" / "onnx" / "language_model_single.onnx"
    if not path.exists():
        print(f"  SKIP: {path} not found — run --stage lm-onnx first")
        return None

    print(f"  Loading {path.name}...")
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = 4
    sess = ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])

    # Measure at three KV cache lengths (decode gets slower as cache grows).
    means = {}
    for past_len in (10, 100, 500):
        embeds, mask, pos, kv_pairs = _make_fixture_lm_onnx(seed=0, past_len=past_len)
        feed = {
            "inputs_embeds": embeds,
            "attention_mask": mask,
            "position_ids": pos,
        }
        for i, (k, v) in enumerate(kv_pairs):
            feed[f"past_key_values.{i}.key"] = k
            feed[f"past_key_values.{i}.value"] = v
        out_names = ["logits"] + sum(
            ([f"present.{i}.key", f"present.{i}.value"] for i in range(GPT2_LAYERS)), []
        )
        means[past_len] = _bench(lambda: sess.run(out_names, feed), f"decode @ past_len={past_len}", runs=10)
    return means


def bench_cond_decoder_onnx():
    """conditional_decoder_single.onnx — full audio synthesis call."""
    print("\n=== conditional_decoder_single.onnx (ONNX Runtime CPU) ===")
    path = ROOT / "out" / "onnx" / "conditional_decoder_single.onnx"
    if not path.exists():
        print(f"  SKIP: {path} not found — run --stage cond-decoder first")
        return None

    print(f"  Loading {path.name}...")
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = 4
    sess = ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])

    # Typical "5-second utterance" sizing: ~140 tokens (prompt+gen), 500 mel frames.
    speech_t, spk_t, feat_t = _make_fixture_cond_decoder(seed=0, n_prompt_tokens=50, n_gen_tokens=90)
    inputs = {
        "speech_tokens": speech_t,
        "speaker_embeddings": spk_t,
        "speaker_features": feat_t,
    }
    print(
        f"  Fixture: speech_tokens={speech_t.shape}, "
        f"speaker_features={feat_t.shape}"
    )
    return _bench(lambda: sess.run(["waveform"], inputs), "synth call", runs=3)


def main():
    print("Chatterbox Turbo CoreML/ONNX pipeline microbenchmarks")
    print("Hardware:", os.uname().machine, "/", os.uname().sysname, os.uname().release)
    print()

    pt = bench_pytorch_end_to_end()
    prefill = bench_prefill_mlpackage()
    lm = bench_lm_onnx_decode()
    cond = bench_cond_decoder_onnx()

    print("\n=== Summary (ms per call, mean) ===")
    if pt is not None:
        pt_ms, n_tok, audio_s = pt
        print(f"  PyTorch end-to-end          {pt_ms*1000:7.1f}  ({n_tok:.0f} tok -> {audio_s:.2f}s audio)")
    if prefill is not None:
        print(f"  T3Prefill.mlpackage         {prefill*1000:7.1f}  (one prefill)")
    if lm is not None:
        for past_len, t in lm.items():
            print(f"  language_model.onnx @ {past_len:>4d}  {t*1000:7.1f}  (per decode step)")
        # Estimate decode-loop time for the same token count as the PyTorch run
        if pt is not None:
            est = lm.get(100, lm.get(500, 0)) * n_tok
            print(f"  est. {int(n_tok)}-token decode loop  {est*1000:7.1f}  (lm.onnx, past_len~100 avg)")
    if cond is not None:
        print(f"  conditional_decoder.onnx    {cond*1000:7.1f}  (one synth)")


if __name__ == "__main__":
    main()
