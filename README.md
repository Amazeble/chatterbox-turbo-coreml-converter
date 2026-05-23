# chatterbox-turbo-coreml-converter

Converts ResembleAI's [Chatterbox Turbo](https://huggingface.co/ResembleAI/chatterbox-turbo)
TTS model to Apple CoreML format for on-device inference on Apple Silicon.

Produces a stateful T3 decoder (GPT-2 medium with `StateType` KV cache), the S3
encoder + U-Net denoiser, vocoder weights, and tokenizer/config files. The
converted weights are published at
[ebrinz/chatterbox-turbo-coreml](https://huggingface.co/ebrinz/chatterbox-turbo-coreml)
if you just want to download and use them.

## What gets produced

Running `--stage all` writes the following into your output directory:

| File | What it is | Backend |
|---|---|---|
| `T3Stateful.mlpackage` | GPT-2 medium decoder with CoreML `StateType` KV cache (2D seq-first layout, dim-0 slice-update). Fixed seq=1 shape; prefill is done token-by-token through the same model. | GPU |
| `S3Encoder.mlpackage` | Conformer encoder + mel projection. Dynamic sequence length via `RangeDim`. | ANE-eligible |
| `S3UNet.mlpackage` | Flow-matching U-Net denoiser (estimator + speaker-affine projection baked in). | ANE-eligible |
| `hift_vocoder.safetensors` | HiFTGenerator vocoder weights (PyTorch — not converted to CoreML). | — |
| `speech_emb.npy`, `text_emb.npy` | T3 embedding tables, for host-side embedding lookup before calling `T3Stateful`. | — |
| `spkr_enc_weight.npy`, `spkr_enc_bias.npy` | Speaker conditioning linear layer weights. | — |
| `tokenizer.json` (+ vocab/merges) | GPT-2 BPE tokenizer copied from the upstream model. | — |
| `config.json` | Architecture constants (vocab sizes, mel bins, hidden dims, etc.). | — |

## Requirements

- **macOS on Apple Silicon** (M1/M2/M3/M4). Required for CoreML conversion and
  for the validation step's `make_state()` call.
- **Python 3.10+**
- **Minimum deployment target**: iOS 18 / macOS 15 (uses `ct.StateType`).
- ~20 GB free disk for model weights + intermediate artifacts.

## Install

```bash
git clone https://github.com/ebrinz/chatterbox-turbo-coreml-converter.git
cd chatterbox-turbo-coreml-converter
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The `chatterbox` PyTorch package is installed from ResembleAI's repo (see
`requirements.txt`).

## Usage

Convert everything in one go:

```bash
python convert_chatterbox_coreml.py --stage all --output-dir ./out
```

Or run individual stages:

```bash
python convert_chatterbox_coreml.py --stage t3       --output-dir ./out
python convert_chatterbox_coreml.py --stage s3       --output-dir ./out
python convert_chatterbox_coreml.py --stage vocoder  --output-dir ./out
```

Add `--validate` to T3 or S3 stages to run a quick numerical sanity check
against the PyTorch reference after conversion.

First run will download the ~6 GB Chatterbox Turbo weights from HuggingFace
into your HF cache.

## Using the converted models

The `.mlpackage` files are standard CoreML artifacts and load with the usual
APIs. You can verify them in Python before wiring up a host application:

```python
import coremltools as ct
import numpy as np

model = ct.models.MLModel("out/T3Stateful.mlpackage",
                         compute_units=ct.ComputeUnit.CPU_AND_GPU)
state = model.make_state()

# Run one decode step (host is responsible for embedding lookup + masking)
out = model.predict({
    "inputs_embeds": np.zeros((1, 1, 1024), dtype=np.float32),
    "position_ids":  np.array([[0]], dtype=np.int32),
    "cache_position": np.array([0], dtype=np.int32),
    "attention_mask": np.zeros((1, 1, 1, 2048), dtype=np.float32),  # 0 valid, -1e9 unfilled
}, state=state)

print(out["logits"].shape)  # (1, 6563)
```

To synthesize audio end-to-end you'll need to drive the full pipeline yourself:

1. Tokenize text (GPT-2 BPE from `tokenizer.json`).
2. Apply speaker conditioning: `cond_emb = spkr_enc_weight @ speaker_emb + spkr_enc_bias`.
3. Run `T3Stateful` decode loop: feed conditioning at position 0, then loop
   through speech embeddings, sampling next token from `logits` each step until
   the stop token (6562) is produced. Update `attention_mask` to mark filled
   positions (0) vs. unfilled (-1e9). KV cache state is carried via the
   `state` object.
4. Embed the speech tokens and run `S3Encoder` once on the concatenated
   prompt+speech sequence.
5. Run the CFM solver: `S3UNet` predicts velocity over N timesteps (the
   upstream config uses 2). Integrate to produce a mel spectrogram.
6. Run the HiFiGAN vocoder on the mel spectrogram (PyTorch — load
   `hift_vocoder.safetensors` into a `HiFTGenerator` from the `chatterbox`
   package) to produce 24 kHz waveform audio.

A standalone Python reference implementation of the full pipeline is on the
roadmap.

## Architecture notes

The interesting design decision in this converter is how the T3 decoder's KV
cache is shaped for CoreML's `StateType`. CoreML state tensors only support
dynamic `slice_update` on dimension 0, which rules out the natural
`(batch, heads, seq, head_dim)` layout per layer. Instead this converter uses
a single 2D seq-first cache shared across layers:

```
keyCache, valueCache: (max_seq=2048, n_layers * n_heads * head_dim = 24576)
```

Writes are dim-0 slices at the current cache position; per-layer K/V are
extracted by slicing the feature dimension. Attention masking handles unfilled
positions (zeros) by setting `attn_mask` to `-1e9` there, so the softmax
ignores them — this lets the model expose a fixed read shape without a dynamic
slice on read, which CoreML doesn't reliably support on state tensors either.

GPT-2 attention is monkey-patched at conversion time to use this cache and to
remove cross-attention / upcast paths that interfere with `torch.jit.trace`.

The T3 model is exported with a fixed seq=1 shape; prefill is done by calling
the same model in a loop. `EnumeratedShapes` and `RangeDim` both produced
CoreML error -14 on stateful models in testing.

## License

This conversion script is released under the [MIT License](LICENSE).

The **converted model weights** are derived from ResembleAI's
[Chatterbox](https://github.com/resemble-ai/chatterbox) and inherit their
licensing terms. Check the upstream license before redistributing the
converted artifacts.

## Acknowledgements

- [ResembleAI](https://www.resemble.ai/) for releasing Chatterbox Turbo.
- [Apple's coremltools](https://github.com/apple/coremltools) team for the
  stateful conversion support.
