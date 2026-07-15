# Audio restoration / super-resolution / scoring tools

Optional post-processing steps for `generate_track.py`:
`generate N candidates -> keep the best (score_track.py) -> Apollo (de-smear/restore) -> AudioSR (bandwidth to ~24 kHz) -> mastering chain -> loudnorm -> MP3`.

Mastering can only polish whatever the diffusion model produced -- it can't
fix smeared transients, a mushy mix, or a wandering arrangement baked into
the raw generation. Seed-to-seed variance on these models is large, so
generating a few candidates per track and automatically keeping the
best-scoring one (`--candidates N` in `generate_track.py`, scored by
`score_track.py` at the project root) is usually a bigger quality lever than
the whole post-processing chain below. See `score_track.py`'s module
docstring for the metrics it uses and why.

All three ML tools below run in their own isolated venvs (invoked as
subprocesses from `generate_track.py`) because their dependency pins
conflict with the main project environment (old `transformers`/`numpy`
pins, older `torch`). None is required for normal generation -- all are
off by default (`--candidates 1`, `--enable-apollo-restoration`,
`--enable-audiosr-upscale`).

## Apollo (de-smear / restoration)

Vendored from the [patriotyk/Apollo](https://huggingface.co/spaces/patriotyk/Apollo)
Hugging Face Space, which bundles the [JusperLee/Apollo](https://github.com/JusperLee/Apollo)
model code + checkpoints in one place (the upstream GitHub repo alone doesn't
publish checkpoint links).

```sh
cd tools
git clone https://huggingface.co/spaces/patriotyk/Apollo apollo
uv venv --python 3.11 apollo/.venv
uv pip install --python apollo/.venv torch==2.7.1 torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu128
uv pip install --python apollo/.venv "numpy<2.0" huggingface_hub librosa \
    omegaconf ml_collections tqdm soundfile
```

Driven by `tools/apollo_infer.py` (tracked in git; expects the clone at
`tools/apollo/`). Checkpoints in `tools/apollo/weights/`: `apollo.bin` (general
codec de-smear, the `--apollo-checkpoint restore` default), `apollo_vocal.bin` /
`apollo_vocal2.bin` (vocal-focused), `apollo_model_uni.ckpt` (universal).

## AudioSR (bandwidth extension to 48 kHz / ~24 kHz)

[haoheliu/versatile_audio_super_resolution](https://github.com/haoheliu/versatile_audio_super_resolution),
installed from PyPI (`audiosr`).

```sh
cd tools
uv venv --python 3.10 audiosr_venv
uv pip install --python audiosr_venv torch==2.7.1 torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu128
uv pip install --python audiosr_venv audiosr matplotlib "setuptools<81"
# Re-pin torch afterwards -- installing `audiosr` alone drags in an
# unrelated, non-CUDA torch build as a transitive dependency:
uv pip install --python audiosr_venv "torch==2.7.1+cu128" "torchaudio==2.7.1+cu128" \
    --index-url https://download.pytorch.org/whl/cu128 --reinstall
```

Python 3.10 is required: `audiosr` pins `numpy<=1.23.5`, which has no wheels
for 3.12+. `setuptools<81` is required too: `audiosr`'s `librosa` dependency
still imports `pkg_resources`, removed in newer `setuptools`. `matplotlib` is
an undeclared runtime dependency of `audiosr` (missing from its own
`requirements.txt`).

Model checkpoints (`haoheliu/audiosr_basic` / `audiosr_speech`) auto-download
from Hugging Face on first run.

**Why a custom wrapper (`tools/audiosr_venv_infer.py`) instead of the `audiosr`
CLI directly:** the underlying model is mono-only (its own code reads channel
0 and silently discards the rest) and was trained on <=5.12s clips -- its
README warns longer input "may degrade model performance." The wrapper
processes in overlapping 5.12s windows recombined with a linear crossfade, so
a full-length track goes in and a full-length track comes out.

Stereo is handled via **mid/side, not independent left/right**: running the
generative model on L and R separately makes it hallucinate different high
frequencies per channel -- heard as decorrelated/"phasey" top end, exactly
the artifact this step is supposed to remove. Only the mid (mono sum) goes
through the diffusion model; the side channel is plain-resampled (no
generation) to the new sample rate, preserving real recorded stereo width
below the original Nyquist while keeping all newly-generated high frequencies
coherent between channels. This also halves runtime versus processing L/R
independently, since only one channel goes through the diffusion model.

Expect roughly real-time-times-N runtime per track (N depends on
`--audiosr-ddim-steps`, default 25); this is still the slowest step in the
pipeline by a wide margin. `--audiosr-guidance-scale 1.0` disables
classifier-free guidance for roughly another 2x speedup (AudioSR's own DDIM
sampler runs a second unconditional forward pass per step whenever guidance
scale isn't exactly 1.0), at some cost to output relevance/quality. VRAM
peaks around 7.5-7.8 GB on an 8 GB card -- fine standalone, but leaves no
headroom to run alongside anything else GPU-heavy.

## audiobox-aesthetics (optional scoring signal for `--candidates`)

[facebookresearch/audiobox-aesthetics](https://github.com/facebookresearch/audiobox-aesthetics),
installed from PyPI (`audiobox-aesthetics`). Used by `score_track.py` when
`generate_track.py` is run with `--candidate-use-aesthetics`, to fold Meta's
CE/CU/PC/PQ (Content Enjoyment / Usefulness, Production Complexity / Quality)
scores into the best-of-N ranking alongside the local DSP metrics.

```sh
cd tools
uv venv --python 3.11 audiobox_venv
uv pip install --python audiobox_venv torch==2.7.1 torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu128
uv pip install --python audiobox_venv "torch==2.7.1+cu128" "torchaudio==2.7.1+cu128" \
    audiobox-aesthetics --index-url https://download.pytorch.org/whl/cu128 \
    --extra-index-url https://pypi.org/simple
uv pip install --python audiobox_venv requests soundfile
```

Unlike `audiosr`, this package's dependency pins are modern and loose (no
`transformers` dependency at all, `torch>=2.2.0`, no `numpy` ceiling) -- an
isolated venv is still used for consistency with the subprocess pattern above,
but conflict risk with the main project env is low. `requests` and
`soundfile` are undeclared runtime dependencies (missing from its own
`requirements.txt`, same story as `matplotlib` for `audiosr` above).

Checkpoint auto-downloads from `facebook/audiobox-aesthetics` on Hugging Face
on first run, no manual step needed. Driven by `tools/audiobox_infer.py`
(tracked in git), which scores a whole candidate batch in one model load.
