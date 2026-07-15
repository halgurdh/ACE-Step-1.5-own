# Audio restoration / super-resolution tools

Optional post-processing steps for `generate_track.py`:
`generate (WAV) -> Apollo (de-smear/restore) -> AudioSR (bandwidth to ~24 kHz) -> mastering chain -> loudnorm -> MP3`.

Both tools run in their own isolated venvs (invoked as subprocesses from
`generate_track.py`) because their dependency pins conflict with the main
project environment (old `transformers`/`numpy` pins, older `torch`). Neither
is required for normal generation -- both are off by default
(`--enable-apollo-restoration`, `--enable-audiosr-upscale`).

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
downmixes each channel separately, processes in overlapping 5.12s windows,
and recombines with a linear crossfade, so a full-length stereo track goes in
and a full-length stereo track comes out. Expect roughly real-time-times-N
runtime per track (N depends on `--audiosr-ddim-steps`); this is the slowest
step in the pipeline by a wide margin. VRAM peaks around 7.5-7.8 GB on an
8 GB card at the default 50 DDIM steps -- fine standalone, but leaves no
headroom to run alongside anything else GPU-heavy.
