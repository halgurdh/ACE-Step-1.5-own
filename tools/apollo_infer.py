#!/usr/bin/env python3
"""Standalone CLI wrapper around the Apollo audio restoration model.

Adapted from the inference logic in the patriotyk/Apollo Hugging Face
Space (https://huggingface.co/spaces/patriotyk/Apollo), stripped of the
Gradio UI so it can run as a subprocess step in an external pipeline.

Expects the vendored clone (weights, configs, look2hear source) at
tools/apollo/ next to this script -- see tools/README.md for setup.
"""

import argparse
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import yaml
from ml_collections import ConfigDict

APOLLO_DIR = Path(__file__).resolve().parent / "apollo"
sys.path.insert(0, str(APOLLO_DIR))

import look2hear.models  # noqa: E402

CHECKPOINTS = {
    "restore": ("configs/apollo.yaml", "weights/apollo.bin"),
    "vocal": ("configs/apollo.yaml", "weights/apollo_vocal.bin"),
    "vocal2": ("configs/config_apollo_vocal.yaml", "weights/apollo_vocal2.bin"),
    "universal": ("configs/config_apollo_uni.yaml", "weights/apollo_model_uni.ckpt"),
}


def get_config(config_path: Path) -> ConfigDict:
    with open(config_path) as f:
        return ConfigDict(yaml.load(f, Loader=yaml.FullLoader))


def load_model(checkpoint: str, device: str):
    config_rel, weights_rel = CHECKPOINTS[checkpoint]
    config = get_config(APOLLO_DIR / config_rel)
    model = look2hear.models.BaseModel.from_pretrain(
        str(APOLLO_DIR / weights_rel), **config["model"]
    ).to(device)
    model.eval()
    return model


def _windowing_array(window_size: int, fade_size: int) -> torch.Tensor:
    """Linear crossfade window for overlap-add recombination.

    The upstream Space shipped this with constant ramps (fade-in of all ones,
    fade-out of all zeros), which zeroes each chunk's tail outright and splices
    chunks together with a hard edge -- an audible discontinuity wherever the
    model's output differs across the seam. Real 0->1 / 1->0 ramps make the
    overlap a weighted average (the existing ``result / counter`` division
    normalizes the summed weights), so seams blend smoothly.
    """
    fadein = torch.linspace(0, 1, fade_size)
    fadeout = torch.linspace(1, 0, fade_size)
    window = torch.ones(window_size)
    window[-fade_size:] *= fadeout
    window[:fade_size] *= fadein
    return window


def enhance(model, device: str, audio_path: Path, chunk_seconds: float = 10.0):
    test_data, samplerate = librosa.load(str(audio_path), mono=False, sr=44100)
    test_data = torch.from_numpy(test_data)
    if test_data.ndim == 1:
        test_data = test_data.unsqueeze(0)

    C = int(chunk_seconds * samplerate)
    N = 2
    step = C // N
    fade_size = 3 * samplerate
    border = C - step

    if test_data.shape[1] > 2 * border and border > 0:
        test_data = torch.nn.functional.pad(test_data, (border, border), mode="reflect")

    windowing_array = _windowing_array(C, fade_size)

    result = torch.zeros((1,) + tuple(test_data.shape), dtype=torch.float32)
    counter = torch.zeros((1,) + tuple(test_data.shape), dtype=torch.float32)

    total_seconds = test_data.shape[1] / samplerate
    total_windows = max(1, -(-test_data.shape[1] // step))

    i = 0
    window_index = 0
    while i < test_data.shape[1]:
        window_index += 1
        print(
            f"[Apollo] window {window_index}/{total_windows} "
            f"(position {i / samplerate:.1f}s/{total_seconds:.1f}s)",
            flush=True,
        )
        part = test_data[:, i : i + C]
        length = part.shape[-1]
        if length < C:
            if length > C // 2 + 1:
                part = torch.nn.functional.pad(part, (0, C - length), mode="reflect")
            else:
                part = torch.nn.functional.pad(part, (0, C - length, 0, 0), mode="constant", value=0)

        chunk = part.unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(chunk).squeeze(0).squeeze(0).cpu()

        window = windowing_array.clone()
        if i == 0:
            window[:fade_size] = 1
        elif i + C >= test_data.shape[1]:
            window[-fade_size:] = 1

        result[..., i : i + length] += out[..., :length] * window[..., :length]
        counter[..., i : i + length] += window[..., :length]

        i += step

    final_output = result / counter
    final_output = final_output.squeeze(0).numpy()
    np.nan_to_num(final_output, copy=False, nan=0.0)

    if test_data.shape[1] > 2 * border and border > 0:
        final_output = final_output[..., border:-border]

    return samplerate, final_output.T


def main() -> int:
    parser = argparse.ArgumentParser(description="Apollo audio restoration (de-smear).")
    parser.add_argument("--input", required=True, help="Input WAV path.")
    parser.add_argument("--output", required=True, help="Output WAV path.")
    parser.add_argument(
        "--checkpoint",
        default="restore",
        choices=list(CHECKPOINTS.keys()),
        help="Which Apollo checkpoint to use. 'restore' is the general MP3/codec de-smear model.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--chunk-seconds", type=float, default=10.0)
    args = parser.parse_args()

    model = load_model(args.checkpoint, args.device)
    samplerate, audio = enhance(model, args.device, Path(args.input), args.chunk_seconds)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), audio, samplerate, subtype="PCM_24")
    print(str(output_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())