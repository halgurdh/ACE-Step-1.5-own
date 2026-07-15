#!/usr/bin/env python3
"""Standalone CLI wrapper around AudioSR for stereo, arbitrary-length music.

AudioSR's underlying model is mono-only (it reads channel 0 only) and was
trained on <=5.12s clips (its own code warns that longer inputs "may degrade
model performance"). This script works around both limits: it processes each
channel independently in overlapping 5.12s windows and recombines them with
a linear crossfade, so a full-length stereo track can be upsampled without
truncation or channel loss.
"""

import argparse
import gc
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from audiosr import build_model, super_resolution

TARGET_SR = 48000
CHUNK_SECONDS = 5.12


def _crossfade_window(window_size: int, fade_size: int) -> torch.Tensor:
    fadein = torch.linspace(0, 1, fade_size)
    fadeout = torch.linspace(1, 0, fade_size)
    window = torch.ones(window_size)
    window[:fade_size] = fadein
    window[-fade_size:] = fadeout
    return window


def process_channel(
    model,
    channel: np.ndarray,
    orig_sr: int,
    device: str,
    ddim_steps: int,
    guidance_scale: float,
    seed: int,
    overlap_seconds: float,
    tmp_dir: Path,
    channel_label: str = "",
) -> np.ndarray:
    chunk_samples_in = int(round(CHUNK_SECONDS * orig_sr))
    overlap_samples_in = int(round(overlap_seconds * orig_sr))
    step_samples_in = chunk_samples_in - overlap_samples_in

    chunk_samples_out = int(round(CHUNK_SECONDS * TARGET_SR))
    overlap_samples_out = int(round(overlap_seconds * TARGET_SR))

    total_in = len(channel)
    total_out = int(round(total_in * TARGET_SR / orig_sr))
    total_seconds = total_in / orig_sr
    total_windows = max(1, -(-total_in // step_samples_in))

    result = np.zeros(total_out, dtype=np.float64)
    counter = np.zeros(total_out, dtype=np.float64)

    temp_path = tmp_dir / "chunk.wav"

    position = 0
    window_index = 0
    while position < total_in:
        window_index += 1
        chunk_end = min(position + chunk_samples_in, total_in)
        chunk = channel[position:chunk_end]
        actual_chunk_seconds = len(chunk) / orig_sr

        print(
            f"[AudioSR] channel {channel_label} window {window_index}/{total_windows} "
            f"(position {position / orig_sr:.1f}s/{total_seconds:.1f}s)",
            flush=True,
        )
        sf.write(str(temp_path), chunk, orig_sr)
        waveform = super_resolution(
            model,
            str(temp_path),
            seed=seed,
            ddim_steps=ddim_steps,
            guidance_scale=guidance_scale,
        )
        upscaled = waveform[0, 0].astype(np.float64) if hasattr(waveform, "astype") else waveform[0, 0].numpy().astype(np.float64)
        del waveform
        gc.collect()
        torch.cuda.empty_cache()

        expected_out_len = int(round(actual_chunk_seconds * TARGET_SR))
        upscaled = upscaled[:expected_out_len]

        out_position = int(round(position * TARGET_SR / orig_sr))
        window = np.ones(len(upscaled))
        fade = min(overlap_samples_out, len(upscaled) // 2)
        if fade > 0:
            if position > 0:
                window[:fade] = np.linspace(0, 1, fade)
            if chunk_end < total_in:
                window[-fade:] = np.linspace(1, 0, fade)

        end_position = out_position + len(upscaled)
        result[out_position:end_position] += upscaled * window
        counter[out_position:end_position] += window

        if chunk_end >= total_in:
            break
        position += step_samples_in

    counter[counter == 0] = 1.0
    return (result / counter).astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description="AudioSR bandwidth extension for stereo music.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-name", default="basic", choices=["basic", "speech"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ddim-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overlap-seconds", type=float, default=0.64)
    args = parser.parse_args()

    data, orig_sr = sf.read(args.input, always_2d=True)
    channels = data.shape[1]
    channel_labels = {0: "L", 1: "R"}

    print(f"[AudioSR] loading model on {args.device}", flush=True)
    model = build_model(model_name=args.model_name, device=args.device)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        upscaled_channels = []
        for ch_index in range(channels):
            upscaled = process_channel(
                model,
                data[:, ch_index],
                orig_sr,
                args.device,
                args.ddim_steps,
                args.guidance_scale,
                args.seed,
                args.overlap_seconds,
                tmp_dir,
                channel_labels.get(ch_index, str(ch_index)),
            )
            upscaled_channels.append(upscaled)

    min_len = min(len(c) for c in upscaled_channels)
    stereo = np.stack([c[:min_len] for c in upscaled_channels], axis=1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), stereo, TARGET_SR, subtype="PCM_24")
    print(str(output_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
