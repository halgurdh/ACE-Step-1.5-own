#!/usr/bin/env python3
"""Score generated tracks for objective audio quality (best-of-N selection).

Mastering (EQ, Apollo, AudioSR) can only polish whatever the diffusion model
produced -- it cannot fix smeared transients, mushy mixes, or a wandering
arrangement baked into the raw generation. Seed-to-seed variance on these
models is large enough that generating several candidates per track and
keeping only the best-scoring one typically moves perceived quality more
than the whole post-processing chain combined.

This module combines cheap local DSP metrics with an optional Meta
audiobox-aesthetics model score:
  - HF spectral flatness  ("fizz" detector -- noise-like high end)
  - Transient crest factor in a percussive band ("smear" detector)
  - Bandwidth ceiling (how far real energy extends before it drops off)
  - Stereo correlation (catches phase-cancelling / decorrelation issues)
  - Spectral-balance distance to a reference track's tonal profile (optional)
  - audiobox-aesthetics CE/CU/PC/PQ scores (optional, needs tools/audiobox_venv)

The combined score's weights are heuristic starting points, not calibrated
against human judgment -- see WEIGHTS below to tune them. Every raw metric
is always returned alongside the composite score so results stay inspectable.
"""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

PROJECT_ROOT = Path(__file__).resolve().parent
AUDIOBOX_DIR = PROJECT_ROOT / "tools" / "audiobox_venv"
AUDIOBOX_SCRIPT = PROJECT_ROOT / "tools" / "audiobox_infer.py"

HF_FIZZ_BAND = (10000, 20000)
TRANSIENT_BAND = (2000, 8000)
TRANSIENT_FRAME_MS = 20
BANDWIDTH_THRESHOLD_DB = -60.0
OCTAVE_BAND_EDGES = (20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 10240, 20000)

# Heuristic composite-score weights. Positive = higher metric value is better.
WEIGHTS = {
    "hf_fizz": -2.0,  # 0..1, higher = more noise-like HF ("fizz") = worse
    "transient_crest_db": 0.05,  # ~20-40 dB range, higher = punchier/less smeared
    "bandwidth_ceiling_hz": 0.00002,  # ~0-24000 Hz range, higher = fuller extension
    "stereo_phase_penalty": -3.0,  # max(0, -correlation), penalizes phase cancellation
    "spectral_balance_distance_db": -0.1,  # only scored when --reference is given
    "aesthetics_mean": 0.3,  # mean(CE, CU, PQ), only scored with --use-aesthetics
}


def _venv_python(venv_dir: Path) -> Path:
    """Return an isolated venv's python executable, Windows or POSIX layout."""
    windows_path = venv_dir / "Scripts" / "python.exe"
    if windows_path.exists():
        return windows_path
    return venv_dir / "bin" / "python"


def _to_mono(audio: np.ndarray) -> np.ndarray:
    return audio.mean(axis=1) if audio.ndim > 1 and audio.shape[1] > 1 else audio.reshape(-1)


def spectral_flatness_band(mono: np.ndarray, sr: int, band=HF_FIZZ_BAND, n_fft: int = 4096) -> float:
    """Geometric/arithmetic mean ratio of PSD in a high band. ~1.0 = noise-like (fizzy)."""
    nyquist = sr / 2 - 1
    if band[0] >= nyquist:
        return 0.0
    freqs, psd = signal.welch(mono, sr, nperseg=min(n_fft, len(mono)))
    band_mask = (freqs >= band[0]) & (freqs <= min(band[1], nyquist))
    if not np.any(band_mask):
        return 0.0
    band_psd = np.maximum(psd[band_mask], 1e-12)
    geo_mean = np.exp(np.mean(np.log(band_psd)))
    arith_mean = np.mean(band_psd)
    return float(geo_mean / arith_mean)


def transient_crest_db(mono: np.ndarray, sr: int, band=TRANSIENT_BAND, frame_ms: float = TRANSIENT_FRAME_MS) -> float:
    """90th-percentile peak/RMS crest factor (dB) in a percussive band. Higher = punchier."""
    nyquist = sr / 2 - 1
    hi = min(band[1], nyquist)
    if band[0] >= hi:
        return 0.0
    sos = signal.butter(4, [band[0], hi], btype="band", fs=sr, output="sos")
    filtered = signal.sosfiltfilt(sos, mono)

    frame_len = max(1, int(sr * frame_ms / 1000))
    n_frames = len(filtered) // frame_len
    if n_frames < 1:
        return 0.0
    frames = filtered[: n_frames * frame_len].reshape(n_frames, frame_len)
    peak = np.max(np.abs(frames), axis=1)
    rms = np.sqrt(np.mean(frames**2, axis=1)) + 1e-9
    crest_db = 20 * np.log10(np.maximum(peak / rms, 1e-9))
    return float(np.percentile(crest_db, 90))


def bandwidth_ceiling_hz(mono: np.ndarray, sr: int, threshold_db: float = BANDWIDTH_THRESHOLD_DB, n_fft: int = 8192) -> float:
    """Highest frequency where PSD stays within threshold_db of the peak."""
    freqs, psd = signal.welch(mono, sr, nperseg=min(n_fft, len(mono)))
    psd_db = 10 * np.log10(np.maximum(psd, 1e-15))
    above = psd_db >= (np.max(psd_db) + threshold_db)
    return float(freqs[above][-1]) if np.any(above) else 0.0


def stereo_correlation(audio: np.ndarray) -> float:
    """Pearson correlation between L/R. Near -1 = phase-cancelling, near 1 = mono-like."""
    if audio.ndim < 2 or audio.shape[1] < 2:
        return 1.0
    left, right = audio[:, 0], audio[:, 1]
    if np.std(left) < 1e-9 or np.std(right) < 1e-9:
        return 1.0
    return float(np.corrcoef(left, right)[0, 1])


def band_energy_profile_db(mono: np.ndarray, sr: int, band_edges=OCTAVE_BAND_EDGES, n_fft: int = 8192) -> np.ndarray:
    """Per-band average PSD (dB, level-normalized) -- a coarse tonal-balance fingerprint."""
    freqs, psd = signal.welch(mono, sr, nperseg=min(n_fft, len(mono)))
    profile = []
    for lo, hi in zip(band_edges[:-1], band_edges[1:]):
        mask = (freqs >= lo) & (freqs < min(hi, sr / 2))
        energy = np.mean(psd[mask]) if np.any(mask) else 1e-15
        profile.append(10 * np.log10(max(energy, 1e-15)))
    profile_arr = np.array(profile)
    return profile_arr - np.mean(profile_arr)


def spectral_balance_distance(candidate_profile: np.ndarray, reference_profile: np.ndarray) -> float:
    """Mean absolute dB difference between two band-energy profiles. Lower = closer match."""
    return float(np.mean(np.abs(candidate_profile - reference_profile)))


def compute_dsp_metrics(path: Path, reference_profile: np.ndarray | None = None) -> dict[str, float]:
    """Compute all local DSP metrics for one audio file."""
    audio, sr = sf.read(str(path), always_2d=True)
    mono = _to_mono(audio)

    metrics = {
        "hf_fizz": spectral_flatness_band(mono, sr),
        "transient_crest_db": transient_crest_db(mono, sr),
        "bandwidth_ceiling_hz": bandwidth_ceiling_hz(mono, sr),
        "stereo_phase_penalty": max(0.0, -stereo_correlation(audio)),
    }
    if reference_profile is not None:
        candidate_profile = band_energy_profile_db(mono, sr)
        metrics["spectral_balance_distance_db"] = spectral_balance_distance(candidate_profile, reference_profile)
    return metrics


def compute_reference_profile(path: Path) -> np.ndarray:
    """Build a tonal-balance fingerprint from a reference (e.g. a mastered) track."""
    audio, sr = sf.read(str(path), always_2d=True)
    return band_energy_profile_db(_to_mono(audio), sr)


def run_audiobox_aesthetics(paths: list[Path]) -> dict[str, dict[str, float]] | None:
    """Score a batch of files with Meta's audiobox-aesthetics model in one subprocess call."""
    python_exe = _venv_python(AUDIOBOX_DIR)
    if not python_exe.exists() or not AUDIOBOX_SCRIPT.exists():
        return None

    command = [str(python_exe), str(AUDIOBOX_SCRIPT)]
    for path in paths:
        command += ["--input", str(path)]
    try:
        proc = subprocess.run(command, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        return None
    if proc.returncode != 0:
        return None
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return None


def composite_score(metrics: dict[str, float]) -> float:
    """Combine metrics into one scalar using WEIGHTS. Higher is better."""
    return sum(WEIGHTS[key] * value for key, value in metrics.items() if key in WEIGHTS)


def score_file(
    path: Path,
    reference_profile: np.ndarray | None = None,
    aesthetics: dict[str, float] | None = None,
) -> dict[str, float]:
    """Return {metric_name: value, ..., "score": composite} for one file."""
    metrics = compute_dsp_metrics(path, reference_profile)
    if aesthetics:
        metrics["aesthetics_mean"] = (
            aesthetics.get("CE", 0.0) + aesthetics.get("CU", 0.0) + aesthetics.get("PQ", 0.0)
        ) / 3.0
        metrics.update({f"aesthetics_{k}": v for k, v in aesthetics.items()})
    metrics["score"] = composite_score(metrics)
    return metrics


def score_files(
    paths: list[Path],
    reference: Path | None = None,
    use_aesthetics: bool = False,
) -> dict[str, dict[str, float]]:
    """Score multiple candidate files and return {path_str: metrics_dict}."""
    reference_profile = compute_reference_profile(reference) if reference else None
    aesthetics_by_path = run_audiobox_aesthetics(paths) if use_aesthetics else None
    if use_aesthetics and aesthetics_by_path is None:
        print("[score_track] audiobox-aesthetics unavailable; scoring without it.", flush=True)

    results = {}
    for path in paths:
        aesthetics = aesthetics_by_path.get(str(path)) if aesthetics_by_path else None
        results[str(path)] = score_file(path, reference_profile, aesthetics)
    return results


def pick_best(paths: list[Path], reference: Path | None = None, use_aesthetics: bool = False) -> tuple[Path, dict]:
    """Score all candidates and return (best_path, best_metrics)."""
    results = score_files(paths, reference, use_aesthetics)
    best_str = max(results, key=lambda key: results[key]["score"])
    return Path(best_str), results[best_str]


def main() -> int:
    parser = argparse.ArgumentParser(description="Score WAV files for objective audio quality.")
    parser.add_argument("--input", action="append", required=True, help="WAV path(s) to score.")
    parser.add_argument("--reference", default=None, help="Reference track for tonal-balance distance.")
    parser.add_argument("--use-aesthetics", action="store_true", help="Also score with audiobox-aesthetics.")
    args = parser.parse_args()

    paths = [Path(p) for p in args.input]
    reference = Path(args.reference) if args.reference else None
    results = score_files(paths, reference, args.use_aesthetics)

    for path_str, metrics in sorted(results.items(), key=lambda item: -item[1]["score"]):
        print(f"{path_str}: score={metrics['score']:.4f}")
        for key, value in metrics.items():
            if key != "score":
                print(f"    {key}: {value:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
