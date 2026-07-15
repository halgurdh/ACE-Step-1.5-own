#!/usr/bin/env python3
"""Generate one or more ACE-Step tracks from a Python script.

This is a small command-line wrapper around the public inference API. It keeps
model loading explicit and leaves the Gradio/API servers out of the path.
"""

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from loguru import logger

from acestep.gpu_config import get_global_gpu_config, resolve_lm_backend
from acestep.handler import AceStepHandler
from acestep.inference import GenerationConfig, GenerationParams, create_sample, generate_music
from acestep.llm_inference import LLMHandler


PROJECT_ROOT = Path(__file__).resolve().parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
MIN_DURATION_SECONDS = 120.0
APPROX_DURATION_VARIANCE_SECONDS = 10.0
VRAM_RETRY_DURATION_STEP_SECONDS = 15.0
DEFAULT_TARGET_LUFS = -16.0
DEFAULT_TRUE_PEAK_DB = -1.5
DEFAULT_LOUDNESS_RANGE = 11.0
DEFAULT_STRUCTURE = "intro, chorus, verse, chorus, verse, outro"
DEFAULT_TIME_SIGNATURE = "4"
QUALITY_PRESETS = {
    "high": {
        "model": "acestep-v15-base",
        "steps": 64,
        "guidance_scale": 7.5,
    },
    "fast": {
        "model": "acestep-v15-turbo",
        "steps": 8,
        "guidance_scale": 1.0,
    },
}
MINOR_KEYS = [
    "A minor",
    "A# minor",
    "B minor",
    "C minor",
    "C# minor",
    "D minor",
    "D# minor",
    "E minor",
    "F minor",
    "F# minor",
    "G minor",
    "G# minor",
]
NATURAL_INSTRUMENT_GUIDANCE = (
    "Use natural, realistic, human-played instrument tones with expressive dynamics, "
    "realistic articulations, tasteful room ambience, and a polished live-studio feel. "
    "Use subtle human timing, velocity variation, realistic drum ghost notes, fills, "
    "strums, slides, bends, breath, finger noise, and performance dynamics where appropriate. "
    "Keep the groove locked and musical without sounding robotic. Avoid plastic, toy-like, "
    "harsh, thin, overly synthetic, or MIDI-demo sounding instruments."
)
SECTION_BLUEPRINT = [
    ("Intro", 0.12, "sparse setup, establish the core groove without full drums"),
    ("Chorus", 0.20, "main hook, fuller drums, strongest motif, higher energy"),
    ("Verse", 0.19, "reduced arrangement, new melodic movement, groove stays steady"),
    ("Chorus", 0.20, "return of the hook with extra layers, fills, and wider energy"),
    ("Verse", 0.19, "second variation, different instrument focus, controlled tension"),
    ("Outro", 0.10, "strip down the groove and resolve cleanly"),
]
TITLE_WORDS = {
    "afropop": ["sunrise", "lagos", "golden", "palm", "market", "joy", "highlife"],
    "chill": ["midnight", "soft", "drift", "haze", "quiet", "cloud", "afterglow"],
    "deep house": ["basement", "velvet", "night", "pulse", "subway", "afterhours", "shadow"],
    "electronic": ["neon", "signal", "circuit", "chrome", "future", "voltage", "motion"],
    "funk": ["pocket", "strut", "groove", "brass", "snap", "velvet", "downtown"],
    "hip hop": ["block", "cipher", "dust", "808", "corner", "sample", "midnight"],
    "house": ["warehouse", "piano", "floor", "jack", "sunset", "groove", "anthem"],
    "indian": ["monsoon", "tabla", "sitar", "rang", "river", "palace", "dusky"],
    "jazz": ["blue", "smoke", "late", "quartet", "brushed", "uptown", "velvet"],
    "pop": ["daylight", "spark", "radio", "summer", "heart", "city", "bright"],
    "r&b": ["silk", "velvet", "slow", "afterglow", "midnight", "rose", "warm"],
    "reggae": ["island", "dub", "sun", "roots", "tide", "skank", "irie"],
    "soul": ["gold", "church", "warm", "velvet", "heart", "brass", "sunday"],
    "spanish": ["luna", "playa", "rosa", "calle", "noche", "guitarra", "sol"],
}
TITLE_SUFFIXES = [
    "session",
    "mix",
    "sketch",
    "take",
    "groove",
    "motion",
    "suite",
    "cut",
]
DEFAULT_INSTRUMENTAL_LYRICS = "\n".join(
    [
        "[Intro]",
        "[Instrumental]",
        "[Chorus]",
        "[Instrumental]",
        "[Verse]",
        "[Instrumental]",
        "[Chorus]",
        "[Instrumental]",
        "[Verse]",
        "[Instrumental]",
        "[Outro]",
        "[Instrumental]",
    ]
)
GENRES = [
    "afropop",
    "chill",
    "deep house",
    "electronic",
    "funk",
    "hip hop",
    "house",
    "indian",
    "jazz",
    "pop",
    "r&b",
    "reggae",
    "soul",
    "spanish",
]
GENRE_PROFILES = {
    "afropop": {
        "caption": "afropop with bright guitar riffs, syncopated percussion, warm bass, call-and-response hooks, and an upbeat dance groove",
        "bpm": 105,
    },
    "chill": {
        "caption": "chill downtempo track with soft keys, mellow pads, relaxed drums, warm bass, and a calm late-night atmosphere",
        "bpm": 82,
    },
    "deep house": {
        "caption": "deep house with a steady four-on-the-floor groove, warm sub bass, muted chords, spacious pads, and hypnotic club energy",
        "bpm": 124,
    },
    "electronic": {
        "caption": "electronic track with polished synth leads, layered arpeggios, punchy drums, evolving textures, and a modern festival-ready build",
        "bpm": 128,
    },
    "funk": {
        "caption": "funk track with tight slap bass, crisp rhythm guitar, live drums, brass stabs, clavinet accents, and an infectious pocket groove",
        "bpm": 108,
    },
    "hip hop": {
        "caption": "hip hop beat with hard drums, deep 808 bass, chopped melodic samples, sparse keys, and a confident head-nod groove",
        "bpm": 92,
    },
    "house": {
        "caption": "house track with a driving four-on-the-floor kick, groovy bassline, piano stabs, vocal chops, and uplifting dancefloor energy",
        "bpm": 126,
    },
    "indian": {
        "caption": "Indian fusion track with tabla rhythms, sitar and flute melodies, cinematic strings, deep bass, and a modern polished arrangement",
        "bpm": 100,
    },
    "jazz": {
        "caption": "jazz track with brushed drums, walking upright bass, warm piano chords, saxophone melodies, and a smooth improvisational feel",
        "bpm": 116,
    },
    "pop": {
        "caption": "pop song with catchy melodic hooks, bright drums, polished synths, warm bass, and a radio-ready chorus-focused structure",
        "bpm": 112,
    },
    "r&b": {
        "caption": "r&b track with silky electric piano chords, smooth bass, crisp drums, lush harmonies, and a slow emotional groove",
        "bpm": 78,
    },
    "reggae": {
        "caption": "reggae track with offbeat guitar skanks, deep rounded bass, relaxed drums, organ bubbles, and a warm island groove",
        "bpm": 76,
    },
    "soul": {
        "caption": "soul track with warm vintage keys, expressive guitar, live drums, rich bass, brass accents, and heartfelt groove-driven energy",
        "bpm": 94,
    },
    "spanish": {
        "caption": "Spanish pop track with nylon guitar, hand percussion, warm bass, romantic melodic phrases, and a polished Latin groove",
        "bpm": 104,
    },
}


def parse_args() -> argparse.Namespace:
    """Parse command-line options for one-shot track generation."""
    parser = argparse.ArgumentParser(description="Generate a track with ACE-Step.")
    parser.add_argument(
        "--prompt",
        default="",
        help="Optional extra prompt detail. Leave empty to generate from genre only.",
    )
    parser.add_argument(
        "--genre",
        default=None,
        choices=GENRES,
        help="Genre to generate from.",
    )
    parser.add_argument(
        "--all-genres",
        action="store_true",
        help="Generate for every built-in genre. --amount applies per genre.",
    )
    parser.add_argument(
        "--lyrics",
        default=DEFAULT_INSTRUMENTAL_LYRICS,
        help="Deprecated; final generation always uses instrumental section tags.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=MIN_DURATION_SECONDS,
        help="Approximate target duration in seconds. Each track varies by +/-10s, minimum 120s.",
    )
    parser.add_argument("--bpm", type=int, default=None, help="Optional BPM.")
    parser.add_argument(
        "--key",
        default=None,
        choices=MINOR_KEYS,
        help="Minor key to use. If omitted, a minor key is selected per track.",
    )
    parser.add_argument(
        "--allow-smart-metadata",
        action="store_true",
        help="Allow smart prompt metadata to override genre BPM/time signature.",
    )
    parser.add_argument("--language", default="unknown", help='Vocal language, e.g. "en".')
    parser.add_argument("--seed", type=int, default=-1, help="-1 uses a random seed.")
    parser.add_argument("--amount", type=int, default=1, help="Number of tracks to generate.")
    parser.add_argument(
        "--concurrency",
        default="1",
        help='Tracks per generation call: "1", "2", or "auto". Use auto to follow VRAM tier.',
    )
    parser.add_argument(
        "--target-lufs",
        type=float,
        default=DEFAULT_TARGET_LUFS,
        help="Post-export integrated loudness target. Default: -16 LUFS.",
    )
    parser.add_argument(
        "--disable-lufs-normalization",
        action="store_true",
        help="Skip the post-export ffmpeg loudnorm pass.",
    )
    parser.add_argument(
        "--disable-vram-duration-retry",
        action="store_true",
        help="Fail immediately on VRAM errors instead of retrying shorter durations.",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Deprecated alias for --amount.")
    parser.add_argument(
        "--quality",
        default="high",
        choices=["high", "fast"],
        help="Quality preset. high uses base model; fast uses turbo.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Override diffusion steps. Defaults come from --quality.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=None,
        help="Override CFG guidance. Only base model uses this.",
    )
    parser.add_argument(
        "--sample-temperature",
        type=float,
        default=0.65,
        help="Creativity for the genre prompt planner. Higher gives more variation.",
    )
    parser.add_argument(
        "--no-smart-prompt",
        action="store_true",
        help="Skip LM prompt planning and use the built-in genre profile directly.",
    )
    parser.add_argument("--format", default="wav", choices=["wav", "flac", "mp3"], help="Output format.")
    parser.add_argument(
        "--mp3-bitrate",
        default="320k",
        choices=["128k", "192k", "256k", "320k"],
        help="MP3 bitrate when --format mp3. Default: 320k.",
    )
    parser.add_argument("--output-dir", default="output/py_generate", help="Directory for audio output.")
    parser.add_argument("--device", default="auto", help='Device: "auto", "cuda", "xpu", "mps", or "cpu".')
    parser.add_argument(
        "--model",
        default=None,
        help="Override DiT checkpoint folder name. Defaults come from --quality.",
    )
    parser.add_argument("--lm-model", default="acestep-5Hz-lm-0.6B", help="5Hz LM checkpoint folder name.")
    parser.add_argument(
        "--lm-backend",
        default=os.environ.get("ACESTEP_LM_BACKEND"),
        help='LM backend: "vllm", "pt", or "mlx". Defaults to GPU recommendation.',
    )
    parser.add_argument(
        "--offload",
        action="store_true",
        help="Force CPU offload. High quality enables offload automatically.",
    )
    parser.add_argument(
        "--no-offload",
        action="store_true",
        help="Disable automatic high-quality CPU offload.",
    )
    args = parser.parse_args()
    amount = args.batch_size if args.batch_size is not None else args.amount
    if amount < 1:
        parser.error("--amount must be 1 or greater")
    if not args.all_genres and not args.genre:
        parser.error("--genre is required unless --all-genres is set")
    if args.concurrency != "auto":
        try:
            concurrency = int(args.concurrency)
        except ValueError:
            parser.error('--concurrency must be "auto", "1", or "2"')
        if concurrency not in (1, 2):
            parser.error('--concurrency must be "auto", "1", or "2"')
    return args


def resolve_genres(args: argparse.Namespace) -> list[str]:
    """Return the genre list requested by the command line."""
    return list(GENRES) if args.all_genres else [args.genre]


def resolve_amount(args: argparse.Namespace) -> int:
    """Return the requested track count from ``--amount`` or legacy ``--batch-size``."""
    amount = args.batch_size if args.batch_size is not None else args.amount
    if amount < 1:
        raise ValueError("--amount must be 1 or greater")
    return amount


def resolve_model(args: argparse.Namespace) -> str:
    """Return the DiT model selected by explicit override or quality preset."""
    return args.model or QUALITY_PRESETS[args.quality]["model"]


def resolve_steps(args: argparse.Namespace) -> int:
    """Return diffusion steps selected by explicit override or quality preset."""
    return args.steps if args.steps is not None else int(QUALITY_PRESETS[args.quality]["steps"])


def resolve_guidance_scale(args: argparse.Namespace) -> float:
    """Return guidance scale selected by explicit override or quality preset."""
    if args.guidance_scale is not None:
        return args.guidance_scale
    return float(QUALITY_PRESETS[args.quality]["guidance_scale"])


def resolve_offload(args: argparse.Namespace) -> bool:
    """Return whether model offload should be used for this run."""
    if args.no_offload:
        return False
    return bool(args.offload or args.quality == "high")


def initialize_dit(args: argparse.Namespace) -> AceStepHandler:
    """Initialize and return the ACE-Step DiT handler."""
    handler = AceStepHandler()
    status, success = handler.initialize_service(
        project_root=str(PROJECT_ROOT),
        config_path=resolve_model(args),
        device=args.device,
        offload_to_cpu=resolve_offload(args),
    )
    if not success:
        raise RuntimeError(f"DiT initialization failed: {status}")
    logger.info(status)
    return handler


def initialize_lm(args: argparse.Namespace) -> LLMHandler:
    """Initialize and return the 5Hz language-model handler."""
    backend = resolve_lm_backend(args.lm_backend, get_global_gpu_config())
    handler = LLMHandler()
    status, success = handler.initialize(
        checkpoint_dir=str(CHECKPOINT_DIR),
        lm_model_path=args.lm_model,
        backend=backend,
        device=args.device,
        offload_to_cpu=resolve_offload(args),
        dtype=None,
    )
    if not success:
        raise RuntimeError(f"5Hz LM initialization failed: {status}")
    logger.info(status)
    return handler


def resolve_concurrency(args: argparse.Namespace) -> int:
    """Return the number of tracks to attempt per generation call."""
    if args.concurrency != "auto":
        return int(args.concurrency)

    gpu_config = get_global_gpu_config()
    max_batch = max(1, int(getattr(gpu_config, "max_batch_size_with_lm", 1)))
    concurrency = min(2, max_batch)
    logger.info(
        "Auto concurrency selected {} from GPU tier {}",
        concurrency,
        getattr(gpu_config, "tier", "unknown"),
    )
    return concurrency


def create_genre_prompt(
    llm_handler: LLMHandler,
    args: argparse.Namespace,
    genre: str,
    track_index: int,
    duration: float,
    keyscale: str,
) -> dict[str, object]:
    """Create a fresh genre-matched prompt with the existing 5Hz LM API."""
    if args.no_smart_prompt:
        return {}

    profile = GENRE_PROFILES[genre]
    amount = resolve_amount(args)
    section_plan = build_section_plan_text(duration)
    extra_detail = f" Extra user direction: {args.prompt.strip()}." if args.prompt.strip() else ""
    query = (
        f"Create one unique {genre} music generation idea for track "
        f"{track_index + 1} of {amount}. The track must be a complete "
        f"approximately {int(round(duration))}-second instrumental. "
        f"It must be fully instrumental with no sung or spoken vocals. It must follow this "
        f"exact section timeline: {section_plan}. It must strongly match "
        f"this genre profile: {profile['caption']}. Make it different from previous ideas, "
        f"with concrete instrumentation, arrangement, groove, mood, and production details. "
        f"Use exact minor key {keyscale}; do not use major key harmony or off-key notes. "
        f"{NATURAL_INSTRUMENT_GUIDANCE} "
        f"Keep the rhythm locked to {profile['bpm']} BPM in 4/4. Drums must be on-beat, "
        f"groovy, natural, and idiomatic for {genre}; avoid rushed, unstable, or off-grid drums. "
        f"Avoid simple looping by changing drums, bass, harmony, melody, fills, and energy "
        f"between sections while staying coherent."
        f"{extra_detail}"
    )
    sample = create_sample(
        llm_handler=llm_handler,
        query=query,
        instrumental=True,
        vocal_language="unknown",
        temperature=args.sample_temperature,
        top_p=0.92,
        use_constrained_decoding=True,
    )
    if not sample.success:
        logger.warning("Smart genre prompt failed; using built-in profile: {}", sample.status_message)
        return {}

    logger.info("Smart caption: {}", sample.caption)
    return sample.to_dict()


def coerce_optional_int(value: object) -> int | None:
    """Return an integer value when metadata can be converted safely."""
    if value in (None, "", "N/A"):
        return None
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return None


def resolve_track_duration(args: argparse.Namespace) -> float:
    """Return one approximate duration target while enforcing the minimum length."""
    base_duration = max(args.duration, MIN_DURATION_SECONDS)
    offset = random.uniform(-APPROX_DURATION_VARIANCE_SECONDS, APPROX_DURATION_VARIANCE_SECONDS)
    return max(MIN_DURATION_SECONDS, round(base_duration + offset, 1))


def resolve_minor_key(args: argparse.Namespace, genre: str, track_index: int, seed_offset: int) -> str:
    """Return an exact minor key for one track."""
    if args.key:
        return args.key
    return MINOR_KEYS[(seed_offset + track_index + len(genre)) % len(MINOR_KEYS)]


def format_seconds(seconds: float) -> str:
    """Format seconds as compact m:ss text for section guidance."""
    total_seconds = max(0, int(round(seconds)))
    minutes, remainder = divmod(total_seconds, 60)
    return f"{minutes}:{remainder:02d}"


def build_section_plan(duration: float) -> list[dict[str, str]]:
    """Build timed instrumental arrangement sections for the target duration."""
    start = 0.0
    sections = []
    for index, (name, ratio, description) in enumerate(SECTION_BLUEPRINT):
        end = duration if index == len(SECTION_BLUEPRINT) - 1 else start + (duration * ratio)
        sections.append(
            {
                "name": name,
                "start": format_seconds(start),
                "end": format_seconds(end),
                "description": description,
            }
        )
        start = end
    return sections


def build_section_plan_text(duration: float) -> str:
    """Return a compact human-readable section timeline."""
    return "; ".join(
        f"{section['name']} {section['start']}-{section['end']}: {section['description']}"
        for section in build_section_plan(duration)
    )


def build_instrumental_lyrics(duration: float) -> str:
    """Return instrumental section tags with timing and arrangement descriptions."""
    lines = []
    for section in build_section_plan(duration):
        lines.append(
            f"[{section['name']} - {section['start']} to {section['end']} - "
            f"{section['description']}]"
        )
        lines.append("[Instrumental]")
        lines.append("")
    return "\n".join(lines).strip()


def build_generation_params(
    args: argparse.Namespace,
    genre: str,
    sample: dict[str, object],
    duration: float,
    keyscale: str,
) -> GenerationParams:
    """Build generation parameters from parsed command-line options."""
    profile = GENRE_PROFILES[genre]
    caption = str(sample.get("caption") or profile["caption"])
    if args.prompt.strip() and not sample:
        caption = f"{caption}, {args.prompt.strip()}"

    if args.duration < MIN_DURATION_SECONDS:
        logger.warning(
            "Duration base raised from {}s to minimum {}s",
            args.duration,
            MIN_DURATION_SECONDS,
        )
    smart_bpm = coerce_optional_int(sample.get("bpm")) if args.allow_smart_metadata else None
    bpm = args.bpm if args.bpm is not None else smart_bpm
    locked_bpm = bpm if bpm is not None else profile["bpm"]
    time_signature = (
        str(sample.get("timesignature") or DEFAULT_TIME_SIGNATURE)
        if args.allow_smart_metadata
        else DEFAULT_TIME_SIGNATURE
    )
    caption = (
        f"{genre} genre lock: {profile['caption']}. {caption}. "
        f"Fully instrumental with no sung or spoken vocals. "
        f"Must follow this exact structure and timing: {build_section_plan_text(duration)}. "
        f"Fixed tempo {locked_bpm} BPM, {time_signature}/4 time. Exact key: {keyscale}. "
        "Use only notes and harmonies that fit this minor key; avoid off-key melodies, "
        "wrong-key bass notes, or accidental major-key modulation. "
        f"{NATURAL_INSTRUMENT_GUIDANCE} "
        f"Drums must be on-beat, groovy, natural, and stylistically correct for {genre}. "
        "Use clear section changes, tasteful fills, realistic transitions, and evolving "
        "arrangement to avoid repetitive looping. Prefer a finished record feel over a loop."
    )

    return GenerationParams(
        task_type="text2music",
        caption=caption,
        lyrics=build_instrumental_lyrics(duration),
        instrumental=True,
        vocal_language="unknown",
        bpm=locked_bpm,
        keyscale=keyscale,
        timesignature=time_signature,
        duration=duration,
        inference_steps=resolve_steps(args),
        guidance_scale=resolve_guidance_scale(args),
        seed=args.seed,
        thinking=True,
        lm_temperature=0.7,
        lm_top_p=0.85,
        use_cot_metas=False,
        use_cot_caption=False,
        use_cot_language=False,
    )


def build_generation_config(args: argparse.Namespace) -> GenerationConfig:
    """Build batch generation settings from command-line options."""
    return build_generation_config_for_track(args, track_index=0)


def build_generation_config_for_track(
    args: argparse.Namespace,
    track_index: int,
    chunk_size: int = 1,
    seed_offset: int = 0,
) -> GenerationConfig:
    """Build generation settings for one sequential export."""
    seeds = None
    if args.seed >= 0:
        seeds = [args.seed + seed_offset + track_index + index for index in range(chunk_size)]

    return GenerationConfig(
        batch_size=chunk_size,
        use_random_seed=args.seed < 0,
        seeds=seeds,
        audio_format=args.format,
        mp3_bitrate=args.mp3_bitrate,
    )


def normalize_file_to_lufs(path: str, target_lufs: float) -> bool:
    """Normalize one exported file in-place to an approximate LUFS target."""
    if not path:
        return False
    source_path = Path(path)
    if not source_path.exists():
        logger.warning("LUFS normalization skipped; file not found: {}", source_path)
        return False
    if shutil.which("ffmpeg") is None:
        logger.warning("LUFS normalization skipped; ffmpeg is not available on PATH")
        return False

    temp_path = source_path.with_name(f"{source_path.stem}.lufs_tmp{source_path.suffix}")
    loudnorm = (
        f"loudnorm=I={target_lufs}:"
        f"TP={DEFAULT_TRUE_PEAK_DB}:"
        f"LRA={DEFAULT_LOUDNESS_RANGE}"
    )
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source_path),
        "-af",
        loudnorm,
        str(temp_path),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, timeout=180)
        temp_path.replace(source_path)
        logger.info("Normalized to about {} LUFS: {}", target_lufs, source_path)
        return True
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="ignore") if exc.stderr else str(exc)
        logger.warning("LUFS normalization failed for {}: {}", source_path, stderr)
        return False
    except subprocess.TimeoutExpired:
        logger.warning("LUFS normalization timed out for {}", source_path)
        return False
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                logger.warning("Could not remove temporary LUFS file: {}", temp_path)


def slugify_title(value: str) -> str:
    """Convert a generated title to a filesystem-friendly lowercase slug."""
    allowed = []
    previous_dash = False
    for character in value.lower():
        if character.isalnum():
            allowed.append(character)
            previous_dash = False
        elif not previous_dash:
            allowed.append("-")
            previous_dash = True
    return "".join(allowed).strip("-") or "track"


def generate_track_title(genre: str, track_number: int) -> str:
    """Generate a short human-readable genre-specific track title."""
    _ = track_number
    words = TITLE_WORDS.get(genre, ["music", "track", "session"])
    first = random.choice(words)
    second = random.choice([word for word in words if word != first] or words)
    suffix = random.choice(TITLE_SUFFIXES)
    return f"{genre} {first} {second} {suffix}"


def rename_audio_file(path: str, genre: str, track_number: int) -> str:
    """Rename an exported hash file to a human-readable generated title."""
    if not path:
        return path
    source_path = Path(path)
    if not source_path.exists():
        logger.warning("Rename skipped; file not found: {}", source_path)
        return path

    base_slug = slugify_title(generate_track_title(genre, track_number))
    target_path = source_path.with_name(f"{base_slug}{source_path.suffix}")
    words = TITLE_WORDS.get(genre, ["music", "track", "session"])
    suffix_index = 0
    while target_path.exists() and target_path != source_path:
        suffix_word = words[suffix_index % len(words)]
        suffix_index += 1
        target_path = source_path.with_name(f"{base_slug}-{suffix_word}{source_path.suffix}")

    if target_path == source_path:
        return str(source_path)
    source_path.replace(target_path)
    logger.info("Renamed export: {}", target_path)
    return str(target_path)


def is_vram_error(result) -> bool:
    """Return whether a generation result failed due to insufficient VRAM."""
    text = " ".join(
        str(value)
        for value in (
            getattr(result, "error", ""),
            getattr(result, "status_message", ""),
        )
    )
    return "Insufficient free VRAM" in text


def next_retry_duration(duration: float) -> float | None:
    """Return the next lower duration for a VRAM retry, or ``None`` if at minimum."""
    if duration <= MIN_DURATION_SECONDS:
        return None
    return max(MIN_DURATION_SECONDS, round(duration - VRAM_RETRY_DURATION_STEP_SECONDS, 1))


def generate_track_chunk(
    dit_handler: AceStepHandler,
    llm_handler: LLMHandler,
    args: argparse.Namespace,
    genre: str,
    track_index: int,
    chunk_size: int,
    initial_duration: float,
    keyscale: str,
    save_dir: Path,
    seed_offset: int,
):
    """Generate one chunk, retrying with shorter durations on VRAM pressure."""
    duration = initial_duration
    while True:
        sample = create_genre_prompt(llm_handler, args, genre, track_index, duration, keyscale)
        result = generate_music(
            dit_handler=dit_handler,
            llm_handler=llm_handler,
            params=build_generation_params(args, genre, sample, duration, keyscale),
            config=build_generation_config_for_track(args, track_index, chunk_size, seed_offset),
            save_dir=str(save_dir),
        )
        if result.success:
            return result, duration

        if args.disable_vram_duration_retry or not is_vram_error(result):
            return result, duration

        retry_duration = next_retry_duration(duration)
        if retry_duration is None:
            return result, duration

        logger.warning(
            "VRAM preflight failed at {}s; retrying track {} at {}s",
            duration,
            track_index + 1,
            retry_duration,
        )
        duration = retry_duration


def main() -> int:
    """Generate tracks and print resulting audio paths."""
    args = parse_args()
    amount = resolve_amount(args)
    genres = resolve_genres(args)
    concurrency = resolve_concurrency(args)
    initial_concurrency = concurrency
    save_dir = PROJECT_ROOT / args.output_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: {}", save_dir)
    logger.info("Genres: {}", ", ".join(genres))
    logger.info("Requested tracks per genre: {}", amount)
    logger.info("Tracks per generation call: {}", concurrency)
    logger.info(
        "Quality: {} model={} steps={} guidance={} offload={}",
        args.quality,
        resolve_model(args),
        resolve_steps(args),
        resolve_guidance_scale(args),
        resolve_offload(args),
    )

    dit_handler = initialize_dit(args)
    llm_handler = initialize_lm(args)
    audios = []
    file_entries = []
    for genre_index, genre in enumerate(genres):
        logger.info("Starting genre: {}", genre)
        seed_offset = genre_index * amount
        track_index = 0
        while track_index < amount:
            chunk_size = min(concurrency, amount - track_index)
            logger.info(
                "Generating {} tracks {}-{} of {}",
                genre,
                track_index + 1,
                track_index + chunk_size,
                amount,
            )
            duration = resolve_track_duration(args)
            keyscale = resolve_minor_key(args, genre, track_index, seed_offset)
            logger.info("Chunk target duration: {}s", duration)
            logger.info("Chunk key: {}", keyscale)
            result, used_duration = generate_track_chunk(
                dit_handler=dit_handler,
                llm_handler=llm_handler,
                args=args,
                genre=genre,
                track_index=track_index,
                chunk_size=chunk_size,
                initial_duration=duration,
                keyscale=keyscale,
                save_dir=save_dir,
                seed_offset=seed_offset,
            )

            if not result.success:
                logger.error(result.status_message)
                return 1
            audios.extend(result.audios)
            saved_count = len(result.audios)
            for offset, audio in enumerate(result.audios):
                audio_path = audio.get("path", "")
                lufs_normalized = False
                if not args.disable_lufs_normalization:
                    lufs_normalized = normalize_file_to_lufs(audio_path, args.target_lufs)
                track_number = track_index + offset + 1
                renamed_path = rename_audio_file(audio_path, genre, track_number)
                audio["path"] = renamed_path
                file_entries.append(
                    {
                        "path": renamed_path,
                        "original_path": audio_path,
                        "genre": genre,
                        "target_duration": used_duration,
                        "key": keyscale,
                        "section_plan": build_section_plan(used_duration),
                        "seed": audio.get("params", {}).get("seed"),
                        "track_number": track_number,
                        "target_lufs": None if args.disable_lufs_normalization else args.target_lufs,
                        "lufs_normalized": lufs_normalized,
                    }
                )
            if saved_count < 1:
                logger.error("Generation returned no audio files for requested chunk")
                return 1
            if saved_count < chunk_size:
                logger.warning(
                    "Requested {} tracks in one call but only {} were saved; continuing sequentially.",
                    chunk_size,
                    saved_count,
                )
                concurrency = 1
            track_index += saved_count
            logger.info("Saved {}/{} tracks for {}", track_index, amount, genre)

    manifest_path = save_dir / f"manifest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    manifest = {
        "requested_tracks_per_genre": amount,
        "saved_tracks": len(audios),
        "genres": genres,
        "base_duration": max(args.duration, MIN_DURATION_SECONDS),
        "duration_variance_seconds": APPROX_DURATION_VARIANCE_SECONDS,
        "vram_retry_duration_step_seconds": VRAM_RETRY_DURATION_STEP_SECONDS,
        "allowed_keys": MINOR_KEYS,
        "requested_concurrency": initial_concurrency,
        "final_concurrency": concurrency,
        "target_lufs": None if args.disable_lufs_normalization else args.target_lufs,
        "quality": args.quality,
        "model": resolve_model(args),
        "steps": resolve_steps(args),
        "guidance_scale": resolve_guidance_scale(args),
        "offload": resolve_offload(args),
        "files": file_entries,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    logger.info("Saved tracks: {}/{}", len(audios), amount)
    logger.info("Manifest: {}", manifest_path)
    for audio in audios:
        print(audio.get("path", "(in-memory)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
