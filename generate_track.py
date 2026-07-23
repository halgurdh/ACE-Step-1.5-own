#!/usr/bin/env python3
"""Generate one or more ACE-Step tracks from a Python script.

This is a small command-line wrapper around the public inference API. It keeps
model loading explicit and leaves the Gradio/API servers out of the path.
"""

import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path

from loguru import logger

import score_track
from acestep.gpu_config import get_global_gpu_config, is_mps_platform, resolve_lm_backend
from acestep.handler import AceStepHandler
from acestep.inference import GenerationConfig, GenerationParams, create_sample, generate_music
from acestep.llm_inference import LLMHandler


PROJECT_ROOT = Path(__file__).resolve().parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
TOOLS_DIR = PROJECT_ROOT / "tools"
APOLLO_DIR = TOOLS_DIR / "apollo"
APOLLO_SCRIPT = TOOLS_DIR / "apollo_infer.py"
AUDIOSR_DIR = TOOLS_DIR / "audiosr_venv"
AUDIOSR_SCRIPT = TOOLS_DIR / "audiosr_venv_infer.py"
MIN_DURATION_SECONDS = 120.0
APPROX_DURATION_VARIANCE_SECONDS = 10.0
VRAM_RETRY_DURATION_STEP_SECONDS = 15.0
DEFAULT_TARGET_LUFS = -14.0  # streaming delivery standard (Spotify/YouTube/Amazon reference level)
DEFAULT_TRUE_PEAK_DB = -1.0  # streaming-safe true-peak ceiling (headroom for lossy re-encoding)
DEFAULT_LOUDNESS_RANGE = 11.0
DEFAULT_FADE_OUT_SECONDS = 4.0
# Some generated tracks end with real content, then a silence gap, then a burst of
# engine-like/noisy artifact instead of a clean stop -- that artifact always follows a
# silence gap, so scanning the last TAIL_SILENCE_WINDOW_SECONDS for where the audio goes
# quiet gives a safe trim point that removes both the gap and whatever garbage follows.
# Threshold/min-duration are deliberately strict (well below normal mix level, and long
# enough) so a genuinely quiet musical passage near the end isn't mistaken for the gap.
TAIL_SILENCE_WINDOW_SECONDS = 12.0
TAIL_SILENCE_NOISE_DB = -35.0
TAIL_SILENCE_MIN_SECONDS = 0.3
_SILENCE_START_RE = re.compile(r"silence_start:\s*([0-9.]+)")
DEFAULT_TONE_PROFILE = "reference"
DEFAULT_STEREO_WIDTH = 0.7  # ffmpeg stereotools slev: 1.0 = unchanged, <1.0 narrows the
# side (difference) signal, >1.0 widens. Generated tracks have come out diffuse/unfocused
# ("all over the place") -- genre-profile captions and WARM_MIX_GUIDANCE both explicitly ask
# for "wide stereo image", which the model leans into more than intended. This is a direct,
# measurable narrowing rather than another soft caption request.
# When the gain needed to hit the LUFS target would push true peak past the ceiling
# minus this margin, the final stage switches from pure gain to gain + oversampled limiting.
LOUDNESS_LIMITER_MARGIN_DB = 0.2
MONO_BASS_CROSSOVER_HZ = 200  # below this the mix is summed to mono (LR4 split); a kick's
# "knock"/punch body commonly sits 150-300 Hz, above the sub-only 120 Hz the reference
# master itself measured at -- raised so kick/bass read as centered by default, not just
# via --mono-bass-crossover-hz (confirmed empirically: 120 Hz left a 200 Hz wide-stereo
# test signal at 0.46 L/R correlation after mastering; 250 Hz brought it to 0.92).
_INTERMEDIATE_CODEC_ARGS = ["-c:a", "pcm_s24le", "-ar", "48000"]
DEFAULT_STRUCTURE = "intro, chorus, verse, chorus, verse, outro"
DEFAULT_TIME_SIGNATURE = "4"
QUALITY_PRESETS = {
    "best": {
        "model": "acestep-v15-sft",
        "steps": 50,
        "guidance_scale": 4.0,
    },
    "extreme": {
        "model": "acestep-v15-base",
        "steps": 150,
        "guidance_scale": 4.0,
    },
    "ultra": {
        "model": "acestep-v15-base",
        "steps": 100,
        "guidance_scale": 4.0,
    },
    "high": {
        "model": "acestep-v15-base",
        "steps": 64,
        "guidance_scale": 4.0,
    },
    "balanced": {
        "model": "acestep-v15-base",
        "steps": 32,
        "guidance_scale": 4.0,
    },
    "fast": {
        "model": "acestep-v15-turbo",
        "steps": 8,
        "guidance_scale": 2.0,
    },
}
MINOR_KEYS = [
    "A minor",
    "Bb minor",
    "B minor",
    "C minor",
    "C# minor",
    "D minor",
    "Eb minor",
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
PRODUCED_INSTRUMENT_GUIDANCE = (
    "Use polished, well-programmed electronic production rather than a live-band feel: tight "
    "drum-machine and sampled drum hits, punchy synth basses and leads, crisp sequenced "
    "arpeggios, and modern studio-quality sound design. Timing should be tight and "
    "grid-locked, with only subtle stylistic swing or groove quantization where the genre "
    "calls for it, not loose live-band timing. Do not add acoustic live-drum-kit character, "
    "ghost notes, finger noise, or acoustic-band performance nuance; the drums and "
    "instruments should sound deliberately produced and electronic, not like a live "
    "recording. Avoid harsh, thin, cheap, or amateur-sounding synths; aim for a "
    "professional, radio-ready electronic mix."
)
ELECTRONIC_GENRES = {"deep house", "drum & bass", "electronic", "hip hop", "house", "reggaeton"}


def resolve_instrument_guidance(genre: str) -> str:
    """Return live-instrument or produced-electronic guidance based on genre idiom."""
    return PRODUCED_INSTRUMENT_GUIDANCE if genre in ELECTRONIC_GENRES else NATURAL_INSTRUMENT_GUIDANCE


def minor_key_guidance(keyscale: str) -> str:
    """Return a strict natural-minor instruction anchored to the locked key.

    The dedicated ``keyscale`` conditioning field alone is not reliably enough to
    keep the model in minor -- it commonly drifts to a relative/parallel major
    resolution or a Picardy third at cadences and outros. Used in the natural-
    language LM query only (not the final DiT caption -- that text encoder never
    saw key described in prose during training, so restating it there would be
    out-of-distribution; the dedicated keyscale field already covers that path).
    """
    return (
        f"Stay in natural minor the entire track, in {keyscale}: every section, chord "
        f"progression, cadence, and the ending must resolve within {keyscale} minor. "
        f"Do not modulate or resolve to the relative or parallel major, do not add a "
        f"Picardy third or major-tonic ending, and avoid bright, happy, or major-mode "
        f"chord borrowings anywhere in the harmony or melody."
    )


def resolve_mix_guidance(args: argparse.Namespace) -> str:
    """Pick the prompt-side mix direction that matches the mastering tone profile.

    The 'reference' profile targets the reference track's signature: sub-forward,
    warm, smooth/dark top. 'neutral' and 'bright' keep the original clarity-first
    direction so generation and mastering always pull the same way.
    """
    profile = getattr(args, "tone_profile", DEFAULT_TONE_PROFILE)
    return WARM_MIX_GUIDANCE if profile == "reference" else MIX_CLARITY_GUIDANCE
MIX_CLARITY_GUIDANCE = (
    "Use a clean full-bandwidth mix with natural extended air, open cymbal detail, and defined "
    "transients. Do not brickwall, lowpass, band-limit, dull, or roll off the high-frequency "
    "content; keep real musical energy and texture above 15 kHz when the instruments naturally "
    "produce it. Hi-hats, rides, crashes, shakers, tambourines, and cymbal tails should sound "
    "crisp, detailed, and realistic without becoming piercing, brittle, fizzy, metallic, or "
    "dominant. Avoid smeared, phasey, watery, low-bitrate, over-compressed, muffled, or harsh "
    "high-end artifacts. When the drum or percussion pattern is busy, syncopated, or uses many "
    "hand-percussion layers (congas, shakers, bells, talking drum, and similar), keep the "
    "rhythmic complexity but render every hit as a clean, distinct, well-separated transient; "
    "busy and syncopated must never mean blurred, grainy, hissy, or noisy. Complexity belongs "
    "in the rhythm and pattern, never in distortion or noise in the drum sound itself."
)
WARM_MIX_GUIDANCE = (
    "Use a warm, full-bandwidth professional mix in the style of a modern commercial record: "
    "a deep, powerful, well-controlled sub and low end that anchors the track, rich warm "
    "low-mids, and a smooth, silky, slightly dark top end. High frequencies should sound "
    "natural, soft, and expensive -- airy but never bright, crisp, fizzy, brittle, piercing, "
    "or hyped. Cymbals, hi-hats, and percussion must be smooth and detailed with clean, "
    "distinct transients, never harsh, grainy, noisy, or smeared. Keep the bass and kick "
    "mono-compatible and centered, with stereo width used for pads, ambience, and air rather "
    "than low end. Avoid low-bitrate, watery, phasey, over-compressed, or lo-fi codec "
    "artifacts; the mix should sound like a polished, warm, radio-ready master, not a bright "
    "or clinical one."
)
MELODY_REGISTER_GUIDANCE = (
    "Keep melodies, hooks, leads, solos, arpeggios, and ornamental phrases in a warm mid-range "
    "or low-mid register. Do not put melodic content in piercing high registers. Avoid shrill "
    "top-line synths, whistling leads, glassy high piano, squeaky strings, thin flutes, or "
    "repetitive high-frequency motifs. High frequencies should provide natural instrument air, "
    "drum detail, and room tone, not carry the main melody."
)
# Positive caption text (MELODY_REGISTER_GUIDANCE above) asks the LM to avoid high-register
# synths, but caption prose is only ever a soft nudge on the LM's own generated caption --
# it's not guaranteed to survive, same class of gap as MIN_LAYER_GUARANTEE. lm_negative_prompt
# is a stronger, more direct lever: the LM handler substitutes this text as the CFG
# *unconditional* prompt, so lm_cfg_scale actively steers composition away from it rather
# than just being asked nicely. Kept as a CLI default (not hardcoded) so more exclusions can
# be added/overridden per run without another code change.
DEFAULT_NEGATIVE_PROMPT = (
    "shrill high-pitched synths, piercing high-frequency lead melodies, squeaky thin "
    "top-line notes, screechy high synth stabs"
)
MINIMAL_ARRANGEMENT_GUIDANCE = (
    "Keep the arrangement minimal and spacious like a polished, professional record that "
    "trusts restraint, not a beginner or demo-level arrangement. Use only a few purposeful "
    "layers at once: drums, bass, and always at least one melodic or harmonic instrument "
    "(guitar, piano, synth, horn, strings, etc.) -- never drums and bass alone, that reads "
    "as an unfinished sketch rather than a minimal production. Each layer should be "
    "performed with the same skill, groove, and detail as a full commercial production; the "
    "simplicity must come from smart arrangement and mixing choices, not from thin, basic, "
    "or amateur-sounding parts. Give instruments complementary rhythmic and melodic roles so "
    "they interlock and answer each other instead of clashing or masking one another; use "
    "call-and-response phrasing, syncopation, and space between hits so the mix breathes "
    "without losing musical sophistication. Avoid thick pad or texture layers that blur "
    "into other instruments, and avoid constant busy fills. It must still sound like a "
    "fully produced, multi-layered real record with a professional mix, achieved by "
    "disciplined arrangement and mixing skill, never by making individual parts sound "
    "simple, empty, or unpolished."
)
PERFORMANCE_SKILL_GUIDANCE = (
    "Every instrument must be performed by expert, professional session musicians with "
    "confident, precise technique: accurate pitch and intonation, steady confident timing, "
    "controlled dynamics, and musically mature phrasing. Do not sound like a beginner, "
    "student, child, or amateur hobbyist playing; avoid hesitant, wobbly, out-of-tune, "
    "rushed, or uncertain-sounding performances. Riffs, melodies, and solos should sound "
    "deliberate and skillfully executed, not simplistic, tentative, or naive."
)
MUSICALITY_GUIDANCE = (
    "Write real musical content, not a static loop: use a chord progression that actually "
    "moves and resolves rather than droning on one chord or note, with occasional passing "
    "chords or secondary harmony idiomatic to the genre. Give melodies a clear shape -- a "
    "beginning, a rise, and a resolution -- instead of repeating the same one or two notes. "
    "Vary dynamics and energy across the arrangement so choruses feel like a lift and verses "
    "feel like a pull-back. Use call-and-response phrasing between instruments and small "
    "melodic variations between repeats so the harmony and melody feel composed, not looped."
)
# Short fallback quality nudge for the final DiT caption when no LM-authored caption is
# available. Training captions read as one natural paragraph (~80-120 words); the DiT
# text encoder was never shown directive/checklist-style prompts, so keep this brief
# rather than concatenating the long guidance blocks above (those are for the LM query
# in create_genre_prompt, which is expected to condense them into natural prose).
CONCISE_QUALITY_HINT = (
    "Performed by skilled professional musicians with a clean, polished mix, real chord "
    "movement, and dynamic contrast between sections rather than a static loop."
)
# Applied unconditionally (see build_generation_params) rather than checked-and-skipped
# like the hints above: "minimal arrangement" guidance can get interpreted as license to
# go all the way down to just drums + bass, which reads as an unfinished sketch rather
# than a minimal *production*. This is a hard floor, safe to always state.
MIN_LAYER_GUARANTEE = (
    "The arrangement must never be just drums and bass alone -- always layer in at least "
    "one clear melodic or harmonic instrument (guitar, piano, synth, horn, strings, etc.) "
    "idiomatic to the genre, audible throughout."
)
# Same unconditional-guarantee pattern as MIN_LAYER_GUARANTEE: the fuller WARM_MIX_GUIDANCE/
# MIX_CLARITY_GUIDANCE text only reaches the LM query (create_genre_prompt), so an LM-authored
# caption isn't guaranteed to carry the bass/highs balance through, and the profile-fallback
# caption never mentioned it at all -- some renders came out with weak/missing bass or
# fizzy, dominant hi-hats and cymbals. State it directly and briefly in every final caption.
FREQUENCY_BALANCE_GUARANTEE = (
    "Full-frequency mix with present, deep, well-defined bass and low end throughout -- "
    "never thin or bass-light. Hi-hats, cymbals, and percussion stay smooth and "
    "well-controlled, never harsh, fizzy, piercing, or dominating the mix."
)
SECTION_BLUEPRINT = [
    ("Intro", 0.12, "sparse setup over the same steady pulse, hinting the groove"),
    ("Chorus", 0.20, "main hook, fuller drums, strongest motif; groove and tempo unchanged"),
    ("Verse", 0.19, "reduced arrangement, new melodic movement; same steady groove"),
    ("Chorus", 0.20, "hook returns with extra layers and fills; groove and tempo unchanged"),
    ("Verse", 0.19, "second variation, different instrument focus; same steady groove"),
    ("Outro", 0.10, "strip down over the same pulse and resolve cleanly"),
]
TITLE_WORDS = {
    "afropop": ["sunrise", "lagos", "golden", "palm", "market", "joy", "highlife"],
    "arabic": ["oud", "desert", "moon", "maqam", "cairo", "silk", "dawn"],
    "bachata": ["luna", "dulce", "corazon", "romance", "guitarra", "noche", "suspiro"],
    "calm jazzy piano": ["nocturne", "candlelight", "velvet", "hush", "lullaby", "twilight", "reverie"],
    "celtic": ["emerald", "highland", "misty", "glen", "windward", "ancient", "moor"],
    "chill": ["midnight", "soft", "drift", "haze", "quiet", "cloud", "afterglow"],
    "country": ["dust", "highway", "porch", "whiskey", "hometown", "backroad", "sundown"],
    "deep house": ["basement", "velvet", "night", "pulse", "subway", "afterhours", "shadow"],
    "drum & bass": ["break", "sub", "rush", "jungle", "night", "pressure", "motion"],
    "electronic": ["neon", "signal", "circuit", "chrome", "future", "voltage", "motion"],
    "funk": ["pocket", "strut", "groove", "brass", "snap", "velvet", "downtown"],
    "hip hop": ["block", "cipher", "dust", "808", "corner", "sample", "midnight"],
    "house": ["warehouse", "piano", "floor", "jack", "sunset", "groove", "anthem"],
    "indian": ["monsoon", "tabla", "sitar", "rang", "river", "palace", "dusky"],
    "jazz": ["blue", "smoke", "late", "quartet", "brushed", "uptown", "velvet"],
    "pop": ["daylight", "spark", "radio", "summer", "heart", "city", "bright"],
    "r&b": ["silk", "velvet", "slow", "afterglow", "midnight", "rose", "warm"],
    "reggae": ["island", "dub", "sun", "roots", "tide", "skank", "irie"],
    "reggaeton": ["perreo", "calle", "noche", "ritmo", "bajo", "dembow", "fuego"],
    "salsa": ["ritmo", "clave", "conga", "sabor", "calle", "noche", "son"],
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
        "[Chorus]",
        "[Verse]",
        "[Chorus]",
        "[Verse]",
        "[Outro]",
    ]
)
GENRES = [
    "afropop",
    "arabic",
    "bachata",
    "calm jazzy piano",
    "celtic",
    "chill",
    "country",
    "deep house",
    "drum & bass",
    "electronic",
    "funk",
    "hip hop",
    "house",
    "indian",
    "jazz",
    "pop",
    "r&b",
    "reggae",
    "reggaeton",
    "salsa",
    "soul",
    "spanish",
]

GENRE_PROFILES = {
    "afropop": {
        "caption": (
            "afropop performed by a small live band: bright interlocking guitar riffs, "
            "warm electric bass, a tight drum kit, and one upfront shaker as the only "
            "hand percussion, close-miked with every hit distinct, recorded live in a "
            "studio with a wide natural stereo image and an upbeat dance groove"
        ),
        "bpm": 105,
        "bpm_range": (96, 116),
    },
    "arabic": {
        "caption": (
            "Arabic fusion quartet: expressive lead oud, warm legato strings, deep bass, "
            "and a single darbuka carrying the rhythm alone, close-miked and articulate, "
            "modal maqam harmony, spacious cinematic studio recording with a wide stereo "
            "image and a calm desert-night mood"
        ),
        "bpm": 96,
        "bpm_range": (78, 112),
    },
    "calm jazzy piano": {
        "caption": (
            "solo jazz piano performance with soft brushed rhythm section support: "
            "expressive rubato piano lead, warm upright bass, brushed drum kit played "
            "very softly, occasional muted trumpet or saxophone coloring, intimate "
            "late-night lounge recording with natural room ambience and a calm, "
            "relaxed, introspective mood"
        ),
        "bpm": 70,
        "bpm_range": (56, 88),
        "extra_guidance": (
            "Drums must stay very soft and quiet the entire track -- brushed kit only, "
            "no loud hits, accents, fills, or cymbal crashes; the drums sit gently in the "
            "background supporting the piano and never draw attention to themselves."
        ),
    },
    "celtic": {
        "caption": (
            "Celtic folk ensemble: lyrical fiddle lead, wooden tin whistle, warm "
            "acoustic guitar, bodhran hand drum, and occasional Celtic harp, "
            "close-miked traditional Irish session recording with natural room "
            "ambience, wide stereo image, and a lilting, danceable folk groove"
        ),
        "bpm": 110,
        "bpm_range": (86, 128),
    },
    "chill": {
        "caption": (
            "chill downtempo trio: soft electric piano, one mellow pad, relaxed sparse "
            "drum kit, and warm round bass, clean and uncluttered, calm late-night "
            "studio atmosphere with gentle stereo width"
        ),
        "bpm": 82,
        "bpm_range": (72, 92),
        "extra_guidance": (
            "Keep the pad and electric piano in a warm mid or low-mid register -- avoid "
            "bright, glassy, shimmering, or high-register synth tones anywhere in the mix."
        ),
    },
    "country": {
        "caption": (
            "country song performed by a small live band: bright acoustic and "
            "electric guitar interplay, weeping pedal steel guitar, warm upright or "
            "electric bass, a tight brushed drum kit, and occasional fiddle accents, "
            "close-miked with a warm Nashville studio sound, wide natural stereo "
            "image and a heartfelt storytelling groove"
        ),
        "bpm": 100,
        "bpm_range": (76, 130),
    },
    "deep house": {
        "caption": (
            "deep house with a steady four-on-the-floor kick, one crisp hi-hat pattern, "
            "warm sub bass, muted chord stabs, and a single spacious pad, hypnotic and "
            "clean club mix with tight punchy low end"
        ),
        "bpm": 124,
        "bpm_range": (118, 126),
        "extra_guidance": (
            "Keep the chord stabs and pad in a warm mid or low-mid register -- avoid "
            "bright, glassy, shimmering, or high-register synth tones anywhere in the mix."
        ),
    },
    "drum & bass": {
        "caption": (
            "drum and bass built on one tightly edited breakbeat, deep rolling sub bass, "
            "a single atmospheric pad, and sparse melodic stabs, every drum hit punchy "
            "and separated, clean energetic club mix"
        ),
        "bpm": 174,
        "bpm_range": (160, 178),
        "extra_guidance": (
            "This must feel unmistakably fast and driving at full tempo (140+ BPM, true "
            "double-time breakbeat energy) throughout -- never a half-time, laid-back, or "
            "downtempo feel; avoid any dragging or slowed-down groove."
        ),
    },
    "electronic": {
        "caption": (
            "electronic track with one polished synth lead, one supporting arpeggio, "
            "punchy tight drums, and deep clean bass, evolving but uncluttered modern "
            "production with clear separation between parts"
        ),
        "bpm": 128,
        "bpm_range": (118, 132),
        "extra_guidance": (
            "Keep the synth lead and arpeggio in a warm mid or low-mid register -- avoid "
            "bright, glassy, shimmering, or high-register synth tones anywhere in the mix."
        ),
    },
    "funk": {
        "caption": (
            "funk quartet recorded live in the studio: tight slap bass, one crisp rhythm "
            "guitar, a pocket drum kit, and clavinet accents, punchy horn stabs used "
            "sparingly, dry close-miked mix with wide stereo drums and an infectious groove"
        ),
        "bpm": 108,
        "bpm_range": (92, 116),
    },
    "hip hop": {
        "caption": (
            "hip hop beat with hard punchy drums, deep 808 bass, one chopped melodic "
            "sample, and sparse keys, roomy uncluttered mix with a confident head-nod "
            "groove"
        ),
        "bpm": 92,
        "bpm_range": (78, 98),
    },
    "house": {
        "caption": (
            "house track with a driving four-on-the-floor kick, one clean hi-hat "
            "pattern, groovy bassline, warm piano stabs, and one vocal chop hook, "
            "uplifting and polished club mix with clear separation"
        ),
        "bpm": 126,
        "bpm_range": (120, 128),
        "extra_guidance": (
            "Keep the piano stabs and vocal chop hook in a warm mid or low-mid register -- "
            "avoid bright, glassy, shimmering, or high-register tones anywhere in the mix."
        ),
    },
    "indian": {
        "caption": (
            "Indian fusion trio: expressive sitar lead, warm cinematic strings, deep "
            "bass, and tabla as the only percussion, close-miked with each stroke "
            "articulate and distinct, polished spacious studio recording with wide "
            "stereo image"
        ),
        "bpm": 100,
        "bpm_range": (84, 112),
    },
    "jazz": {
        "caption": (
            "jazz quartet recorded live to tape in a warm studio room: brushed drum kit, "
            "walking upright bass, warm piano comping, and one saxophone lead, intimate "
            "close-miked 1960s session sound with natural room ambience, wide stereo "
            "image, and a relaxed improvisational feel"
        ),
        "bpm": 116,
        "bpm_range": (88, 128),
    },
    "pop": {
        "caption": (
            "pop song with one catchy lead hook, bright tight drums, one polished synth "
            "layer, warm bass, and clean rhythm guitar, radio-ready chorus-focused "
            "production with clear separation between parts"
        ),
        "bpm": 112,
        "bpm_range": (96, 124),
        "extra_guidance": (
            "Keep the lead hook and synth layer in a warm mid or low-mid register -- avoid "
            "bright, glassy, shimmering, or high-register synth tones anywhere in the mix."
        ),
    },
    "r&b": {
        "caption": (
            "r&b band: silky electric piano, smooth deep bass, a crisp minimal drum kit, "
            "and one lead guitar or synth line, lush but uncluttered studio mix with a "
            "slow emotional groove and wide warm stereo image"
        ),
        "bpm": 78,
        "bpm_range": (68, 92),
        "extra_guidance": (
            "Keep the lead guitar or synth line in a warm mid or low-mid register -- avoid "
            "bright, glassy, shimmering, or high-register tones anywhere in the mix."
        ),
    },
    "reggae": {
        "caption": (
            "reggae band recorded live: offbeat guitar skank, deep rounded bass, one "
            "relaxed drum kit with rimshots and one-drop groove, and organ bubble, warm "
            "spacious island mix with roomy stereo drums"
        ),
        "bpm": 76,
        "bpm_range": (68, 88),
    },
    "soul": {
        "caption": (
            "soul band recorded live in the studio: warm vintage electric piano, one "
            "expressive guitar, a live drum kit, rich bass, and brass accents used "
            "sparingly, heartfelt vintage session sound with natural room ambience and "
            "wide stereo image"
        ),
        "bpm": 94,
        "bpm_range": (76, 104),
    },
    "salsa": {
        "caption": (
            "high-energy salsa dance band: bright piano montuno pattern, syncopated conga "
            "and timbale percussion driving the clave, güira scraper, walking upright bass "
            "tumbao, and punchy trumpet and trombone horn stabs, polished Latin dance-floor "
            "recording with a wide, energetic stereo mix"
        ),
        "bpm": 180,
        "bpm_range": (160, 200),
        "extra_guidance": (
            "This must sound unmistakably like classic salsa dance music -- driving "
            "clave-based percussion, a clear piano montuno, and horn stabs must all be "
            "clearly audible throughout, not generic Latin pop."
        ),
    },
    "bachata": {
        "caption": (
            "romantic bachata band: lead requinto guitar playing melodic arpeggiated runs, "
            "rhythm guitar strumming the bachata pattern, bongo and güira percussion, warm "
            "rounded electric bass, intimate close-miked Dominican studio recording with a "
            "romantic, danceable groove"
        ),
        "bpm": 130,
        "bpm_range": (120, 145),
        "extra_guidance": (
            "This must sound unmistakably like bachata -- the lead requinto guitar melody, "
            "rhythm guitar strum pattern, and bongo/güira groove must all be clearly audible "
            "throughout, not generic Latin pop or flamenco."
        ),
    },
    "reggaeton": {
        "caption": (
            "modern reggaeton track built on a produced dembow riddim drum pattern (kick-kick-"
            "snare boom-ch-boom-chick syncopation) and deep sub-bass, layered with real live "
            "percussion (congas and bongos) and a punchy horn section or Spanish guitar riff "
            "for organic texture, dark moody synth pads used sparingly underneath, crisp "
            "modern hi-hats, polished urban Latin club production with heavy low end"
        ),
        "bpm": 95,
        "bpm_range": (88, 100),
        "extra_guidance": (
            "This must sound unmistakably like reggaeton, built on the dembow riddim -- not "
            "a hip-hop, trap, or boom-bap beat; the drum pattern must follow the dembow "
            "kick-kick-snare syncopation, not a generic hip-hop groove. Real live percussion "
            "(congas/bongos) and a horn section or guitar riff must be clearly audible "
            "layered with the produced beat -- this should not be synths alone. Keep any "
            "synth pads in a warm mid or low-mid register, used sparingly and only as "
            "background texture -- avoid bright, glassy, shimmering, or high-register synth "
            "tones anywhere in the mix."
        ),
    },
}

ARTIST_REFERENCES = {
    "afropop": ["Burna Boy", "Wizkid", "Davido", "Tiwa Savage", "Yemi Alade"],
    "arabic": ["Amr Diab", "Fairuz", "Nancy Ajram", "Saad Lamjarred", "Elissa"],
    "calm jazzy piano": ["Bill Evans", "Brad Mehldau", "Keith Jarrett", "Ryuichi Sakamoto", "Yiruma"],
    "celtic": ["Enya", "The Chieftains", "Clannad", "Loreena McKennitt", "Altan"],
    "chill": ["Bonobo", "Tycho", "Jinsang", "Rhye", "Kiasmos"],
    "country": ["Chris Stapleton", "Luke Combs", "Kacey Musgraves", "Zach Bryan", "Miranda Lambert"],
    "deep house": ["Black Coffee", "Disclosure", "Lane 8", "Bonobo", "Kerri Chandler"],
    "drum & bass": ["Netsky", "Sub Focus", "Andy C", "High Contrast", "Pendulum"],
    "electronic": ["ODESZA", "Flume", "RUFUS DU SOL", "Kaskade", "Porter Robinson"],
    "funk": ["Bruno Mars", "Vulfpeck", "Cory Wong", "Earth, Wind & Fire", "Chromeo"],
    "hip hop": ["Kendrick Lamar", "J. Cole", "Drake", "Travis Scott", "Nas"],
    "house": ["Fisher", "Disclosure", "Duke Dumont", "Purple Disco Machine", "CamelPhat"],
    "indian": ["A.R. Rahman", "Shreya Ghoshal", "Arijit Singh", "Nusrat Fateh Ali Khan", "Ravi Shankar"],
    "jazz": ["Miles Davis", "John Coltrane", "Herbie Hancock", "Norah Jones", "Diana Krall"],
    "pop": ["Taylor Swift", "Dua Lipa", "Ed Sheeran", "The Weeknd", "Ariana Grande"],
    "r&b": ["SZA", "Frank Ocean", "H.E.R.", "Daniel Caesar", "Summer Walker"],
    "reggae": ["Bob Marley", "Chronixx", "Damian Marley", "Sean Paul", "Protoje"],
    "soul": ["Aretha Franklin", "Sam Cooke", "Anderson .Paak", "Leon Bridges", "Amy Winehouse"],
    "salsa": ["Marc Anthony", "Celia Cruz", "Rubén Blades", "Héctor Lavoe", "Willie Colón"],
    "bachata": ["Romeo Santos", "Aventura", "Juan Luis Guerra", "Prince Royce", "Antony Santos"],
    "reggaeton": ["Bad Bunny", "J Balvin", "Daddy Yankee", "Karol G", "Rauw Alejandro"],
}

# Signature instruments/production elements per genre, drawn from that genre's own
# GENRE_PROFILES caption so they never contradict it. Soft mentions in the profile caption
# (e.g. afropop's "bright interlocking guitar riffs") aren't reliably followed -- generated
# tracks have come out missing the one instrument that actually makes a genre read as
# genre-authentic and "professional" rather than generic. resolve_key_ingredients() picks
# 1-2 per track and key_ingredient_guidance() states them as a hard, unconditional
# requirement, the same escalation already used for MIN_LAYER_GUARANTEE.
KEY_INGREDIENTS = {
    "afropop": ["interlocking electric guitar riffs", "upfront shaker percussion", "bright horn stabs"],
    "arabic": ["expressive oud lead", "darbuka hand percussion", "legato string swells"],
    "calm jazzy piano": ["expressive piano lead", "muted trumpet or saxophone coloring", "softly brushed drum kit"],
    "celtic": ["lyrical fiddle lead", "wooden tin whistle", "celtic harp accents"],
    "chill": ["soft electric piano", "mellow synth pad", "warm round bass"],
    "country": ["weeping pedal steel guitar", "fiddle accents", "bright acoustic guitar interplay"],
    "deep house": ["muted chord stabs", "spacious analog pad", "crisp hi-hat pattern"],
    "drum & bass": ["atmospheric pad", "sparse melodic stabs", "rolling sub bass"],
    "electronic": ["polished synth lead", "supporting arpeggio", "deep clean bass"],
    "funk": ["tight slap bass", "clavinet accents", "punchy horn stabs"],
    "hip hop": ["chopped melodic sample", "sparse electric keys", "deep 808 bass"],
    "house": ["warm piano stabs", "groovy bassline", "vocal chop hook"],
    "indian": ["expressive sitar lead", "tabla percussion", "cinematic string swells"],
    "jazz": ["walking upright bass", "warm piano comping", "saxophone lead"],
    "pop": ["polished synth layer", "clean rhythm guitar", "catchy lead hook"],
    "r&b": ["silky electric piano", "lead guitar or synth line", "smooth deep bass"],
    "reggae": ["offbeat guitar skank", "organ bubble", "deep rounded bass"],
    "soul": ["vintage electric piano", "expressive guitar", "brass accents"],
    "salsa": ["piano montuno pattern", "conga and timbale percussion", "punchy horn stabs"],
    "bachata": ["lead requinto guitar", "bongo and güira percussion", "rhythm guitar strum pattern"],
    "reggaeton": ["dembow riddim drum pattern", "live congas or horn section", "deep sub-bass"],
}


def resolve_key_ingredients(genre: str) -> list[str]:
    """Return 1-2 randomly chosen signature instruments/elements to force for this track."""
    candidates = KEY_INGREDIENTS.get(genre, [])
    if not candidates:
        return []
    count = min(len(candidates), random.randint(1, 2))
    return random.sample(candidates, count)


def key_ingredient_guidance(key_ingredients: list[str]) -> str:
    """Return a hard, unconditional instruction to make specific ingredients audible."""
    if not key_ingredients:
        return ""
    joined = " and ".join(key_ingredients)
    return (
        f"This track must clearly and audibly feature {joined} -- not faintly buried in "
        "the background, but a clear, prominent, unmistakable presence in the mix "
        "throughout the track."
    )


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
        "--continue",
        dest="resume",
        action="store_true",
        help=(
            "Resume into an existing --output-dir instead of regenerating from track 1. "
            "For each genre, counts audio files already named with that genre's slug "
            "prefix and skips ahead to that track index, so an interrupted --all-genres "
            "batch can be restarted without redoing finished genres/tracks."
        ),
    )
    parser.add_argument(
        "--cover",
        action="store_true",
        help=(
            "Add a real-artist style reference (\"<artist> type beat\") to the "
            "generation prompt for closer genre-matching production style. Picks a "
            "random well-known artist for the genre unless --artist is given. This "
            "steers instrumentation/mood only -- the model has no knowledge of the "
            "artist's actual catalog, so output will not sound like the artist's "
            "voice or any specific real song."
        ),
    )
    parser.add_argument(
        "--artist",
        default=None,
        help="Specific artist name to reference with --cover. Random per-genre pick if omitted.",
    )
    parser.add_argument(
        "--style-reference",
        default=None,
        help=(
            "Path to a reference audio file (e.g. a mastered track) whose production "
            "style influences generation via ACE-Step's reference-audio conditioning. "
            "Duration stays controlled by --duration -- this does not lock output "
            "length to the reference file's length, and does not copy its content."
        ),
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
        "--candidates",
        type=int,
        default=1,
        help=(
            "Generate this many independent candidates per track and automatically keep "
            "only the best-scoring one (see score_track.py: spectral fizz, transient "
            "smear, bandwidth, stereo phase, optionally audiobox-aesthetics). Diffusion "
            "seed-to-seed variance is large -- this is usually a bigger quality lever "
            "than the whole post-processing chain. Default 1 = current single-render "
            "behavior. Multiplies generation cost by N and forces --concurrency 1."
        ),
    )
    parser.add_argument(
        "--candidate-reference",
        default=None,
        help="Reference track for --candidates scoring's tonal-balance-distance metric.",
    )
    parser.add_argument(
        "--candidate-use-aesthetics",
        action="store_true",
        help="Also score --candidates with the audiobox-aesthetics model (tools/audiobox_venv).",
    )
    parser.add_argument(
        "--target-lufs",
        type=float,
        default=DEFAULT_TARGET_LUFS,
        help="Post-export integrated loudness target. Default: -14 LUFS.",
    )
    parser.add_argument(
        "--true-peak-db",
        type=float,
        default=DEFAULT_TRUE_PEAK_DB,
        help=(
            "True-peak ceiling for the final limiting stage, in dBTP. Default: -1.0 "
            "(streaming-safe; leaves headroom for lossy re-encoding)."
        ),
    )
    parser.add_argument(
        "--disable-lufs-normalization",
        action="store_true",
        help="Skip the final loudness stage (gain to target LUFS + true-peak limiting).",
    )
    parser.add_argument(
        "--tone-profile",
        default=DEFAULT_TONE_PROFILE,
        choices=list(TONE_PROFILE_NAMES),
        help=(
            "Mastering tone profile. 'reference' matches the analyzed reference master "
            "(sub-forward, warm, smooth dark top, mono bass); 'neutral' is corrective "
            "only; 'bright' is the previous clarity/exciter chain. Default: reference."
        ),
    )
    parser.add_argument(
        "--disable-clarity-mastering",
        action="store_true",
        help="Skip the tone-mastering pass (EQ, dynamic de-harsh, mono-bass crossover).",
    )
    parser.add_argument(
        "--mono-bass-crossover-hz",
        type=int,
        default=MONO_BASS_CROSSOVER_HZ,
        help=(
            "Frequency below which the mix is summed to mono during mastering. Default "
            "200 Hz covers sub weight, bass fundamentals, and a kick's knock/punch body; "
            "lower toward 100-120 if this is squashing legitimate bassline stereo "
            "movement, or raise past 250 for still-wider kick/bass."
        ),
    )
    parser.add_argument(
        "--stereo-width",
        type=float,
        default=DEFAULT_STEREO_WIDTH,
        help=(
            "Stereo side-channel level applied during tone mastering (ffmpeg stereotools "
            "slev). 1.0 = unchanged, below 1.0 narrows the stereo image (mono content is "
            "unaffected), above 1.0 widens. Default 0.7 pulls in a diffuse/unfocused image "
            "toward a more centered, professional-sounding mix."
        ),
    )
    parser.add_argument(
        "--enable-apollo-restoration",
        action="store_true",
        help=(
            "Run the isolated Apollo model (tools/apollo) to de-smear/restore audio "
            "before mastering. Requires the separate Apollo env to be set up."
        ),
    )
    parser.add_argument(
        "--apollo-checkpoint",
        default="restore",
        choices=["restore", "vocal", "vocal2", "universal"],
        help="Apollo checkpoint to use. 'restore' is the general codec de-smear model.",
    )
    parser.add_argument(
        "--enable-audiosr-upscale",
        action="store_true",
        help=(
            "Run the isolated AudioSR model (tools/audiosr_venv) to extend bandwidth to "
            "~24 kHz (48 kHz output) before mastering. Requires the separate AudioSR env "
            "to be set up; adds significant runtime per track."
        ),
    )
    parser.add_argument(
        "--audiosr-model",
        default="basic",
        choices=["basic", "speech"],
        help="AudioSR checkpoint to use.",
    )
    parser.add_argument(
        "--audiosr-ddim-steps",
        type=int,
        default=25,
        help=(
            "AudioSR DDIM sampling steps. Lower is faster but lower quality. Default "
            "halved from the library's own default of 50 -- 25 is a fast/quality "
            "middle ground for bandwidth extension (not generative content)."
        ),
    )
    parser.add_argument(
        "--audiosr-guidance-scale",
        type=float,
        default=3.5,
        help=(
            "AudioSR classifier-free guidance scale. AudioSR runs a second "
            "(unconditional) forward pass per DDIM step whenever this is not exactly "
            "1.0, roughly doubling per-step cost -- set to 1.0 to disable guidance "
            "and roughly halve runtime again, at some cost to output relevance/quality."
        ),
    )
    parser.add_argument(
        "--disable-vram-duration-retry",
        action="store_true",
        help="Fail immediately on VRAM errors instead of retrying shorter durations.",
    )
    parser.add_argument(
        "--fade-out-seconds",
        type=float,
        default=DEFAULT_FADE_OUT_SECONDS,
        help="Linear fade-out length at the end of each track. Default: 4.0s.",
    )
    parser.add_argument(
        "--disable-fade-out",
        action="store_true",
        help="Skip the end-of-track fade-out and leave a hard cutoff.",
    )
    parser.add_argument(
        "--fadeout",
        action="store_true",
        help=(
            "Repair mode: skip generation entirely and apply a --fade-out-seconds fade to "
            "every existing .mp3/.wav file directly in --output-dir. For tracks already on "
            "disk with an abrupt or noisy cutoff (e.g. from before the fade-out default was "
            "raised, or generated with --disable-fade-out)."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Deprecated alias for --amount.")
    parser.add_argument(
        "--quality",
        default="balanced",
        choices=["best", "extreme", "ultra", "high", "balanced", "fast"],
        help=(
            "Quality preset. best uses SFT; extreme/ultra/high use base+ADG "
            "at 150/100/64 steps; balanced uses base faster; fast uses turbo."
        ),
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
        "--lm-cfg-scale",
        type=float,
        default=2.0,
        help=(
            "5Hz LM classifier-free guidance scale (was hardcoded at 2.0). Higher = "
            "stronger LM adherence to the prompt/locked metadata (including keyscale) "
            "during composition -- this is the phase that writes the actual note/pitch "
            "content, so it's a more direct lever for key adherence than --guidance-scale, "
            "which mostly affects the DiT's rendering of caption/mix description."
        ),
    )
    parser.add_argument(
        "--negative-prompt",
        default=DEFAULT_NEGATIVE_PROMPT,
        help=(
            "5Hz LM negative prompt. The LM substitutes this as the CFG unconditional "
            "prompt, so lm_cfg_scale actively steers composition away from whatever this "
            "describes -- a stronger lever than caption-text guidance for excluding a "
            "specific character (e.g. shrill high synths). Only takes effect when "
            "--lm-cfg-scale > 1.0 (default 2.0). Pass '' to disable."
        ),
    )
    parser.add_argument(
        "--no-adg",
        action="store_true",
        help="Disable Adaptive Dual Guidance for non-turbo quality runs.",
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
    parser.add_argument("--lm-model", default="acestep-5Hz-lm-1.7B", help="5Hz LM checkpoint folder name.")
    parser.add_argument(
        "--lm-backend",
        default=os.environ.get("ACESTEP_LM_BACKEND", "pt"),
        help='LM backend: "vllm", "pt", or "mlx". Defaults to PyTorch for stable 1.7B CPU offload.',
    )
    parser.add_argument(
        "--offload",
        action="store_true",
        help="Force CPU offload. High quality enables offload automatically.",
    )
    parser.add_argument(
        "--offload-dit",
        action="store_true",
        help="Also offload the DiT model to CPU between phases to save more VRAM.",
    )
    parser.add_argument(
        "--no-offload",
        action="store_true",
        help="Disable automatic high-quality CPU offload.",
    )
    parser.add_argument(
        "--no-offload-dit",
        action="store_true",
        help="Disable automatic DiT CPU offload.",
    )
    parser.add_argument(
        "--quantization",
        default="auto",
        choices=["auto", "int8_weight_only", "fp8_weight_only", "w8a8_dynamic", "none"],
        help=(
            "DiT weight quantization. 'auto' only applies to XL (4B) DiT models, following "
            "the GPU tier default there (int8_weight_only on most CUDA GPUs, w8a8_dynamic "
            "on pre-Volta, disabled on Mac) -- 2B models (base/sft/turbo) always run full "
            "precision under 'auto' since they fit 8GB fine unquantized and quantizing them "
            "forces a broken per-parameter CPU<->GPU transfer fallback on older torch that "
            "isn't stable over long runs. Force a value to override either way."
        ),
    )
    args = parser.parse_args()
    amount = args.batch_size if args.batch_size is not None else args.amount
    if amount < 1:
        parser.error("--amount must be 1 or greater")
    if not args.all_genres and not args.genre and not args.fadeout:
        parser.error("--genre is required unless --all-genres or --fadeout is set")
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


SPANISH_SUBSTYLES = ["salsa", "bachata", "reggaeton"]


def resolve_spanish_substyle(genre: str) -> str:
    """Resolve the umbrella 'spanish' choice to one concrete Latin dance genre.

    salsa/bachata/reggaeton are selectable on their own via --genre, each with a
    dedicated GENRE_PROFILES/KEY_INGREDIENTS/ARTIST_REFERENCES entry. "spanish" is
    kept as a lighter-weight umbrella choice for when the caller wants "some Latin
    flavor" without picking a specific one -- it randomly resolves to one of the
    three per track so the caption, BPM, and instrumentation all agree on one
    concrete style instead of blurring several together. Every other genre,
    including salsa/bachata/reggaeton themselves, passes through unchanged.
    """
    if genre != "spanish":
        return genre
    return random.choice(SPANISH_SUBSTYLES)


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


def resolve_dcw_enabled(args: argparse.Namespace) -> bool:
    """Return the Gradio-default DCW setting for the selected DiT model."""
    return "turbo" in resolve_model(args).lower()


def resolve_use_adg(args: argparse.Namespace) -> bool:
    """Return whether Adaptive Dual Guidance should be used for quality output."""
    if args.no_adg:
        return False
    return (
        args.quality in ("best", "extreme", "ultra", "high")
        and "turbo" not in resolve_model(args).lower()
    )


def resolve_offload(args: argparse.Namespace) -> bool:
    """Return whether model offload should be used for this run."""
    if args.no_offload:
        return False
    return bool(
        args.offload
        or args.quality in ("best", "extreme", "ultra", "high", "balanced")
    )


def resolve_offload_dit(args: argparse.Namespace) -> bool:
    """Return whether DiT should be offloaded to CPU between generation phases."""
    if args.no_offload or args.no_offload_dit:
        return False
    if args.offload_dit:
        return True
    if args.quality in ("best", "extreme", "ultra", "high", "balanced"):
        return True
    gpu_config = get_global_gpu_config()
    return bool(resolve_offload(args) and getattr(gpu_config, "offload_dit_to_cpu_default", False))


def resolve_quantization(args: argparse.Namespace) -> str | None:
    """Return the DiT weight-quantization mode, mirroring acestep_v15_pipeline.py's default
    but scoped to models that actually need it.

    Previously unwired here -- DiT init always ran full precision regardless of GPU tier,
    which is harmless on the small 2B models but risks VRAM exhaustion (observed: a hard
    segfault) with the XL (4B) DiT / 4B LM on <=8GB GPUs where the tier config calls for
    INT8 quantization by default. Fixing that by blindly following the tier default for
    *every* model was itself a regression: quantizing the small 2B models forces a broken
    fallback path on this torch version (AffineQuantizedTensor.to() raises
    NotImplementedError, so parameters get moved one at a time on every single CPU<->GPU
    offload cycle instead of a batched .to() call) that isn't stable over many repeated
    cycles -- observed as a hard crash (silent, no Python traceback) after ~9 tracks in an
    --all-genres batch that ran for hours crash-free before quantization was ever wired up.
    2B models (base/sft/turbo) fit comfortably in 8GB unquantized -- only the XL (4B) DiT
    variants actually need this tradeoff.
    """
    if args.quantization == "none":
        return None
    if args.quantization != "auto":
        return args.quantization
    if is_mps_platform():
        return None
    if "-xl-" not in resolve_model(args).lower():
        return None
    gpu_config = get_global_gpu_config()
    if not getattr(gpu_config, "quantization_default", False):
        return None
    quantization = "int8_weight_only"
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] < 7:
            quantization = "w8a8_dynamic"
    except Exception as exc:
        logger.warning("Quantization auto-detect failed, using int8_weight_only: {}", exc)
    return quantization


def initialize_dit(args: argparse.Namespace) -> AceStepHandler:
    """Initialize and return the ACE-Step DiT handler."""
    handler = AceStepHandler()
    status, success = handler.initialize_service(
        project_root=str(PROJECT_ROOT),
        config_path=resolve_model(args),
        device=args.device,
        offload_to_cpu=resolve_offload(args),
        offload_dit_to_cpu=resolve_offload_dit(args),
        quantization=resolve_quantization(args),
    )
    if not success:
        raise RuntimeError(f"DiT initialization failed: {status}")
    logger.info(status)
    return handler


def initialize_lm(args: argparse.Namespace) -> LLMHandler:
    """Initialize and return the 5Hz language-model handler."""
    backend = resolve_lm_backend(args.lm_backend, get_global_gpu_config())
    lm_model_path = resolve_lm_model_path(args.lm_model)
    handler = LLMHandler()
    status, success = handler.initialize(
        checkpoint_dir=str(CHECKPOINT_DIR),
        lm_model_path=lm_model_path,
        backend=backend,
        device=args.device,
        offload_to_cpu=resolve_offload(args),
        dtype=None,
    )
    if not success:
        raise RuntimeError(f"5Hz LM initialization failed: {status}")
    logger.info(status)
    return handler


def resolve_lm_model_path(lm_model: str | None) -> str | None:
    """Return a normalized LM checkpoint path/name and fail early if it is missing."""
    if lm_model is None:
        return None

    normalized = lm_model.strip().strip("\"'")
    if not normalized:
        return None

    candidate = Path(normalized)
    if not candidate.is_absolute():
        candidate = CHECKPOINT_DIR / normalized

    if candidate.is_dir():
        return str(candidate)

    available_models = sorted(
        path.name
        for path in CHECKPOINT_DIR.iterdir()
        if path.is_dir() and "5Hz-lm" in path.name
    )
    available = ", ".join(available_models) if available_models else "none"
    raise RuntimeError(
        f"5Hz LM model not found at {candidate}. Available local LM models: {available}"
    )


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
    cover_artist: str | None = None,
    key_ingredients: list[str] | None = None,
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
        f"exact section timeline: {section_plan}. Open the caption by explicitly naming "
        f"the genre '{genre}' and it must strongly match "
        f"this genre profile: {profile['caption']}. Make it different from previous ideas, "
        f"with concrete instrumentation, arrangement, groove, mood, and production details. "
        f"{minor_key_guidance(keyscale)} "
        f"{key_ingredient_guidance(key_ingredients or [])} "
        f"{resolve_instrument_guidance(genre)} "
        f"Keep the rhythm locked to {profile['bpm']} BPM in 4/4. Drums must be on-beat, "
        f"groovy, natural, and idiomatic for {genre}; avoid rushed, unstable, or off-grid drums. "
        f"{profile.get('extra_guidance', '')} "
        f"{resolve_mix_guidance(args)} "
        f"{MELODY_REGISTER_GUIDANCE} "
        f"{MINIMAL_ARRANGEMENT_GUIDANCE} "
        f"{PERFORMANCE_SKILL_GUIDANCE} "
        f"{MUSICALITY_GUIDANCE} "
        f"Avoid simple looping by changing drums, bass, harmony, melody, fills, and energy "
        f"between sections while staying coherent."
        f"{extra_detail}"
        f"{cover_style_guidance(cover_artist)}"
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
    _ = genre, track_index, seed_offset
    if args.key:
        return args.key
    return random.choice(MINOR_KEYS)


def resolve_genre_bpm(args: argparse.Namespace, genre: str, sample: dict[str, object]) -> int:
    """Return user, smart, or random genre-safe BPM for one track."""
    if args.bpm is not None:
        return args.bpm
    profile = GENRE_PROFILES[genre]
    low, high = profile.get("bpm_range", (profile["bpm"], profile["bpm"]))
    if args.allow_smart_metadata:
        smart_bpm = coerce_optional_int(sample.get("bpm"))
        if smart_bpm is not None:
            # Clamp to the genre's bpm_range so smart metadata can't drift a track
            # (e.g. drum & bass) below the tempo the genre requires.
            return max(int(low), min(int(high), smart_bpm))
    return random.randint(int(low), int(high))


def resolve_cover_artist(args: argparse.Namespace, genre: str) -> str | None:
    """Return a real-artist style reference name for --cover prompting, or None."""
    if not args.cover:
        return None
    if args.artist:
        return args.artist
    candidates = ARTIST_REFERENCES.get(genre, [])
    if not candidates:
        return None
    return random.choice(candidates)


CAPTION_TOKEN_BUDGET = 190
# The DiT text encoder's live tokenization call (acestep/core/generation/handler/
# conditioning_text.py: self.text_tokenizer(text_prompt, truncation=True, max_length=256))
# truncates the caption *and the bpm/key/timesignature/duration metadata block that follows
# it in the same wrapped prompt* together to 256 tokens -- verified directly against that
# tokenizer: an unbounded version of this session's assembled caption (smart caption + all
# the guarantee sentences appended unconditionally) measured 326 tokens, and the overflow
# silently dropped not just the trailing guarantee text but the entire "# Metas" block.
# Budgeting the caption body to ~190 leaves the ~35-45 tokens the instruction+metas wrapper
# itself costs, so metadata is never what gets silently cut off the end.
_CAPTION_TOKENIZER = None
_CAPTION_TOKENIZER_UNAVAILABLE = False


def _caption_tokenizer():
    """Lazily load the DiT text encoder's own tokenizer for real token-budget checks."""
    global _CAPTION_TOKENIZER, _CAPTION_TOKENIZER_UNAVAILABLE
    if _CAPTION_TOKENIZER is not None or _CAPTION_TOKENIZER_UNAVAILABLE:
        return _CAPTION_TOKENIZER
    try:
        from transformers import AutoTokenizer

        _CAPTION_TOKENIZER = AutoTokenizer.from_pretrained(
            str(CHECKPOINT_DIR / "Qwen3-Embedding-0.6B"), trust_remote_code=True
        )
    except Exception as exc:
        logger.warning("Caption token-budget check unavailable, using char-count fallback: {}", exc)
        _CAPTION_TOKENIZER_UNAVAILABLE = True
    return _CAPTION_TOKENIZER


def _caption_token_count(text: str) -> int:
    """Return a real token count for budget checks, or a ~4-chars/token estimate as fallback."""
    tokenizer = _caption_tokenizer()
    if tokenizer is not None:
        return len(tokenizer(text).input_ids)
    return max(1, len(text) // 4)


def assemble_capped_caption(
    base_caption: str, additions: list[str], budget: int = CAPTION_TOKEN_BUDGET
) -> str:
    """Append caption additions in priority order, skipping any that would push the caption
    past the shared 256-token DiT text-encoder budget (see CAPTION_TOKEN_BUDGET above).
    Lower-priority additions are dropped whole -- never truncated mid-sentence -- and a
    shorter lower-priority addition can still slot in after an earlier, larger one didn't fit.
    """
    caption = base_caption.rstrip(".")
    for addition in additions:
        addition = (addition or "").strip().rstrip(".")
        if not addition:
            continue
        candidate = f"{caption}. {addition}"
        if _caption_token_count(candidate) <= budget:
            caption = candidate
    return caption + "."


def cover_style_guidance(artist: str | None) -> str:
    """Return a caption fragment steering generation toward an artist's style."""
    if not artist:
        return ""
    return (
        f" Style reference: {artist} type beat, matching their signature production "
        "style, instrumentation, and mood, without copying any specific existing song."
    )


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
    """Return simple instrumental section tags for the model lyrics field."""
    _ = duration
    return DEFAULT_INSTRUMENTAL_LYRICS


def build_generation_params(
    args: argparse.Namespace,
    genre: str,
    sample: dict[str, object],
    duration: float,
    keyscale: str,
    cover_artist: str | None = None,
    key_ingredients: list[str] | None = None,
) -> GenerationParams:
    """Build generation parameters from parsed command-line options."""
    profile = GENRE_PROFILES[genre]

    if args.duration < MIN_DURATION_SECONDS:
        logger.warning(
            "Duration base raised from {}s to minimum {}s",
            args.duration,
            MIN_DURATION_SECONDS,
        )
    locked_bpm = resolve_genre_bpm(args, genre, sample)
    time_signature = (
        str(sample.get("timesignature") or DEFAULT_TIME_SIGNATURE)
        if args.allow_smart_metadata
        else DEFAULT_TIME_SIGNATURE
    )

    # bpm/keyscale/timesignature/duration are conditioned via their own GenerationParams
    # fields below (matching the training data's separate JSON fields) -- do not restate
    # them as caption text, and do not embed literal section timestamps in the caption;
    # both are out-of-distribution for the caption text encoder, which only ever saw
    # natural single-paragraph song descriptions (~80-120 words) during training.
    smart_caption = str(sample.get("caption") or "").strip()
    if smart_caption:
        base_caption = smart_caption
    else:
        base_caption = profile["caption"]
        if args.prompt.strip():
            base_caption = f"{base_caption}, {args.prompt.strip()}"

    # GenerationParams has no dedicated genre field -- genre is conveyed purely through
    # caption text, and training captions consistently name the genre right at the start
    # (e.g. "An explosive... pop-rock track...", "A dark, atmospheric trap track...").
    # The LM's own caption isn't guaranteed to say "afropop" explicitly even when asked
    # to write one, so anchor it explicitly here rather than trusting that alone --
    # without it, ambiguous instrumentation descriptions can drift toward a different
    # genre's idiom (e.g. brushed/swung drums read as jazz instead of afropop). Done before
    # the budget-capped assembly below so the prefix is never at risk of being dropped.
    if genre.lower() not in base_caption.lower():
        base_caption = f"{genre} track: {base_caption}"

    # Everything below competes for a shared ~256-token DiT text-encoder budget alongside
    # the bpm/key/timesignature/duration metadata that follows the caption in the same
    # tokenized prompt (see CAPTION_TOKEN_BUDGET) -- listed in priority order, most
    # important first, since a lower-priority item gets dropped whole rather than corrupting
    # everything after it via mid-sentence truncation.
    quality_hint = (
        "" if "skilled" in base_caption.lower() or "professional" in base_caption.lower()
        else CONCISE_QUALITY_HINT
    )
    caption = assemble_capped_caption(
        base_caption,
        [
            # Highest priority: corrects the LM's own caption, which isn't guaranteed to
            # avoid describing vocals/singers even when instrumental=True is requested
            # (observed in practice: LM captions mentioning "female voice", "vocal chops").
            "fully instrumental with no sung or spoken vocals",
            key_ingredient_guidance(key_ingredients or []),
            MIN_LAYER_GUARANTEE,
            FREQUENCY_BALANCE_GUARANTEE,
            quality_hint,
            cover_style_guidance(cover_artist).strip(" ."),
        ],
    )

    return GenerationParams(
        task_type="text2music",
        caption=caption,
        reference_audio=args.style_reference,
        lyrics=build_instrumental_lyrics(duration),
        instrumental=True,
        vocal_language="unknown",
        bpm=locked_bpm,
        keyscale=keyscale,
        timesignature=time_signature,
        duration=duration,
        inference_steps=resolve_steps(args),
        guidance_scale=resolve_guidance_scale(args),
        use_adg=resolve_use_adg(args),
        shift=3.0,
        dcw_enabled=resolve_dcw_enabled(args),
        seed=args.seed,
        thinking=True,
        lm_temperature=0.85,
        lm_cfg_scale=args.lm_cfg_scale,
        lm_negative_prompt=args.negative_prompt or "NO USER INPUT",
        lm_top_p=0.9,
        use_cot_metas=True,
        use_cot_caption=False,
        use_cot_language=True,
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
        allow_lm_batch=True,
        use_random_seed=args.seed < 0,
        seeds=seeds,
        audio_format="wav" if args.format == "mp3" else args.format,
        mp3_bitrate=args.mp3_bitrate,
    )


def _run_ffmpeg(args: list[str], timeout: int = 180) -> subprocess.CompletedProcess:
    """Run ffmpeg with common non-interactive options."""
    command = ["ffmpeg", "-y", "-hide_banner", "-nostats", *args]
    return subprocess.run(command, check=True, capture_output=True, timeout=timeout)


def measure_loudness(path: Path, target_lufs: float) -> dict | None:
    """Measure integrated loudness, true peak, LRA, and threshold for loudnorm."""
    loudnorm = (
        f"loudnorm=I={target_lufs}:TP={DEFAULT_TRUE_PEAK_DB}:"
        f"LRA={DEFAULT_LOUDNESS_RANGE}:print_format=json"
    )
    try:
        proc = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-nostats",
                "-i",
                str(path),
                "-af",
                loudnorm,
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            timeout=180,
        )
    except subprocess.TimeoutExpired:
        logger.warning("Loudness measurement timed out for {}", path)
        return None

    stderr = proc.stderr.decode("utf-8", errors="ignore")
    match = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", stderr, re.DOTALL)
    if not match:
        logger.warning("Could not parse loudnorm measurement for {}", path)
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        logger.warning("Invalid loudnorm JSON for {}", path)
        return None


def finalize_loudness(
    path: str,
    target_lufs: float,
    true_peak_db: float = DEFAULT_TRUE_PEAK_DB,
) -> bool:
    """Final loudness stage: static gain to target LUFS, limiting only when needed.

    This is the mastering-engineer approach rather than loudnorm's linear mode:
    loudnorm with ``linear=true`` silently under-gains whenever the true-peak
    ceiling would be exceeded, so peaky tracks landed quieter than the target.
    Here the gain to hit the integrated target is always applied; if that would
    push the true peak past the ceiling, a 4x-oversampled limiter (true-peak
    aware in practice) catches only the overs. This must run *after* the tone
    stage and the fade-out so the measurement reflects the final audio.
    """
    if not path:
        return False
    source_path = Path(path)
    if not source_path.exists():
        logger.warning("Loudness finalization skipped; file not found: {}", source_path)
        return False
    if shutil.which("ffmpeg") is None:
        logger.warning("Loudness finalization skipped; ffmpeg is not available on PATH")
        return False

    measured = measure_loudness(source_path, target_lufs)
    if measured is None:
        return False
    try:
        input_i = float(measured["input_i"])
        input_tp = float(measured["input_tp"])
    except (KeyError, TypeError, ValueError):
        logger.warning("Loudness finalization skipped; unusable measurement for {}", source_path)
        return False
    if not (-70.0 < input_i < 10.0):
        logger.warning(
            "Loudness finalization skipped; measured {} LUFS looks like silence: {}",
            input_i,
            source_path,
        )
        return False

    gain_db = target_lufs - input_i
    predicted_tp = input_tp + gain_db
    if predicted_tp <= true_peak_db - LOUDNESS_LIMITER_MARGIN_DB:
        audio_filter = f"volume={gain_db:.2f}dB"
        mode = "pure gain"
    else:
        # Limit 0.1 dB below the ceiling: the 192 kHz-domain limiter is true-peak
        # accurate, but the final downsample to 48 kHz can reconstruct ~0.05 dB over.
        limit_linear = 10.0 ** ((true_peak_db - 0.1) / 20.0)
        audio_filter = (
            f"volume={gain_db:.2f}dB,"
            f"aresample=192000,"
            f"alimiter=limit={limit_linear:.6f}:attack=2:release=120:level=false,"
            f"aresample=48000"
        )
        mode = f"gain + limiter at {true_peak_db:.1f} dBTP"

    temp_path = source_path.with_name(f"{source_path.stem}.lufs_tmp{source_path.suffix}")
    try:
        _run_ffmpeg(
            [
                "-i",
                str(source_path),
                "-af",
                audio_filter,
                *_INTERMEDIATE_CODEC_ARGS,
                str(temp_path),
            ],
            timeout=300,
        )
        temp_path.replace(source_path)
        logger.info(
            "Finalized loudness to {} LUFS ({}; {:+.2f} dB): {}",
            target_lufs,
            mode,
            gain_db,
            source_path,
        )
        return True
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="ignore") if exc.stderr else str(exc)
        logger.warning("Loudness finalization failed for {}: {}", source_path, stderr)
        return False
    except subprocess.TimeoutExpired:
        logger.warning("Loudness finalization timed out for {}", source_path)
        return False
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                logger.warning("Could not remove temporary LUFS file: {}", temp_path)


def _mono_bass_split(inner_chain: str, crossover_hz: int = MONO_BASS_CROSSOVER_HZ) -> str:
    """Wrap a filter chain in an LR4 crossover that sums lows to mono.

    Two cascaded 2nd-order Butterworth sections per band form a Linkwitz-Riley 4th-order
    crossover, so the recombined bands sum flat. Everything below the crossover is
    center-panned -- standard practice on commercial masters (and what the reference does).

    Default crossover_hz (200) covers sub weight, bass fundamentals, and a kick's
    "knock"/punch body, which commonly extends past the sub-only ~120 Hz range.
    """
    xo = crossover_hz
    return (
        f"[0:a]asplit=2[lo_in][hi_in];"
        f"[lo_in]lowpass=f={xo}:p=2,lowpass=f={xo}:p=2,"
        f"pan=stereo|c0=0.5*c0+0.5*c1|c1=0.5*c0+0.5*c1[lo];"
        f"[hi_in]highpass=f={xo}:p=2,highpass=f={xo}:p=2[hi];"
        f"[lo][hi]amix=inputs=2:normalize=0,{inner_chain}[out]"
    )


TONE_PROFILE_NAMES = ("reference", "neutral", "bright")
_DYNEQ_LEGACY_SYNTAX: bool | None = None


def _dyneq_uses_legacy_syntax() -> bool:
    """Detect whether this ffmpeg's adynamicequalizer uses 5.x or 6.x option names.

    ffmpeg 5.x: ``mode=cutabove``. ffmpeg 6+: ``mode=cut:direction=downward``.
    The old hardcoded 'cutabove' made the entire mastering pass fail (and be
    silently skipped) on ffmpeg 6 and newer.
    """
    global _DYNEQ_LEGACY_SYNTAX
    if _DYNEQ_LEGACY_SYNTAX is None:
        try:
            proc = subprocess.run(
                ["ffmpeg", "-hide_banner", "-h", "filter=adynamicequalizer"],
                capture_output=True,
                timeout=15,
            )
            _DYNEQ_LEGACY_SYNTAX = b"cutabove" in proc.stdout + proc.stderr
        except (OSError, subprocess.TimeoutExpired):
            _DYNEQ_LEGACY_SYNTAX = False
    return _DYNEQ_LEGACY_SYNTAX


def _dyneq_deharsh(threshold: float, freq: int, q: float, ratio: float, range_db: float) -> str:
    """Build a downward dynamic-EQ band with version-appropriate syntax."""
    base = (
        f"adynamicequalizer=threshold={threshold}:dfrequency={freq}:dqfactor={q}:"
        f"tfrequency={freq}:tqfactor={q}:ratio={ratio}:range={range_db}:"
        f"attack=8:release=140"
    )
    if _dyneq_uses_legacy_syntax():
        return f"{base}:mode=cutabove"
    return f"{base}:mode=cut:direction=downward"


def build_tone_profile_chain(profile: str, stereo_width: float = DEFAULT_STEREO_WIDTH) -> str:
    """Tone-shaping chains derived from spectral analysis of the reference master
    (heavenly.mp3): sub-forward low end, mono bass, controlled 300 Hz region, soft
    presence, smooth top rolling off ~14 kHz, width concentrated in the air band.
    No chain limits or clips -- final dynamics control happens exactly once, at the
    loudness stage, after the fade-out.
    """
    if profile == "reference":
        # Matches the reference: warm, sub-weighted, dark-smooth top, de-harshed AI highs.
        stages = [
            "highpass=f=20:p=2",  # DC/rumble only -- the reference keeps deep sub energy
            "bass=g=1.5:f=90:w=0.6",  # sub/low-bass weight
            "equalizer=f=350:t=q:w=1.4:g=-1.5",  # clear mud so the added sub stays defined
            _dyneq_deharsh(10, 7500, 0.8, 2.5, 5),  # tames AI fizz only when it flares up
            "treble=g=-1.5:f=13500:w=0.5",  # gentle dark tilt like the reference's soft top
        ]
    elif profile == "bright":
        # Previous clarity-first behavior (minus the premature limiter).
        stages = [
            "highpass=f=28",
            "equalizer=f=240:t=q:w=1.8:g=-2.5",
            "equalizer=f=3200:t=q:w=0.8:g=0.7",
            "equalizer=f=5500:t=q:w=0.8:g=0.8",
            _dyneq_deharsh(12, 10000, 0.5, 2, 6),
            "aexciter=amount=1.4:drive=4:blend=0:freq=7500:ceil=15500",
        ]
    else:
        # Flat, corrective-only: safe default for unknown material.
        stages = [
            "highpass=f=24:p=2",
            "equalizer=f=300:t=q:w=1.4:g=-1.2",
            _dyneq_deharsh(11, 8500, 0.7, 2, 5),
        ]
    if stereo_width != 1.0:
        stages.append(f"stereotools=slev={stereo_width}")
    return ",".join(stages)


def apply_tone_mastering(
    path: str,
    profile: str = DEFAULT_TONE_PROFILE,
    mono_bass_crossover_hz: int = MONO_BASS_CROSSOVER_HZ,
    stereo_width: float = DEFAULT_STEREO_WIDTH,
) -> bool:
    """Apply the tone-shaping stage (EQ + dynamic de-harsh + mono bass, no limiting)."""
    if not path:
        return False
    source_path = Path(path)
    if not source_path.exists():
        logger.warning("Tone mastering skipped; file not found: {}", source_path)
        return False
    if shutil.which("ffmpeg") is None:
        logger.warning("Tone mastering skipped; ffmpeg is not available on PATH")
        return False

    if profile not in TONE_PROFILE_NAMES:
        profile = DEFAULT_TONE_PROFILE
    inner_chain = build_tone_profile_chain(profile, stereo_width)
    filter_complex = _mono_bass_split(inner_chain, mono_bass_crossover_hz)
    temp_path = source_path.with_name(f"{source_path.stem}.clarity_tmp{source_path.suffix}")
    try:
        _run_ffmpeg(
            [
                "-i",
                str(source_path),
                "-filter_complex",
                filter_complex,
                "-map",
                "[out]",
                *_INTERMEDIATE_CODEC_ARGS,
                str(temp_path),
            ]
        )
        temp_path.replace(source_path)
        logger.info(
            "Applied tone mastering ('{}' profile, mono-bass LR4 @ {} Hz, stereo width {}): {}",
            profile,
            mono_bass_crossover_hz,
            stereo_width,
            source_path,
        )
        return True
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="ignore") if exc.stderr else str(exc)
        logger.warning("Tone mastering failed for {}: {}", source_path, stderr)
        return False
    except subprocess.TimeoutExpired:
        logger.warning("Tone mastering timed out for {}", source_path)
        return False
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                logger.warning("Could not remove temporary tone-mastering file: {}", temp_path)


def _venv_python(venv_dir: Path) -> Path:
    """Return an isolated venv's python executable, Windows or POSIX layout."""
    windows_path = venv_dir / "Scripts" / "python.exe"
    if windows_path.exists():
        return windows_path
    return venv_dir / "bin" / "python"


def _run_subprocess_streaming(command: list[str], timeout: int) -> tuple[int, str]:
    """Run a subprocess, echoing its output live instead of buffering it silently."""
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    lines: list[str] = []

    def _pump() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            lines.append(line)
            print(line, end="", flush=True)

    pump_thread = threading.Thread(target=_pump, daemon=True)
    pump_thread.start()
    try:
        returncode = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        pump_thread.join(timeout=5)
        raise
    pump_thread.join(timeout=5)
    return returncode, "".join(lines)


def apply_apollo_restoration(path: str, checkpoint: str = "restore") -> bool:
    """Run the isolated Apollo model to de-smear/restore codec-damaged audio."""
    if not path:
        return False
    source_path = Path(path).resolve()
    if not source_path.exists():
        logger.warning("Apollo restoration skipped; file not found: {}", source_path)
        return False
    python_exe = _venv_python(APOLLO_DIR / ".venv")
    if not python_exe.exists() or not APOLLO_SCRIPT.exists():
        logger.warning("Apollo restoration skipped; isolated env not found at {}", python_exe)
        return False

    temp_path = source_path.with_name(f"{source_path.stem}.apollo_tmp{source_path.suffix}")
    try:
        returncode, output = _run_subprocess_streaming(
            [
                str(python_exe),
                str(APOLLO_SCRIPT),
                "--input",
                str(source_path),
                "--output",
                str(temp_path),
                "--checkpoint",
                checkpoint,
            ],
            timeout=900,
        )
        if returncode != 0:
            logger.warning("Apollo restoration failed for {}: {}", source_path, output)
            return False
        temp_path.replace(source_path)
        logger.info("Applied Apollo restoration ({}): {}", checkpoint, source_path)
        return True
    except subprocess.TimeoutExpired:
        logger.warning("Apollo restoration timed out for {}", source_path)
        return False
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                logger.warning("Could not remove temporary Apollo file: {}", temp_path)


def apply_audiosr_upscale(
    path: str,
    model_name: str = "basic",
    ddim_steps: int = 25,
    guidance_scale: float = 3.5,
) -> bool:
    """Run the isolated AudioSR model to extend bandwidth to ~24 kHz (48 kHz output)."""
    if not path:
        return False
    source_path = Path(path).resolve()
    if not source_path.exists():
        logger.warning("AudioSR upscale skipped; file not found: {}", source_path)
        return False
    python_exe = _venv_python(AUDIOSR_DIR)
    if not python_exe.exists() or not AUDIOSR_SCRIPT.exists():
        logger.warning("AudioSR upscale skipped; isolated env not found at {}", python_exe)
        return False

    temp_path = source_path.with_name(f"{source_path.stem}.audiosr_tmp{source_path.suffix}")
    try:
        returncode, output = _run_subprocess_streaming(
            [
                str(python_exe),
                str(AUDIOSR_SCRIPT),
                "--input",
                str(source_path),
                "--output",
                str(temp_path),
                "--model-name",
                model_name,
                "--ddim-steps",
                str(ddim_steps),
                "--guidance-scale",
                str(guidance_scale),
            ],
            timeout=3600,
        )
        if returncode != 0:
            logger.warning("AudioSR upscale failed for {}: {}", source_path, output)
            return False
        temp_path.replace(source_path)
        logger.info("Applied AudioSR bandwidth extension: {}", source_path)
        return True
    except subprocess.TimeoutExpired:
        logger.warning("AudioSR upscale timed out for {}", source_path)
        return False
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                logger.warning("Could not remove temporary AudioSR file: {}", temp_path)


def get_audio_duration(path: Path) -> float | None:
    """Return an audio file's duration in seconds via ffprobe, or ``None`` on failure."""
    if shutil.which("ffprobe") is None:
        return None
    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            timeout=30,
            check=True,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return None
    try:
        return float(proc.stdout.decode("utf-8", errors="ignore").strip())
    except ValueError:
        return None


def find_tail_silence_start(path: Path, duration: float) -> float | None:
    """Return where a trailing silence run starts within the last TAIL_SILENCE_WINDOW_SECONDS.

    Returns ``None`` if no qualifying silence is found (nothing to trim). Uses fast
    input-side seeking since only the tail needs scanning, then re-anchors the reported
    (window-relative) timestamps back to absolute file time.
    """
    window_start = max(0.0, duration - TAIL_SILENCE_WINDOW_SECONDS)
    try:
        proc = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-nostats",
                "-ss",
                f"{window_start:.3f}",
                "-i",
                str(path),
                "-af",
                f"silencedetect=noise={TAIL_SILENCE_NOISE_DB}dB:d={TAIL_SILENCE_MIN_SECONDS}",
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            timeout=60,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    stderr = proc.stderr.decode("utf-8", errors="ignore")
    last_start = None
    for match in _SILENCE_START_RE.finditer(stderr):
        last_start = window_start + float(match.group(1))
    return last_start


def apply_fade_out(path: str, fade_seconds: float, mp3_bitrate: str = "320k") -> bool:
    """Trim any trailing silence-then-garbage artifact, then fade out cleanly before it.

    Normally called pre-mp3-conversion (source is still the raw generated WAV, so the
    PCM intermediate codec is correct). --fadeout repair mode calls this directly on
    files already on disk, which may already be .mp3 -- writing PCM into a ".mp3" temp
    file there would produce an invalid/oversized file, so pick the codec from the
    actual suffix rather than assuming WAV.
    """
    if not path or fade_seconds <= 0:
        return False
    source_path = Path(path)
    if not source_path.exists():
        logger.warning("Fade-out skipped; file not found: {}", source_path)
        return False
    if shutil.which("ffmpeg") is None:
        logger.warning("Fade-out skipped; ffmpeg is not available on PATH")
        return False

    duration = get_audio_duration(source_path)
    if duration is None or duration <= fade_seconds:
        logger.warning("Fade-out skipped; could not determine a usable duration for {}", source_path)
        return False

    trim_point = duration
    silence_start = find_tail_silence_start(source_path, duration)
    if silence_start is not None and silence_start > fade_seconds:
        trim_point = silence_start
        logger.info(
            "Trimming trailing silence/artifact: {:.2f}s -> {:.2f}s: {}",
            duration,
            trim_point,
            source_path,
        )

    is_mp3 = source_path.suffix.lower() == ".mp3"
    codec_args = (
        ["-codec:a", "libmp3lame", "-b:a", mp3_bitrate, "-ar", "48000"]
        if is_mp3
        else _INTERMEDIATE_CODEC_ARGS
    )
    fade_start = trim_point - fade_seconds
    fade_filter = f"afade=t=out:st={fade_start:.3f}:d={fade_seconds}:curve=qsin"
    temp_path = source_path.with_name(f"{source_path.stem}.fade_tmp{source_path.suffix}")
    try:
        _run_ffmpeg(
            [
                "-i",
                str(source_path),
                "-t",
                f"{trim_point:.3f}",
                "-af",
                fade_filter,
                *codec_args,
                str(temp_path),
            ]
        )
        temp_path.replace(source_path)
        logger.info("Applied {}s fade-out (final duration {:.2f}s): {}", fade_seconds, trim_point, source_path)
        return True
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="ignore") if exc.stderr else str(exc)
        logger.warning("Fade-out failed for {}: {}", source_path, stderr)
        return False
    except subprocess.TimeoutExpired:
        logger.warning("Fade-out timed out for {}", source_path)
        return False
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                logger.warning("Could not remove temporary fade file: {}", temp_path)


def convert_wav_to_mp3(wav_path: str, bitrate: str, keep_wav: bool = False) -> str:
    """Convert a WAV master to MP3 using ffmpeg and return the MP3 path."""
    source_path = Path(wav_path)
    if source_path.suffix.lower() != ".wav":
        return wav_path
    if not source_path.exists():
        logger.warning("MP3 conversion skipped; WAV file not found: {}", source_path)
        return wav_path
    if shutil.which("ffmpeg") is None:
        logger.warning("MP3 conversion skipped; ffmpeg is not available on PATH")
        return wav_path

    mp3_path = source_path.with_suffix(".mp3")
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source_path),
        "-codec:a",
        "libmp3lame",
        "-b:a",
        bitrate,
        "-ar",
        "48000",
        "-id3v2_version",
        "3",
        str(mp3_path),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, timeout=180)
        logger.info("Encoded MP3 from WAV master: {}", mp3_path)
        if not keep_wav:
            try:
                source_path.unlink()
                logger.info("Removed temporary WAV after MP3 export: {}", source_path)
            except OSError as exc:
                logger.warning("Could not remove temporary WAV {}: {}", source_path, exc)
        return str(mp3_path)
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="ignore") if exc.stderr else str(exc)
        logger.warning("MP3 conversion failed for {}: {}", source_path, stderr)
        return wav_path
    except subprocess.TimeoutExpired:
        logger.warning("MP3 conversion timed out for {}", source_path)
        return wav_path


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


def count_existing_tracks(save_dir: Path, genre: str) -> int:
    """Count audio files in save_dir already named for this genre (for --continue)."""
    prefix = slugify_title(genre) + "-"
    return sum(
        1
        for path in save_dir.iterdir()
        if path.is_file() and path.suffix.lower() in (".mp3", ".wav") and path.name.startswith(prefix)
    )


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
    cover_artist = resolve_cover_artist(args, genre)
    if cover_artist:
        logger.info(
            "Cover style reference: {} type beat (style-inspired only, not a literal cover)",
            cover_artist,
        )
    key_ingredients = resolve_key_ingredients(genre)
    if key_ingredients:
        logger.info("Forced key ingredients: {}", ", ".join(key_ingredients))
    while True:
        sample = create_genre_prompt(
            llm_handler, args, genre, track_index, duration, keyscale, cover_artist, key_ingredients
        )
        result = generate_music(
            dit_handler=dit_handler,
            llm_handler=llm_handler,
            params=build_generation_params(
                args, genre, sample, duration, keyscale, cover_artist, key_ingredients
            ),
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


# Large enough that per-candidate seeds never collide with adjacent tracks'
# seed_offset ranges (see build_generation_config_for_track).
CANDIDATE_SEED_STRIDE = 10_000


def generate_track_candidates(
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
    """Generate --candidates independent single-track renders and keep the best.

    Mastering can only polish whatever the diffusion model produced; seed-to-seed
    variance on these models is large enough that picking the best of several raw
    candidates is usually a bigger quality lever than the whole post-processing
    chain. Falls back to a single render at the caller's chunk_size (previous
    behavior, batching allowed) when candidates <= 1.
    """
    num_candidates = max(1, args.candidates)
    if num_candidates == 1:
        return generate_track_chunk(
            dit_handler, llm_handler, args, genre, track_index, chunk_size,
            initial_duration, keyscale, save_dir, seed_offset,
        )

    candidate_paths: list[Path] = []
    candidate_results = []
    result = None
    used_duration = initial_duration
    for candidate_index in range(num_candidates):
        logger.info(
            "Generating candidate {}/{} for track {}",
            candidate_index + 1,
            num_candidates,
            track_index + 1,
        )
        result, used_duration = generate_track_chunk(
            dit_handler,
            llm_handler,
            args,
            genre,
            track_index,
            1,
            initial_duration,
            keyscale,
            save_dir,
            seed_offset + candidate_index * CANDIDATE_SEED_STRIDE,
        )
        if not result.success or not result.audios:
            logger.warning(
                "Candidate {}/{} failed: {}", candidate_index + 1, num_candidates, result.status_message
            )
            continue
        path = Path(result.audios[0].get("path", ""))
        if path.exists():
            candidate_paths.append(path)
            candidate_results.append(result)

    if not candidate_paths:
        # Every candidate failed; return the last result so the caller's
        # existing error handling (and status message) kicks in as usual.
        return result, used_duration
    if len(candidate_paths) == 1:
        return candidate_results[0], used_duration

    reference = Path(args.candidate_reference) if args.candidate_reference else None
    best_path, best_metrics = score_track.pick_best(
        candidate_paths, reference, args.candidate_use_aesthetics
    )
    logger.info(
        "Best of {}: kept {} (score={:.4f}); discarding {} losing candidate(s)",
        num_candidates,
        best_path.name,
        best_metrics["score"],
        len(candidate_paths) - 1,
    )
    for path in candidate_paths:
        if path != best_path:
            try:
                path.unlink()
            except OSError:
                logger.warning("Could not remove losing candidate: {}", path)

    return candidate_results[candidate_paths.index(best_path)], used_duration


def apply_fadeout_repair(save_dir: Path, fade_seconds: float, mp3_bitrate: str) -> int:
    """Apply a fade-out to every existing audio file in save_dir; return failure count.

    Repair mode for tracks already on disk with an abrupt or noisy cutoff -- no model
    init or generation needed, just re-runs apply_fade_out() directly against files
    that already exist (unlike the normal pipeline, which only ever fades the raw WAV
    immediately after generation, before mp3 conversion).
    """
    paths = sorted(
        p for p in save_dir.iterdir()
        if p.is_file() and p.suffix.lower() in (".mp3", ".wav") and ".fade_tmp" not in p.name
    )
    logger.info("Fadeout repair: {} file(s) found in {}", len(paths), save_dir)
    failures = 0
    for path in paths:
        if not apply_fade_out(str(path), fade_seconds, mp3_bitrate):
            failures += 1
    logger.info("Fadeout repair done: {}/{} succeeded", len(paths) - failures, len(paths))
    return failures


def main() -> int:
    """Generate tracks and print resulting audio paths."""
    args = parse_args()
    if args.fadeout:
        save_dir = PROJECT_ROOT / args.output_dir
        if not save_dir.is_dir():
            logger.error("Fadeout repair: output directory not found: {}", save_dir)
            return 1
        failures = apply_fadeout_repair(save_dir, args.fade_out_seconds, args.mp3_bitrate)
        return 1 if failures else 0
    amount = resolve_amount(args)
    genres = resolve_genres(args)
    concurrency = resolve_concurrency(args)
    if args.candidates > 1 and concurrency != 1:
        logger.warning("--candidates > 1 requires one track per call; overriding --concurrency to 1")
        concurrency = 1
    initial_concurrency = concurrency
    save_dir = PROJECT_ROOT / args.output_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: {}", save_dir)
    logger.info("Genres: {}", ", ".join(genres))
    logger.info("Requested tracks per genre: {}", amount)
    logger.info("Tracks per generation call: {}", concurrency)
    logger.info(
        "Quality: {} model={} steps={} guidance={} offload={} offload_dit={}",
        args.quality,
        resolve_model(args),
        resolve_steps(args),
        resolve_guidance_scale(args),
        resolve_offload(args),
        resolve_offload_dit(args),
    )

    dit_handler = initialize_dit(args)
    llm_handler = initialize_lm(args)
    audios = []
    file_entries = []
    failed_chunks = []
    for genre_index, genre in enumerate(genres):
        logger.info("Starting genre: {}", genre)
        seed_offset = genre_index * amount
        track_index = 0
        if args.resume:
            track_index = min(count_existing_tracks(save_dir, genre), amount)
            if track_index > 0:
                logger.info(
                    "Resuming {}: {} track(s) already present, continuing at {}/{}",
                    genre,
                    track_index,
                    track_index,
                    amount,
                )
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
            generation_genre = resolve_spanish_substyle(genre)
            if generation_genre != genre:
                logger.info("Spanish sub-style: {}", generation_genre)
            logger.info("Chunk target duration: {}s", duration)
            logger.info("Chunk key: {}", keyscale)
            result, used_duration = generate_track_candidates(
                dit_handler=dit_handler,
                llm_handler=llm_handler,
                args=args,
                genre=generation_genre,
                track_index=track_index,
                chunk_size=chunk_size,
                initial_duration=duration,
                keyscale=keyscale,
                save_dir=save_dir,
                seed_offset=seed_offset,
            )

            if not result.success:
                logger.error(
                    "Skipping rest of {} (track {}/{}) after generation failure: {}",
                    genre,
                    track_index + 1,
                    amount,
                    result.status_message,
                )
                failed_chunks.append(
                    {"genre": genre, "track_index": track_index, "error": result.status_message}
                )
                break
            audios.extend(result.audios)
            saved_count = len(result.audios)
            for offset, audio in enumerate(result.audios):
                audio_path = audio.get("path", "")
                apollo_restored = False
                if args.enable_apollo_restoration:
                    apollo_restored = apply_apollo_restoration(audio_path, args.apollo_checkpoint)
                audiosr_upscaled = False
                if args.enable_audiosr_upscale:
                    audiosr_upscaled = apply_audiosr_upscale(
                        audio_path,
                        args.audiosr_model,
                        args.audiosr_ddim_steps,
                        args.audiosr_guidance_scale,
                    )
                clarity_mastered = False
                if not args.disable_clarity_mastering:
                    clarity_mastered = apply_tone_mastering(
                        audio_path, args.tone_profile, args.mono_bass_crossover_hz, args.stereo_width
                    )
                faded_out = False
                if not args.disable_fade_out:
                    faded_out = apply_fade_out(audio_path, args.fade_out_seconds, args.mp3_bitrate)
                # Loudness runs last so the integrated/true-peak measurement is of the
                # exact audio being delivered (tone-shaped and faded).
                lufs_normalized = False
                if not args.disable_lufs_normalization:
                    lufs_normalized = finalize_loudness(
                        audio_path, args.target_lufs, args.true_peak_db
                    )
                track_number = track_index + offset + 1
                renamed_path = rename_audio_file(audio_path, genre, track_number)
                final_path = (
                    convert_wav_to_mp3(renamed_path, args.mp3_bitrate)
                    if args.format == "mp3"
                    else renamed_path
                )
                audio["path"] = final_path
                file_entries.append(
                    {
                        "path": final_path,
                        "original_path": audio_path,
                        "wav_master_path": None,
                        "genre": genre,
                        "spanish_substyle": generation_genre if generation_genre != genre else None,
                        "target_duration": used_duration,
                        "key": keyscale,
                        "section_plan": build_section_plan(used_duration),
                        "seed": audio.get("params", {}).get("seed"),
                        "track_number": track_number,
                        "target_lufs": None if args.disable_lufs_normalization else args.target_lufs,
                        "true_peak_db": None if args.disable_lufs_normalization else args.true_peak_db,
                        "tone_profile": None if args.disable_clarity_mastering else args.tone_profile,
                        "apollo_restored": apollo_restored,
                        "audiosr_upscaled": audiosr_upscaled,
                        "clarity_mastered": clarity_mastered,
                        "lufs_normalized": lufs_normalized,
                        "fade_out_seconds": None if args.disable_fade_out else args.fade_out_seconds,
                        "faded_out": faded_out,
                    }
                )
            if saved_count < 1:
                logger.error(
                    "Skipping rest of {} (track {}/{}): generation returned no audio files",
                    genre,
                    track_index + 1,
                    amount,
                )
                failed_chunks.append(
                    {"genre": genre, "track_index": track_index, "error": "no audio files returned"}
                )
                break
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
        "apollo_restoration": args.enable_apollo_restoration,
        "audiosr_upscale": args.enable_audiosr_upscale,
        "candidates": args.candidates,
        "candidate_reference": args.candidate_reference,
        "candidate_use_aesthetics": args.candidate_use_aesthetics,
        "clarity_mastering": not args.disable_clarity_mastering,
        "quality": args.quality,
        "model": resolve_model(args),
        "steps": resolve_steps(args),
        "guidance_scale": resolve_guidance_scale(args),
        "offload": resolve_offload(args),
        "files": file_entries,
        "failed_chunks": failed_chunks,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    logger.info("Saved tracks: {}/{}", len(audios), amount * len(genres))
    logger.info("Manifest: {}", manifest_path)
    if failed_chunks:
        logger.warning(
            "{} genre(s) stopped early after a generation failure -- see failed_chunks in the manifest: {}",
            len(failed_chunks),
            ", ".join(f"{c['genre']} (track {c['track_index'] + 1})" for c in failed_chunks),
        )
    for audio in audios:
        print(audio.get("path", "(in-memory)"))
    return 1 if (failed_chunks and not audios) else 0


if __name__ == "__main__":
    sys.exit(main())
