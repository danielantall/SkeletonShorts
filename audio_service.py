"""
audio_service.py — Module D: Narration + Word Timestamps via ElevenLabs

Takes the full voiceover string from Module A and calls the ElevenLabs
Text-to-Speech with Timestamps endpoint. Produces two artefacts:

  1. narration.mp3  — the rendered audio file
  2. word_timestamps.json — word-level timing data used by Module E for captions

ElevenLabs /with-timestamps returns a JSON body containing:
  - audio_base64:  The MP3 audio encoded as base64
  - alignment:     Character/word-level timing data

We parse the character-level alignment into clean word-level entries:
  [{"word": "sugar", "start": 1.23, "end": 1.61}, ...]
"""

import base64
import json
import logging
from pathlib import Path

import requests

import config

# ──────────────────────────────────────────────────────────────────────────────
# Logger
# ──────────────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# ElevenLabs API endpoint
# ──────────────────────────────────────────────────────────────────────────────
_TTS_WITH_TIMESTAMPS_URL = (
    "{base}/v1/text-to-speech/{voice_id}/with-timestamps"
)

# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def generate_audio(voiceover: str) -> tuple[Path, Path]:
    """
    Send the voiceover text to ElevenLabs and get back audio + word timestamps.

    Args:
        voiceover: The full narration string from Module A.

    Returns:
        A tuple of (narration_mp3_path, word_timestamps_json_path).

    Raises:
        requests.HTTPError: On API error.
        ValueError: If the response body is missing expected fields.
    """
    url = _TTS_WITH_TIMESTAMPS_URL.format(
        base=config.ELEVENLABS_BASE_URL,
        voice_id=config.ELEVENLABS_VOICE_ID,
    )

    headers = {
        "xi-api-key": config.ELEVENLABS_API_KEY,
        "Content-Type": "application/json",
    }

    payload = {
        "text": voiceover,
        "model_id": config.ELEVENLABS_MODEL,
        # Voice settings — tuned for fast, punchy short-form delivery
        "voice_settings": {
            "stability": 0.40,          # higher = more consistent pacing, fewer random pauses
            "similarity_boost": 0.95,   # high fidelity to the selected voice
            "style": 0.00,              # lower = less dramatic pauses, quicker pace
            "use_speaker_boost": True,
            "speed": 1.2,              # slightly faster than natural (1.0 = normal)
        },
    }

    logger.info(
        f"[Audio] Requesting TTS from ElevenLabs "
        f"(voice={config.ELEVENLABS_VOICE_ID}, model={config.ELEVENLABS_MODEL})..."
    )
    logger.debug(f"[Audio] Voiceover ({len(voiceover)} chars): {voiceover[:120]}...")

    response = requests.post(url, json=payload, headers=headers, timeout=120)
    response.raise_for_status()

    data: dict = response.json()

    # ── 1. Decode and save the MP3 ─────────────────────────────────────────────
    audio_b64: str | None = data.get("audio_base64")
    if not audio_b64:
        raise ValueError(
            "ElevenLabs response is missing 'audio_base64'. "
            f"Keys received: {list(data.keys())}"
        )

    audio_bytes = base64.b64decode(audio_b64)
    config.NARRATION_MP3_PATH.write_bytes(audio_bytes)
    logger.info(
        f"[Audio] Saved narration.mp3 "
        f"({len(audio_bytes) / 1024:.1f} KB) → {config.NARRATION_MP3_PATH}"
    )

    # ── 2. Parse word timestamps ───────────────────────────────────────────────
    alignment: dict | None = data.get("alignment")
    if not alignment:
        raise ValueError(
            "ElevenLabs response is missing 'alignment'. "
            "Ensure the endpoint is /with-timestamps."
        )

    word_timestamps = _parse_word_timestamps(alignment)

    config.WORD_TIMESTAMPS_PATH.write_text(
        json.dumps(word_timestamps, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(
        f"[Audio] Saved word_timestamps.json "
        f"({len(word_timestamps)} words) → {config.WORD_TIMESTAMPS_PATH}"
    )

    return config.NARRATION_MP3_PATH, config.WORD_TIMESTAMPS_PATH


# ──────────────────────────────────────────────────────────────────────────────
# Timestamp parsing helpers
# ──────────────────────────────────────────────────────────────────────────────

def _parse_word_timestamps(alignment: dict) -> list[dict]:
    """
    Convert ElevenLabs character-level alignment data into clean word entries.

    ElevenLabs /with-timestamps returns character-level data:
      {
        "characters": ["H","e","l","l","o"," ","w","o","r","l","d"],
        "character_start_times_seconds": [0.0, 0.1, 0.15, ...],
        "character_end_times_seconds":   [0.1, 0.15, 0.2, ...]
      }

    We group consecutive non-space characters into words, taking the start time
    of the first character and the end time of the last character in each group.

    Returns:
        [{"word": str, "start": float, "end": float}, ...]
    """
    characters: list[str] = alignment.get("characters", [])
    starts: list[float]   = alignment.get("character_start_times_seconds", [])
    ends: list[float]     = alignment.get("character_end_times_seconds", [])

    if not characters or not starts or not ends:
        logger.warning("[Audio] Alignment data is empty or incomplete.")
        return []

    if len(characters) != len(starts) or len(characters) != len(ends):
        logger.warning(
            f"[Audio] Alignment arrays have mismatched lengths: "
            f"chars={len(characters)}, starts={len(starts)}, ends={len(ends)}"
        )

    word_entries: list[dict] = []
    current_word_chars: list[str] = []
    current_word_start: float = 0.0
    current_word_end: float   = 0.0

    for char, start, end in zip(characters, starts, ends):
        if char == " " or char == "\n":
            # Flush the current word
            if current_word_chars:
                word_entries.append({
                    "word":  "".join(current_word_chars).strip(),
                    "start": round(current_word_start, 4),
                    "end":   round(current_word_end, 4),
                })
                current_word_chars = []
        else:
            if not current_word_chars:
                # First character of a new word
                current_word_start = start
            current_word_chars.append(char)
            current_word_end = end

    # Flush the final word (no trailing space)
    if current_word_chars:
        word_entries.append({
            "word":  "".join(current_word_chars).strip(),
            "start": round(current_word_start, 4),
            "end":   round(current_word_end, 4),
        })

    # Filter out any empty strings (punctuation-only tokens etc.)
    word_entries = [w for w in word_entries if w["word"]]

    logger.debug(f"[Audio] Parsed {len(word_entries)} word timestamps.")
    return word_entries
