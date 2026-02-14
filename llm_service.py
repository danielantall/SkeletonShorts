"""
llm_service.py — Module A: Script Generation via Google Gemini

Uses the same google-genai SDK already powering the image generation step,
so no additional API key is needed — just GOOGLE_API_KEY.

Two modes of operation:

  1. FULL MODE (default):
     generate_script(scenario) → ScriptJSON
     Gemini writes both the voiceover AND the scene image prompts.

  2. VOICEOVER-ONLY MODE (used when --scenes_md is supplied):
     generate_voiceover_only(scenario, scenes) → str
     Scene prompts come from a Markdown file.
     Gemini writes only the narration, informed by the pre-defined scenes.
"""

import json
import logging
import time
from typing import TypedDict

from google import genai
from google.genai import types

import config

# ──────────────────────────────────────────────────────────────────────────────
# Logger
# ──────────────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Type definitions
# ──────────────────────────────────────────────────────────────────────────────
class SceneDict(TypedDict):
    prompt: str      # Gemini image-generation prompt
    duration: float  # Target clip length in seconds

class ScriptJSON(TypedDict):
    voiceover: str
    scenes: list[SceneDict]

# ──────────────────────────────────────────────────────────────────────────────
# System prompts
# ──────────────────────────────────────────────────────────────────────────────
_SCRIPT_SYSTEM_PROMPT = """\
# ROLE
You are a Viral Short-Form Retention Engineer specializing in "Skeleton-style" high-velocity scripts. Your goal is to transform a "What if" topic and a list of scenes into a 25-35 second script designed for 100%+ retention.

# THE RULES OF THE SKELETON METHOD
1. THE HOOK (0-2s): Start with a high-status contradiction or a shocking "What if" claim. No intros. No "So..." No breathing. 1.3 seconds to stop the thumb.
2. THE SMART TRIGGER: Every sentence must make the viewer feel like they are discovering a hidden truth or a complex simulation. Use "intellectual" escalation.
3. COMPRESSION: Short, aggressive sentences. Remove all filler words (basically, so, actually, um). Cut the "breaths" between ideas.
4. THE ESCALATION CURVE:
   - 0-2s: Hook (The Shock)
   - 3-8s: Expand Tension (The Stakes)
   - 9-18s: Build Complexity (The Logic)
   - 19-25s: The "Aha" Moment (The Insight)
   - Final 3s: The Abrupt Cut.
5. NO OUTROS: No "follow for more," no "comment below."
6. THE LOOP: End the script with a sentence that makes the viewer want to rewatch the video.

# OUTPUT TASK
Write a raw block of script text. Use line breaks to indicate rapid-fire pacing. The script must be under {duration} seconds when spoken at a fast, urgent pace.
Output ONLY the raw voiceover text — no JSON, no scene labels, no formatting.
""".strip()

_VOICEOVER_SYSTEM_PROMPT = """\
# ROLE
You are a Viral Short-Form Retention Engineer specializing in "Skeleton-style" high-velocity scripts. Your goal is to transform a "What if" topic and a list of scenes into a 25-35 second script designed for 100%+ retention.

# THE RULES OF THE SKELETON METHOD
1. THE HOOK (0-2s): Start with a high-status contradiction or a shocking "What if" claim. No intros. No "So..." No breathing. 1.3 seconds to stop the thumb.
2. THE SMART TRIGGER: Every sentence must make the viewer feel like they are discovering a hidden truth or a complex simulation. Use "intellectual" escalation.
3. COMPRESSION: Short, aggressive sentences. Remove all filler words (basically, so, actually, um). Cut the "breaths" between ideas.
4. THE ESCALATION CURVE:
   - 0-2s: Hook (The Shock)
   - 3-8s: Expand Tension (The Stakes)
   - 9-18s: Build Complexity (The Logic)
   - 19-25s: The "Aha" Moment (The Insight)
   - Final 3s: The Abrupt Cut.
5. NO OUTROS: No "follow for more," no "comment below."
6. THE LOOP: End the script with a sentence that makes the viewer want to rewatch the video.
7. Ensure to follow the chronological timeline of the scenes provided.

# OUTPUT TASK
Write a raw block of script text. Use line breaks to indicate rapid-fire pacing. The script must be spoken at a fast, urgent pace.
Output ONLY the raw voiceover text — no JSON, no scene labels, no formatting.
""".strip()


# ──────────────────────────────────────────────────────────────────────────────
# Shared Gemini client (initialised once)
# ──────────────────────────────────────────────────────────────────────────────
def _get_client() -> genai.Client:
    return genai.Client(api_key=config.GOOGLE_API_KEY)


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def generate_script(scenario: str) -> ScriptJSON:
    """
    Generate a full video script (voiceover + scenes) using Gemini.

    Args:
        scenario: The "What if" question string.

    Returns:
        ScriptJSON dict with 'voiceover' and 'scenes'.

    Raises:
        ValueError: On malformed JSON or missing keys after all retries.
        Exception: On unrecoverable API errors.
    """
    client = _get_client()

    system_prompt = _SCRIPT_SYSTEM_PROMPT.format(duration=config.LLM_TARGET_DURATION_S)

    user_message = (
        f'Create a "What If" short-form video script for this scenario:\n\n'
        f'"{scenario}"\n\n'
        f'Return ONLY the JSON object, nothing else.'
    )

    last_error: Exception | None = None

    for attempt in range(1, config.LLM_MAX_RETRIES + 1):
        logger.info(f"[LLM] Attempt {attempt}/{config.LLM_MAX_RETRIES} — calling Gemini ({config.GEMINI_TEXT_MODEL})...")
        try:
            response = client.models.generate_content(
                model=config.GEMINI_TEXT_MODEL,
                contents=user_message,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    # JSON mode — model MUST return a valid JSON object
                    response_mime_type="application/json",
                    temperature=config.LLM_TEMPERATURE,
                ),
            )

            raw_content: str = response.text or ""
            logger.debug(f"[LLM] Raw response:\n{raw_content[:500]}...")

            script = json.loads(raw_content)
            _validate_script(script)

            logger.info(
                f"[LLM] Script generated: {len(script['scenes'])} scenes, "
                f"{sum(s['duration'] for s in script['scenes']):.1f}s total"
            )
            return script  # type: ignore[return-value]

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"[LLM] Attempt {attempt} failed — bad response: {e}")
            last_error = e
            if attempt < config.LLM_MAX_RETRIES:
                time.sleep(2 ** attempt)
        except Exception as e:
            logger.error(f"[LLM] Gemini API error on attempt {attempt}: {e}")
            last_error = e
            if attempt < config.LLM_MAX_RETRIES:
                time.sleep(3)

    raise ValueError(
        f"Failed to generate a valid script after {config.LLM_MAX_RETRIES} attempts. "
        f"Last error: {last_error}"
    )


def generate_voiceover_only(scenario: str, scenes: list[dict]) -> str:
    """
    Generate ONLY the narration voiceover when scenes are pre-defined
    (loaded from a Markdown file via md_parser).

    Args:
        scenario: The "What if" question string.
        scenes:   List of scene dicts [{"prompt": str, "duration": float}, ...]

    Returns:
        The voiceover narration as a plain string.

    Raises:
        ValueError: After all retries fail.
    """
    client = _get_client()

    total_duration = sum(float(s.get("duration", 5.0)) for s in scenes)

    system_prompt = _VOICEOVER_SYSTEM_PROMPT.format(duration=int(total_duration))

    scene_summary = "\n".join(
        f"  Scene {i+1} ({s.get('duration', 8.0):.0f}s): {s['prompt']}"
        for i, s in enumerate(scenes)
    )

    user_message = (
        f'# INPUT DATA\n'
        f'Topic (Crucial Context): "{scenario}"\n\n'
        f'Visual Scenes to Narrate:\n{scene_summary}\n\n'
        f'[START SCRIPT]'
    )

    last_error: Exception | None = None

    for attempt in range(1, config.LLM_MAX_RETRIES + 1):
        logger.info(f"[LLM] Voiceover-only attempt {attempt}/{config.LLM_MAX_RETRIES}...")
        try:
            response = client.models.generate_content(
                model=config.GEMINI_TEXT_MODEL,
                contents=user_message,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    # Plain text — no JSON mode here
                    temperature=config.LLM_TEMPERATURE,
                ),
            )

            voiceover: str = (response.text or "").strip()

            if not voiceover:
                raise ValueError("Gemini returned an empty voiceover.")

            logger.info(
                f"[LLM] Voiceover generated: {len(voiceover)} chars, "
                f"~{len(voiceover.split())} words"
            )
            return voiceover

        except Exception as e:
            logger.error(f"[LLM] Error on attempt {attempt}: {e}")
            last_error = e
            if attempt < config.LLM_MAX_RETRIES:
                time.sleep(2 ** attempt)

    raise ValueError(
        f"Failed to generate voiceover after {config.LLM_MAX_RETRIES} attempts. "
        f"Last error: {last_error}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Private helpers
# ──────────────────────────────────────────────────────────────────────────────

def _validate_script(script: dict) -> None:
    """Validate the parsed JSON script has the expected shape."""
    if "voiceover" not in script:
        raise ValueError("Script JSON is missing 'voiceover' key")
    if "scenes" not in script:
        raise ValueError("Script JSON is missing 'scenes' key")
    if not isinstance(script["scenes"], list) or len(script["scenes"]) == 0:
        raise ValueError("'scenes' must be a non-empty list")
    for i, scene in enumerate(script["scenes"]):
        if "prompt" not in scene:
            raise ValueError(f"Scene {i} is missing 'prompt' key")
        if "duration" not in scene:
            raise ValueError(f"Scene {i} is missing 'duration' key")
        try:
            scene["duration"] = float(scene["duration"])
        except (TypeError, ValueError):
            raise ValueError(f"Scene {i} 'duration' is not a valid number: {scene['duration']!r}")
        if scene["duration"] <= 0:
            raise ValueError(f"Scene {i} has non-positive duration: {scene['duration']}")
