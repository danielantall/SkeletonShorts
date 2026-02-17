"""
md_parser.py — Markdown Scene File Parser

Parses a structured Markdown file that defines the scenario and image prompts
for each scene. This lets you hand-craft precise image generation prompts
instead of relying on GPT-4o to invent them.

──────────────────────────────────────────────────────────────────────────────
EXPECTED FILE FORMAT
──────────────────────────────────────────────────────────────────────────────

The file can be structured in two ways — both are supported.

Format 1 — Scenario on the first line (optional):

    What if you ate only sugar for 30 days?

    Scene: Skeleton Surrounded by Mountains of Sugar
    Place the skeleton character sitting at a kitchen table...

    🎬 B-Roll Prompt 1: Energy Rush ⚡
    Place the skeleton character vibrating with hyper energy...

    🎬 B-Roll Prompt 2: Sugar Crash 😵
    Place the skeleton character slumped face-down...

Format 2 — Just scenes (no scenario header):

    ## Scene 1
    Place the skeleton character...

    ## Scene 2
    Place the skeleton character...

──────────────────────────────────────────────────────────────────────────────
Scene header detection (any of these patterns triggers a new scene):
  - "Scene:"              e.g.  Scene: Skeleton at a Table
  - "🎬"                  e.g.  🎬 B-Roll Prompt 1: Energy Rush
  - "## "  or  "### "     Markdown h2/h3 headings
  - "B-Roll"              e.g.  B-Roll Prompt 2:
  - "Prompt N:"           e.g.  Prompt 3: Sugar Crash

Lines that are blank or only contain the header text are skipped.
All remaining paragraph text under a header becomes the scene prompt.
──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import re
import logging
from pathlib import Path

import config

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Regex patterns that signal "this line starts a new scene"
# ──────────────────────────────────────────────────────────────────────────────
_SCENE_HEADER_PATTERNS: list[re.Pattern] = [
    re.compile(r"^scene\s*\d*\s*:", re.IGNORECASE),     # "Scene:" or "Scene 1:"
    re.compile(r"^🎬"),                                 # "🎬 B-Roll Prompt ..."
    re.compile(r"^#{1,3}\s"),                          # "## " or "### " headings
    re.compile(r"^b-roll\s", re.IGNORECASE),           # "B-Roll Prompt ..."
    re.compile(r"^prompt\s+\d+\s*:", re.IGNORECASE),  # "Prompt 3: ..."
]

# Pattern to detect a scenario / question line (first non-empty line that looks
# like a "What if ...?" sentence — not a header)
_SCENARIO_PATTERN = re.compile(r"what\s+if", re.IGNORECASE)

# Default duration per scene if not explicitly specified (seconds)
_DEFAULT_DURATION: float = 5.0


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def parse_scenes_md(filepath: str | Path) -> tuple[str | None, list[dict]]:
    """
    Parse a Markdown scene file and return (scenario, scenes).

    Args:
        filepath: Path to the .md scene file.

    Returns:
        A tuple of:
          - scenario (str | None): The "What if" question if found in the file,
            otherwise None (caller falls back to the CLI --scenario argument).
          - scenes (list[dict]): List of scene dicts:
              [{"prompt": str, "duration": float}, ...]

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If no scene prompts could be parsed from the file.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Scene MD file not found: {filepath}")

    raw_text = filepath.read_text(encoding="utf-8")
    lines    = raw_text.splitlines()

    logger.info(f"[MDParser] Parsing scene file: {filepath} ({len(lines)} lines)")

    # ── Pass 1: detect optional scenario line ─────────────────────────────────
    scenario: str | None = None
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        # If the very first substantial line looks like a "What if" question
        # and is NOT a scene header, treat it as the scenario.
        if _SCENARIO_PATTERN.search(stripped) and not _is_scene_header(stripped):
            # Clean surrounding quotes and whitespace
            scenario = stripped.strip('"\'').strip()
            logger.info(f"[MDParser] Found scenario in file: {scenario!r}")
            break
        # Stop searching at the first scene header — scenario must come before scenes
        if _is_scene_header(stripped):
            break

    # ── Pass 2: split into scenes ─────────────────────────────────────────────
    scenes: list[dict] = _extract_scenes(lines)

    if not scenes:
        raise ValueError(
            f"No scene prompts found in {filepath}. "
            "Make sure scenes are separated by headers like 'Scene:', '##', or '🎬'."
        )

    logger.info(f"[MDParser] Parsed {len(scenes)} scenes.")
    for i, s in enumerate(scenes):
        logger.debug(f"[MDParser] Scene {i}: {s['prompt'][:80]}...")

    return scenario, scenes


# ──────────────────────────────────────────────────────────────────────────────
# Private helpers
# ──────────────────────────────────────────────────────────────────────────────

def _is_scene_header(line: str) -> bool:
    """Return True if the line matches any of the scene-header patterns."""
    return any(pat.match(line.strip()) for pat in _SCENE_HEADER_PATTERNS)


def _extract_scenes(lines: list[str]) -> list[dict]:
    """
    Walk through all lines and split them into scene blocks.

    Each scene block starts at a header line.  All subsequent non-header,
    non-empty lines are the prompt body until the next header (or EOF).

    A duration can optionally be embedded in the header with syntax like:
      "Scene: Sugar Rush [7s]" → duration = 7.0
    Otherwise _DEFAULT_DURATION is used.
    """
    scenes: list[dict] = []
    current_header: str = ""
    current_body_lines: list[str] = []

    def _flush():
        """Save the accumulated scene body as a scene dict."""
        nonlocal current_header, current_body_lines
        if not current_body_lines:
            return

        prompt = " ".join(current_body_lines).strip()
        if not prompt:
            return

        # Try to extract a duration tag from the header: [5s] or [5.0s]
        duration = _parse_duration_from_header(current_header)

        scenes.append({"prompt": prompt, "duration": duration})
        current_header = ""
        current_body_lines = []

    for line in lines:
        stripped = line.strip()

        if _is_scene_header(stripped):
            # Save whatever we were building
            _flush()
            current_header = stripped
            # Don't include the header text itself in the prompt body

        elif current_header:
            # We're inside a scene block
            if stripped:  # skip blank lines within body
                # Skip lines that just restate the header (emoji-only, etc.)
                if stripped not in ("🎬", "##", "###"):
                    current_body_lines.append(stripped)
        # else: lines before the first header (e.g. scenario line) — ignored here

    # Flush the last scene
    _flush()

    return scenes


def _parse_duration_from_header(header: str) -> float:
    """
    Look for an optional duration tag in the header, e.g. "[5s]" or "[7.5s]".
    Returns _DEFAULT_DURATION if none is found.
    """
    match = re.search(r"\[(\d+(?:\.\d+)?)\s*s\]", header, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return _DEFAULT_DURATION
