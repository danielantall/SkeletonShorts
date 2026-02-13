"""
config.py — Centralised configuration for the Skeleton Shorts Pipeline.

All constants, paths, API settings, and style parameters live here.
Import this module from any service to keep values DRY.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ──────────────────────────────────────────────────────────────────────────────
# Load .env file (project root)
# ──────────────────────────────────────────────────────────────────────────────
load_dotenv()

# ──────────────────────────────────────────────────────────────────────────────
# API Keys  (read from environment — never hard-code)
# ──────────────────────────────────────────────────────────────────────────────
GOOGLE_API_KEY: str       = os.environ["GOOGLE_API_KEY"]
PIAPI_KEY: str            = os.environ["PIAPI_KEY"]
ELEVENLABS_API_KEY: str   = os.environ["ELEVENLABS_API_KEY"]
ELEVENLABS_VOICE_ID: str  = os.getenv("ELEVENLABS_VOICE_ID", "pNInz6obpgDQGcFmaJgB")

# ──────────────────────────────────────────────────────────────────────────────
# Directory Layout
# ──────────────────────────────────────────────────────────────────────────────
ROOT_DIR: Path    = Path(__file__).parent.resolve()
OUTPUT_DIR: Path  = ROOT_DIR / "output"
IMAGES_DIR: Path  = OUTPUT_DIR / "images"
VIDEOS_DIR: Path  = OUTPUT_DIR / "videos"
ASSETS_DIR: Path  = ROOT_DIR / "assets"

# Auto-create output folders on import
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# Output File Paths
# ──────────────────────────────────────────────────────────────────────────────
NARRATION_MP3_PATH: Path      = OUTPUT_DIR / "narration.mp3"
WORD_TIMESTAMPS_PATH: Path    = OUTPUT_DIR / "word_timestamps.json"
FINAL_VIDEO_PATH: Path        = OUTPUT_DIR / "final_short.mp4"

# ──────────────────────────────────────────────────────────────────────────────
# Video / Render Settings
# ──────────────────────────────────────────────────────────────────────────────
VIDEO_WIDTH: int   = 1080   # pixels
VIDEO_HEIGHT: int  = 1920   # pixels  (9:16 portrait)
VIDEO_FPS: int     = 30
VIDEO_CODEC: str   = "libx264"
AUDIO_CODEC: str   = "aac"

# ──────────────────────────────────────────────────────────────────────────────
# Caption / Typography Settings
# ──────────────────────────────────────────────────────────────────────────────
# Put LuckiestGuy-Regular.ttf (or any TTF) in ./assets/
# Download: https://fonts.google.com/specimen/Luckiest+Guy → Download family
CAPTION_FONT_PATH: Path    = ASSETS_DIR / "LuckiestGuy-Regular.ttf"
CAPTION_FONT_SIZE: int     = 100          # px — large, punchy, mobile-readable

# Fill & stroke
CAPTION_FILL_COLOR: str    = "white"      # text fill
CAPTION_STROKE_COLOR: str  = "black"      # outline colour
# Stroke scales with resolution: 8px at 720p → 12px at 1080p → 15px at 1440p
CAPTION_STROKE_WIDTH: int  = max(8, min(15, int(VIDEO_WIDTH * 0.011)))

# Drop shadow  (offset in px, opacity 0–255; 70% ≈ 178)
CAPTION_SHADOW_OFFSET: tuple[int, int] = (4, 4)   # (x, y) px
CAPTION_SHADOW_COLOR: tuple[int, int, int, int] = (0, 0, 0, 178)   # RGBA, 70% opacity

# Letter-spacing / tracking (-5 to -10 px tightening)
CAPTION_LETTER_SPACING: int = -8          # negative = tighter; 0 = default

# Vertical position of captions (0.0 = top, 1.0 = bottom)
CAPTION_Y_POSITION: float  = 0.72        # lower-centre — standard reels position

# ──────────────────────────────────────────────────────────────────────────────
# LLM Settings  (now using Gemini — same GOOGLE_API_KEY as image generation)
# ──────────────────────────────────────────────────────────────────────────────
# Text model used for script / voiceover generation
GEMINI_TEXT_MODEL: str         = "models/gemini-2.5-flash"   # confirmed available for this key
LLM_MAX_RETRIES: int           = 3
LLM_TEMPERATURE: float         = 0.8
# Target total video duration (s) the LLM should aim for
LLM_TARGET_DURATION_S: int     = 45

# ──────────────────────────────────────────────────────────────────────────────
# Image Generation Settings (Google GenAI)
# ──────────────────────────────────────────────────────────────────────────────
GEMINI_IMAGE_MODEL: str = "models/gemini-2.5-flash-image"
# Number of candidate images per prompt — we always pick the first
GEMINI_NUM_IMAGES: int  = 1

# ──────────────────────────────────────────────────────────────────────────────
# Motion Settings (Gemini Veo)
# ──────────────────────────────────────────────────────────────────────────────
GEMINI_VIDEO_MODEL: str      = "veo-3.1-fast-generate-preview"
POLLING_INTERVAL_S: int      = 10                      # seconds between status checks
POLLING_TIMEOUT_S: int       = 1200                    # 20 minutes max per task

# ──────────────────────────────────────────────────────────────────────────────
# Audio Settings (ElevenLabs)
# ──────────────────────────────────────────────────────────────────────────────
ELEVENLABS_MODEL: str    = "eleven_turbo_v2_5"
ELEVENLABS_BASE_URL: str = "https://api.elevenlabs.io"

# ──────────────────────────────────────────────────────────────────────────────
# Background Music
# ──────────────────────────────────────────────────────────────────────────────
BGM_PATH: Path | None    = None        # set via --bgm CLI flag; None = disabled
BGM_VOLUME_DB: float     = -16.0       # dB relative to original level
BGM_MAX_DURATION_S: float = 35.0       # auto-cut music at this point

# ──────────────────────────────────────────────────────────────────────────────
# Per-Scene Speed Map  (1-indexed scene number → speed multiplier)
# Scenes not in the map default to 1.0× (original speed).
# ──────────────────────────────────────────────────────────────────────────────
SCENE_SPEED_MAP: dict[int, float] = {
    1: 1.2,
    2: 1.1,
    3: 1.3,
    4: 1.2,
    5: 1.0,
    6: 1.0,
    7: 1.2,
    8: 1.2,
}
