"""
render_service.py — Module E: Final Video Composition with Dynamic Captions

Takes all the generated pieces and assembles them into the final short-form video:

  Inputs:
    - video_clips:       List of MP4 filepaths (from Module C)
    - narration_mp3:     Path to narration.mp3 (from Module D)
    - word_timestamps:   Path to word_timestamps.json (from Module D)

  Pipeline:
    1. Load each MP4 clip, resize to 1080×1920
    2. Concatenate all clips into one composite video
    3. Attach the narration audio track
    4. For each word in word_timestamps.json:
         a. Draw the word onto a transparent 1080×1920 PIL canvas using
            white fill + thick black stroke (YouTube-style caption look)
         b. Convert PIL image → numpy array → moviepy ImageClip
         c. Set .start and .duration from the timestamp data
    5. Composite all caption clips over the video
    6. Export final_short.mp4 at 1080×1920, 30 fps

  Output:
    ./output/final_short.mp4
"""

import json
import logging
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ──────────────────────────────────────────────────────────────────────────────
# Pillow 10+ compatibility fix for moviepy 1.0.3
# PIL.Image.ANTIALIAS was renamed to PIL.Image.LANCZOS in Pillow 10.
# moviepy 1.0.3 still references the old name — this one-liner restores it.
# ──────────────────────────────────────────────────────────────────────────────
if not hasattr(Image, "ANTIALIAS"):
    Image.ANTIALIAS = Image.LANCZOS  # type: ignore[attr-defined]

from moviepy.editor import (
    AudioFileClip,
    CompositeAudioClip,
    CompositeVideoClip,
    ImageClip, 
    VideoFileClip,
    concatenate_videoclips,
)
import moviepy.video.fx.all as vfx

import config

# ──────────────────────────────────────────────────────────────────────────────
# Logger
# ──────────────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def render_final_video(
    video_clip_paths: list[Path],
    narration_mp3_path: Path,
    word_timestamps_path: Path,
) -> Path:
    """
    Assemble the final short-form video from clips, audio, and captions.

    Args:
        video_clip_paths:    Ordered list of scene MP4 paths.
        narration_mp3_path:  Path to the narration audio file.
        word_timestamps_path: Path to the JSON word timestamp file.

    Returns:
        Path to the exported final_short.mp4.

    Raises:
        FileNotFoundError: If any input file is missing.
        RuntimeError: On moviepy or PIL rendering errors.
    """
    # ── Validate inputs ────────────────────────────────────────────────────────
    for p in video_clip_paths:
        if not Path(p).exists():
            raise FileNotFoundError(f"Video clip not found: {p}")
    if not narration_mp3_path.exists():
        raise FileNotFoundError(f"Narration MP3 not found: {narration_mp3_path}")
    if not word_timestamps_path.exists():
        raise FileNotFoundError(f"Word timestamps not found: {word_timestamps_path}")

    # ── Load font ──────────────────────────────────────────────────────────────
    font = _load_font()

    # ── Step 1 & 2: Load and concatenate video clips ───────────────────────────
    logger.info(f"[Render] Loading {len(video_clip_paths)} video clips...")
    clips = []
    for idx, path in enumerate(video_clip_paths):
        clip = VideoFileClip(str(path))
        # Resize to target resolution
        clip = clip.resize(newsize=(config.VIDEO_WIDTH, config.VIDEO_HEIGHT))

        # Apply per-scene speed multiplier (1-indexed lookup)
        scene_num = idx + 1
        speed = config.SCENE_SPEED_MAP.get(scene_num, 1.0)
        if speed != 1.0:
            clip = clip.fx(vfx.speedx, speed)
            logger.info(f"[Render] Scene {scene_num} sped up to {speed}× → {clip.duration:.2f}s")

        clips.append(clip)
        logger.debug(f"[Render] Loaded clip: {path} ({clip.duration:.2f}s)")

    logger.info("[Render] Concatenating clips...")
    base_video = concatenate_videoclips(clips, method="compose")
    total_duration = base_video.duration
    logger.info(f"[Render] Base video duration: {total_duration:.2f}s")

    # ── Step 3: Attach audio ───────────────────────────────────────────────────
    logger.info(f"[Render] Loading narration audio: {narration_mp3_path}")
    narration_audio = AudioFileClip(str(narration_mp3_path))

    # Trim audio to video length (or vice versa) to avoid length mismatches;
    # we trim whichever is longer.
    audio_duration = narration_audio.duration
    logger.info(f"[Render] Audio duration: {audio_duration:.2f}s")

    if audio_duration > total_duration:
        # Extend video by freezing the last frame of the last clip
        logger.warning(
            f"[Render] Audio ({audio_duration:.2f}s) is longer than video "
            f"({total_duration:.2f}s). Extending video with freeze frame."
        )
        freeze = clips[-1].to_ImageClip(t=clips[-1].duration - 0.1)
        freeze = freeze.set_duration(audio_duration - total_duration)
        base_video = concatenate_videoclips([base_video, freeze], method="compose")
        total_duration = audio_duration
    else:
        # Trim narration to video length
        narration_audio = narration_audio.subclip(0, total_duration)

    # ── Mix in background music (if configured) ─────────────────────────────
    bgm_clip = None
    if config.BGM_PATH and Path(config.BGM_PATH).exists():
        logger.info(f"[Render] Loading background music: {config.BGM_PATH}")
        bgm_clip = AudioFileClip(str(config.BGM_PATH))

        # Trim to the shorter of BGM_MAX_DURATION_S and total video duration
        bgm_duration = min(config.BGM_MAX_DURATION_S, total_duration, bgm_clip.duration)
        bgm_clip = bgm_clip.subclip(0, bgm_duration)

        # Convert dB to linear multiplier:  -16 dB → 10^(-16/20) ≈ 0.158
        linear_vol = 10 ** (config.BGM_VOLUME_DB / 20)
        bgm_clip = bgm_clip.volumex(linear_vol)
        logger.info(
            f"[Render] BGM: {bgm_duration:.1f}s at {config.BGM_VOLUME_DB} dB "
            f"(×{linear_vol:.3f})"
        )

        # Mix narration + BGM into a composite audio track
        narration_audio = CompositeAudioClip([narration_audio, bgm_clip])
    else:
        if config.BGM_PATH:
            logger.warning(f"[Render] BGM file not found: {config.BGM_PATH} — skipping")

    base_video = base_video.set_audio(narration_audio)

    # ── Step 4: Build dynamic caption clips ───────────────────────────────────
    logger.info(f"[Render] Loading word timestamps: {word_timestamps_path}")
    with word_timestamps_path.open("r", encoding="utf-8") as f:
        word_timestamps: list[dict] = json.load(f)

    logger.info(f"[Render] Building {len(word_timestamps)} caption clips...")
    caption_clips: list[ImageClip] = []

    for entry in word_timestamps:
        word:  str   = entry.get("word", "")
        start: float = float(entry.get("start", 0.0))
        end:   float = float(entry.get("end", start + 0.3))

        if not word.strip():
            continue

        duration = max(end - start, 0.05)  # minimum 50ms to avoid moviepy errors

        # Skip captions that would appear beyond the video duration
        if start >= total_duration:
            continue

        # Create the PIL caption image
        caption_img_array = _render_caption_frame(word, font)

        # Build moviepy ImageClip from numpy array
        caption_clip = (
            ImageClip(caption_img_array)
            .set_start(start)
            .set_duration(duration)
            # Center the caption horizontally; place at configured vertical position
            .set_position(("center", config.CAPTION_Y_POSITION), relative=True)
        )
        caption_clips.append(caption_clip)

    # ── Step 5: Composite and export ───────────────────────────────────────────
    logger.info("[Render] Compositing video + captions...")
    all_clips = [base_video] + caption_clips
    final = CompositeVideoClip(all_clips, size=(config.VIDEO_WIDTH, config.VIDEO_HEIGHT))
    final = final.set_duration(total_duration)

    logger.info(f"[Render] Exporting → {config.FINAL_VIDEO_PATH}")
    final.write_videofile(
        str(config.FINAL_VIDEO_PATH),
        fps=config.VIDEO_FPS,
        codec=config.VIDEO_CODEC,
        audio_codec=config.AUDIO_CODEC,
        # ffmpeg preset — 'fast' balances quality and encode speed
        ffmpeg_params=["-preset", "fast"],
        logger="bar",  # moviepy progress bar
    )

    logger.info(f"[Render] ✅ Export complete: {config.FINAL_VIDEO_PATH}")

    # Clean up moviepy clips to release file handles
    for clip in clips:
        clip.close()
    narration_audio.close()
    if bgm_clip:
        bgm_clip.close()
    final.close()

    return config.FINAL_VIDEO_PATH


# ──────────────────────────────────────────────────────────────────────────────
# Caption rendering helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_font() -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """
    Load the caption font from config. Falls back to PIL's built-in bitmap font
    if the configured TTF is not found (so the pipeline never hard-crashes on font).
    """
    font_path = config.CAPTION_FONT_PATH
    if font_path.exists():
        logger.info(f"[Render] Using font: {font_path.name}")
        return ImageFont.truetype(str(font_path), size=config.CAPTION_FONT_SIZE)
    else:
        logger.warning(
            f"[Render] Font not found at {font_path}. "
            "Falling back to default PIL font. "
            "Download LuckiestGuy-Regular.ttf from fonts.google.com and place it in ./assets/"
        )
        return ImageFont.load_default()


def _render_caption_frame(
    word: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> np.ndarray:
    """
    Render a single word as a styled caption on a transparent 1080×1920 RGBA canvas.

    Style:
      - Drop shadow (offset + 70% opacity) drawn first
      - Thick black stroke (8-direction offset technique)
      - White text fill on top
      - Letter-spacing (tracking) applied by drawing characters individually

    Returns:
        A numpy uint8 array of shape (1920, 1080, 4) — RGBA.
    """
    canvas = Image.new("RGBA", (config.VIDEO_WIDTH, config.VIDEO_HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    # ── Measure total width with tracking applied ──────────────────────────────
    tracking = config.CAPTION_LETTER_SPACING  # negative = tighter

    # Get per-character advance widths and total width
    char_widths: list[int] = []
    for ch in word:
        bbox = draw.textbbox((0, 0), ch, font=font)
        char_widths.append(bbox[2] - bbox[0])

    # Total rendered width = sum of char widths + tracking between each gap
    total_width = sum(char_widths) + tracking * max(0, len(word) - 1)

    # Height from a full-word bbox (more reliable than single char for ascenders)
    full_bbox   = draw.textbbox((0, 0), word, font=font)
    text_height = full_bbox[3] - full_bbox[1]

    # ── Compute top-left origin ────────────────────────────────────────────────
    x_start = (config.VIDEO_WIDTH - total_width) // 2
    y       = int(config.VIDEO_HEIGHT * config.CAPTION_Y_POSITION) - text_height // 2

    # ── Helper: draw word char-by-char with tracking ──────────────────────────
    def _draw_tracked(d: "ImageDraw.ImageDraw", x0: int, y0: int,
                      fill: str | tuple, stroke_w: int = 0, stroke_fill=None) -> None:
        cx = x0
        for i, (ch, cw) in enumerate(zip(word, char_widths)):
            if stroke_w and stroke_fill:
                # Draw stroke offsets for this character
                for ox in range(-stroke_w, stroke_w + 1):
                    for oy in range(-stroke_w, stroke_w + 1):
                        if ox == 0 and oy == 0:
                            continue
                        d.text((cx + ox, y0 + oy), ch, font=font, fill=stroke_fill)
            d.text((cx, y0), ch, font=font, fill=fill)
            cx += cw + tracking

    # ── 1. Drop shadow layer ──────────────────────────────────────────────────
    sx, sy = config.CAPTION_SHADOW_OFFSET
    shadow_color = config.CAPTION_SHADOW_COLOR   # RGBA with 70% opacity

    shadow_layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    shadow_draw  = ImageDraw.Draw(shadow_layer)
    _draw_tracked(shadow_draw, x_start + sx, y + sy, fill=shadow_color)

    # Compose shadow below text
    canvas = Image.alpha_composite(canvas, shadow_layer)
    draw   = ImageDraw.Draw(canvas)   # re-bind draw to composited canvas

    # ── 2. Stroke + fill layer ────────────────────────────────────────────────
    stroke = config.CAPTION_STROKE_WIDTH

    # Draw stroke offsets first (black outline)
    _draw_tracked(draw, x_start, y,
                  fill=config.CAPTION_STROKE_COLOR,
                  stroke_w=stroke, stroke_fill=config.CAPTION_STROKE_COLOR)

    # Draw white fill on top
    _draw_tracked(draw, x_start, y, fill=config.CAPTION_FILL_COLOR)

    return np.array(canvas)
