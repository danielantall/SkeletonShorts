"""
main.py — Skeleton Shorts Pipeline Orchestrator

CLI entrypoint. Wires together all five service modules into a single pipeline:

  A → B → [C ∥ D] → E

  A: llm_service    — GPT-4o generates script (voiceover + scenes)
  B: vision_service — Gemini generates one PNG per scene
  C: motion_service — Kling AI animates each PNG into an MP4   ┐  concurrent
  D: audio_service  — ElevenLabs renders voiceover + timestamps ┘
  E: render_service — moviepy composites final video with captions

Usage:
  python main.py --scenario "What if you ate nothing but sugar for 30 days?" \\
                 --character_img ./assets/skeleton_base.png

Dry-run (no API calls; validates render pipeline only):
  python main.py --scenario "..." --character_img ./assets/skeleton_base.png --dry-run
"""

import argparse
import asyncio
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Logging configuration — set up BEFORE importing service modules so that their
# module-level loggers inherit this config.
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        # Also write to a log file for debugging
        logging.FileHandler("pipeline.log", mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger("main")

# ──────────────────────────────────────────────────────────────────────────────
# Import service modules (after logging is configured)
# ──────────────────────────────────────────────────────────────────────────────
import config  # loads .env on import
import llm_service
import vision_service
import motion_service
import audio_service
import render_service
import md_parser   # Markdown scene file parser


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="skeleton-shorts",
        description="Generate a skeleton 'What If' short-form video from a scenario.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full LLM mode — GPT-4o writes both scenes and voiceover:
  python main.py \\
    --scenario "What if you ate nothing but sugar for 30 days?" \\
    --character_img ./assets/skeleton_base.png

  # Manual scenes mode — scenes from Markdown, GPT-4o writes voiceover:
  python main.py \\
    --scenario "What if you ate nothing but sugar for 30 days?" \\
    --character_img ./assets/skeleton_base.png \\
    --scenes_md ./scenes/sugar_30_days.md

  # Dry-run (no API keys required):
  python main.py \\
    --scenario "What if you ate nothing but sugar for 30 days?" \\
    --character_img ./assets/skeleton_base.png \\
    --dry-run
        """,
    )
    parser.add_argument(
        "--scenario",
        type=str,
        required=True,
        help=(
            'The "What If" scenario. Used as the CLI fallback if the --scenes_md '
            'file also contains a scenario header. '
            'e.g. "What if you ate only sugar for 30 days?"'
        ),
    )
    parser.add_argument(
        "--character_img",
        type=str,
        required=True,
        help="Path to the skeleton character reference image (PNG/JPG).",
    )
    parser.add_argument(
        "--scenes_md",
        type=str,
        default=None,
        help=(
            "Optional path to a Markdown file containing hand-crafted scene prompts. "
            "When supplied, GPT-4o generates the voiceover ONLY — the scene image "
            "prompts come directly from the file. "
            "See the README for the expected file format."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help=(
            "Skip all real API calls. Generates placeholder assets and validates "
            "the render pipeline. Useful for testing without spending API credits."
        ),
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        default=False,
        help="Run all generation steps (script, images, video, audio) but skip the final render/stitch.",
    )
    parser.add_argument(
        "--bgm",
        type=str,
        default=None,
        help="Optional path to a background music file (MP3/WAV) to mix into the final video.",
    )
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Main Pipeline
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    scenario      = args.scenario
    character_img = Path(args.character_img)
    scenes_md     = Path(args.scenes_md) if args.scenes_md else None
    dry_run       = args.dry_run

    if args.bgm:
        config.BGM_PATH = Path(args.bgm)

    t_total_start = time.time()

    logger.info("=" * 70)
    logger.info("🦴  Skeleton Shorts Pipeline Starting")
    logger.info(f"   Scenario:     {scenario}")
    logger.info(f"   Character:    {character_img}")
    logger.info(f"   Scenes MD:    {scenes_md or '(none — GPT-4o will generate scenes)'}")
    logger.info(f"   BGM:          {config.BGM_PATH or '(none)'}")
    logger.info(f"   Dry-run:      {dry_run}")
    logger.info("=" * 70)

    # ── Dry-run: generate placeholder assets then jump straight to render ──────
    if dry_run:
        _run_dry(scenario, character_img, scenes_md)
        return

    # ── Validate character image exists before spending any API credits ────────
    if not character_img.exists():
        logger.error(f"Character image not found: {character_img}")
        sys.exit(1)

    # ══════════════════════════════════════════════════════════════════════════
    # Module A — LLM: Generate script OR voiceover-only
    #
    # BRANCH 1 (no --scenes_md): FULL MODE
    #   GPT-4o generates both the voiceover AND the scene image prompts.
    #
    # BRANCH 2 (--scenes_md supplied): MANUAL SCENES MODE
    #   Scene prompts are read from the Markdown file via md_parser.
    #   GPT-4o generates ONLY the narration voiceover, informed by the scenes.
    # ══════════════════════════════════════════════════════════════════════════
    if scenes_md is None:
        # ─── FULL MODE: GPT-4o writes everything ──────────────────────────────
        _log_step("A", "Generating full script via GPT-4o (scenes + voiceover)...")
        t0     = time.time()
        script = llm_service.generate_script(scenario)
        scenes    = script["scenes"]
        voiceover = script["voiceover"]
        logger.info(
            f"[Step A] Done in {time.time()-t0:.1f}s — "
            f"{len(scenes)} scenes, voiceover: {len(voiceover)} chars"
        )
    else:
        # ─── MANUAL SCENES MODE: parse MD → GPT-4o voiceover only ────────────
        _log_step("A", f"Loading scenes from Markdown file: {scenes_md}")
        t0 = time.time()

        # Step A-i: Parse the Markdown file
        md_scenario, scenes = md_parser.parse_scenes_md(scenes_md)

        # If the MD file embedded a "What if..." line, use it; else fall back to CLI
        effective_scenario = md_scenario or scenario
        if md_scenario and md_scenario != scenario:
            logger.info(
                f"[Step A] Scenario from MD file overrides CLI arg: {effective_scenario!r}"
            )

        logger.info(
            f"[Step A] Parsed {len(scenes)} scenes from {scenes_md.name} "
            f"({sum(s['duration'] for s in scenes):.1f}s total)"
        )

        # Step A-ii: Generate voiceover only (scenes already known)
        _log_step("A", "Generating voiceover via GPT-4o (scenes pre-defined)...")
        voiceover = llm_service.generate_voiceover_only(effective_scenario, scenes)
        scenario  = effective_scenario   # use for logging / any downstream refs

        logger.info(
            f"[Step A] Done in {time.time()-t0:.1f}s — "
            f"{len(voiceover)} chars voiceover, {len(scenes)} scenes from MD"
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Module B — Vision: Generate scene PNGs
    # ══════════════════════════════════════════════════════════════════════════
    _log_step("B", f"Generating {len(scenes)} scene images via Gemini...")
    t0 = time.time()
    image_paths = vision_service.generate_scene_images(character_img, scenes)
    logger.info(f"[Step B] Done in {time.time()-t0:.1f}s — {len(image_paths)} images")

    # ══════════════════════════════════════════════════════════════════════════
    # Modules C + D — Motion & Audio: Run CONCURRENTLY
    #
    # These two modules don't depend on each other:
    #   C (motion) depends on B (images) ✅ — already done
    #   D (audio)  depends on A (script) ✅ — already done
    #
    # We use ThreadPoolExecutor to run them in parallel threads. The async
    # polling inside motion_service uses asyncio.run() internally, which is
    # perfectly fine when called from a ThreadPoolExecutor worker thread.
    # ══════════════════════════════════════════════════════════════════════════
    _log_step("C+D", "Running Motion (Kling) and Audio (ElevenLabs) concurrently...")
    t0 = time.time()

    video_clip_paths = None
    narration_mp3    = None
    word_timestamps  = None

    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="pipeline") as executor:
        # Submit both tasks
        future_c: Future = executor.submit(
            motion_service.generate_video_clips, image_paths, scenes
        )
        future_d: Future = executor.submit(
            audio_service.generate_audio, voiceover
        )

        # Collect results as they complete (not necessarily in order)
        futures_map = {future_c: "C (Motion)", future_d: "D (Audio)"}
        for future in as_completed(futures_map):
            step_name = futures_map[future]
            try:
                result = future.result()
                if future is future_c:
                    video_clip_paths = result
                    logger.info(
                        f"[Step C] Motion done — {len(video_clip_paths)} clips ready"
                    )
                elif future is future_d:
                    narration_mp3, word_timestamps = result
                    logger.info(
                        f"[Step D] Audio done — {narration_mp3.name}, {word_timestamps.name}"
                    )
            except Exception as e:
                logger.error(f"[Step {step_name}] FAILED: {e}")
                executor.shutdown(wait=False, cancel_futures=True)
                raise

    logger.info(f"[Step C+D] Both complete in {time.time()-t0:.1f}s")

    # ══════════════════════════════════════════════════════════════════════════
    # Module E — Render: Assemble final video
    # ══════════════════════════════════════════════════════════════════════════
    if args.no_render:
        logger.info("[Step E] Skipped (--no-render). Assets saved in output/")
    else:
        _log_step("E", "Rendering final video with dynamic captions...")
        t0 = time.time()
        final_path = render_service.render_final_video(
            video_clip_paths=video_clip_paths,
            narration_mp3_path=narration_mp3,
            word_timestamps_path=word_timestamps,
        )
        logger.info(f"[Step E] Done in {time.time()-t0:.1f}s")

    # ══════════════════════════════════════════════════════════════════════════
    # Done
    # ══════════════════════════════════════════════════════════════════════════
    total_time = time.time() - t_total_start
    logger.info("=" * 70)
    logger.info(f"🎬  Pipeline complete in {total_time:.1f}s")
    if not args.no_render:
        logger.info(f"   Output: {final_path.resolve()}")
    else:
        logger.info(f"   Assets saved in: {config.OUTPUT_DIR.resolve()}")
    logger.info("=" * 70)


# ──────────────────────────────────────────────────────────────────────────────
# Dry-run mode — no real API calls
# ──────────────────────────────────────────────────────────────────────────────

def _run_dry(scenario: str, character_img: Path, scenes_md: Path | None = None) -> None:
    """
    Simulate the pipeline with placeholder assets to validate the render logic.
    Creates solid-colour PNG frames and a silent audio stand-in.

    If scenes_md is provided, the real scene prompts are parsed from it so the
    dry-run matches the actual number of scenes you'll generate for real.
    """
    import json
    import struct
    import wave
    from PIL import Image as PILImage

    logger.info("[DRY-RUN] Generating placeholder assets...")

    # ─── Determine scenes list ─────────────────────────────────────────────────────
    if scenes_md is not None:
        # Use the real scenes from the MD file — gives accurate scene count
        logger.info(f"[DRY-RUN] Loading scenes from {scenes_md.name}...")
        _, fake_scenes = md_parser.parse_scenes_md(scenes_md)
        fake_voiceover = (
            f"Dry-run voiceover for: {scenario}. "
            + " ".join(s["prompt"][:30] + "..." for s in fake_scenes[:3])
        )
    else:
        # Default placeholder scenes
        fake_scenes = [
            {"prompt": "Skeleton dancing in a sunlit meadow", "duration": 5.0},
            {"prompt": "Skeleton eating a sugar mountain",    "duration": 5.0},
            {"prompt": "Skeleton turned to dust dramatically","duration": 5.0},
        ]
        fake_voiceover = (
            "What if you ate nothing but sugar for thirty days? "
            "Day one seems fine. Day ten and your bones are rattling. "
            "Day thirty... you might not have bones at all."
        )

    logger.info(f"[DRY-RUN] Using {len(fake_scenes)} scenes")

    # Generate one cycling colour per scene (works for any scene count)
    _palette = [(200, 100, 100), (100, 200, 100), (100, 100, 200),
                (200, 200, 100), (100, 200, 200), (200, 100, 200)]
    colours = [_palette[i % len(_palette)] for i in range(len(fake_scenes))]

    # Fake PNG images (solid coloured)
    image_paths = []
    for i, colour in enumerate(colours):
        p = config.IMAGES_DIR / f"scene_{i}.png"
        img = PILImage.new("RGB", (config.VIDEO_WIDTH, config.VIDEO_HEIGHT), colour)
        img.save(str(p))
        image_paths.append(p)
        logger.info(f"[DRY-RUN] Created placeholder image: {p}")

    # Fake MP4 clips — solid-colour video using moviepy ColorClip
    from moviepy.editor import ColorClip

    video_clip_paths = []
    for i, (colour, scene) in enumerate(zip(colours, fake_scenes)):
        p = config.VIDEOS_DIR / f"scene_{i}.mp4"
        clip = ColorClip(
            size=(config.VIDEO_WIDTH, config.VIDEO_HEIGHT),
            color=colour,
            duration=float(scene.get("duration", 5.0)),
        )
        clip.write_videofile(
            str(p),
            fps=config.VIDEO_FPS,
            codec=config.VIDEO_CODEC,
            audio=False,
            logger=None,
        )
        clip.close()
        video_clip_paths.append(p)
        logger.info(f"[DRY-RUN] Created placeholder video: {p}")

    # Silent audio — duration must be >= total video length to avoid seek errors
    total_dur = sum(float(s.get("duration", 5.0)) for s in fake_scenes)
    narration_path = config.NARRATION_MP3_PATH
    _write_silent_wav(narration_path, duration_s=total_dur + 1)  # +1s safety buffer
    logger.info(f"[DRY-RUN] Created silent audio: {narration_path} ({total_dur:.1f}s)")

    # Fake word timestamps
    fake_words = [
        {"word": w, "start": i * 0.4, "end": i * 0.4 + 0.35}
        for i, w in enumerate(fake_voiceover.split())
    ]
    timestamps_path = config.WORD_TIMESTAMPS_PATH
    timestamps_path.write_text(json.dumps(fake_words, indent=2))
    logger.info(f"[DRY-RUN] Created fake timestamps: {timestamps_path}")

    # Run render
    logger.info("[DRY-RUN] Running render module with placeholder assets...")
    final = render_service.render_final_video(
        video_clip_paths=video_clip_paths,
        narration_mp3_path=narration_path,
        word_timestamps_path=timestamps_path,
    )
    logger.info(f"[DRY-RUN] ✅ Render successful: {final}")


def _write_silent_wav(path: Path, duration_s: float = 15, sample_rate: int = 44100) -> None:
    """Write a silent WAV file to path (moviepy can read WAV as audio)."""
    import wave, struct
    path = path.with_suffix(".wav")
    num_samples = int(sample_rate * duration_s)
    with wave.open(str(path), "w") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(struct.pack("<" + "h" * num_samples, *([0] * num_samples)))
    # Rename to .mp3 extension so config path matches
    path.rename(config.NARRATION_MP3_PATH)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _log_step(step_id: str, message: str) -> None:
    """Print a prominent step header to the log."""
    logger.info("")
    logger.info(f"─── Step {step_id} {'─' * (60 - len(step_id))}")
    logger.info(f"    {message}")
    logger.info("")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
