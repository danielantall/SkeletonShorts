"""
motion_service.py — Module C: Image-to-Video via Gemini Veo

Takes a list of generated scene PNG filepaths, submits each to the Google GenAI 
video generation endpoint (Gemini Veo), then polls for completion asynchronously.

Flow per scene:
  1. Read PNG bytes → create google.genai.types.Part
  2. POST generate_videos  → get operation
  3. Async polling loop → client.operations.get(operation.name) until done
  4. Download the MP4 from the result URI
  5. Save to ./output/videos/scene_N.mp4

All tasks are polled concurrently via asyncio so the pipeline waits for the
slowest scene rather than the sum of all scenes.
"""

import asyncio
import logging
from pathlib import Path

import aiohttp
import requests
from google import genai
from google.genai import types

import config

# ──────────────────────────────────────────────────────────────────────────────
# Logger
# ──────────────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Public API (sync wrapper around async core)
# ──────────────────────────────────────────────────────────────────────────────

def generate_video_clips(
    image_paths: list[Path],
    scenes: list[dict],
    start_idx: int = 0,
) -> list[Path]:
    """
    Convert a list of scene PNG images into short MP4 clips via Gemini Veo.

    Args:
        image_paths: Ordered list of Paths to scene PNG files (from vision_service).
        scenes: Matching scene dicts — each dict's 'duration' is used as the
                target clip length.
        start_idx: Scene index offset for file naming (default 0).

    Returns:
        Ordered list of Paths to the downloaded MP4 clips.

    Raises:
        RuntimeError: If any task fails or times out.
    """
    if len(image_paths) != len(scenes):
        raise ValueError(
            f"Mismatch: {len(image_paths)} images vs {len(scenes)} scenes."
        )

    # Run the entire async workflow in a new event loop
    return asyncio.run(_run_all_tasks(image_paths, scenes, start_idx))


# ──────────────────────────────────────────────────────────────────────────────
# Async core
# ──────────────────────────────────────────────────────────────────────────────

# Configured batch size for concurrency
_BATCH_SIZE = 3


async def _run_all_tasks(
    image_paths: list[Path],
    scenes: list[dict],
    start_idx: int = 0,
) -> list[Path]:
    """
    Submit image-to-video tasks in batches of _BATCH_SIZE, poll each batch
    to completion before starting the next. Returns results in input order.
    """
    client = genai.Client(
        api_key=config.GOOGLE_API_KEY,
        http_options={"api_version": "v1beta"}
    )
    all_video_urls: list[str] = ["" for _ in image_paths]  # placeholder list

    # Process in batches of _BATCH_SIZE
    for batch_start in range(0, len(image_paths), _BATCH_SIZE):
        batch_end = min(batch_start + _BATCH_SIZE, len(image_paths))
        batch_indices = list(range(batch_start, batch_end))

        logger.info(
            f"[Motion] ── Batch {batch_start // _BATCH_SIZE + 1}: "
            f"scenes {[start_idx + i for i in batch_indices]} ──"
        )

        # Submit this batch — track (local_idx, real_idx, operation)
        operations: list[tuple[int, int, any]] = []
        for local_idx in batch_indices:
            real_idx = start_idx + local_idx
            op = await _submit_task(
                client, real_idx, image_paths[local_idx], scenes[local_idx]
            )
            operations.append((local_idx, real_idx, op))
            logger.info(f"[Motion] Scene {real_idx} submitted → operation started.")

        # Poll this batch concurrently until all complete
        poll_coros = [
            _poll_until_complete(client, op, ridx)
            for _, ridx, op in operations
        ]
        batch_urls = await asyncio.gather(*poll_coros)

        for (local_idx, _, _), url in zip(operations, batch_urls):
            all_video_urls[local_idx] = url

    # Download all MP4s (use start_idx offset for correct file naming)
    output_paths: list[Path] = []
    for local_idx, url in enumerate(all_video_urls):
        out_path = await asyncio.to_thread(_download_mp4, url, start_idx + local_idx)
        output_paths.append(out_path)

    logger.info(f"[Motion] All {len(output_paths)} clips downloaded.")
    return output_paths


async def _submit_task(
    client: genai.Client,
    scene_idx: int,
    img_path: Path,
    scene: dict,
) -> str:
    """
    POST one image-to-video task to Gemini Veo and return the operation name.
    """
    # Read the local image file bytes
    image_bytes = await asyncio.to_thread(img_path.read_bytes)
    
    ext = img_path.suffix.lower()
    mime_type_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
    mime_type = mime_type_map.get(ext, "image/png")
    
    image_part = types.Image(image_bytes=image_bytes, mime_type=mime_type)

    prompt = scene.get("prompt", "")
    logger.debug(f"[Motion] Submitting scene {scene_idx} to {config.GEMINI_VIDEO_MODEL}")
    logger.info(f"[Motion] Scene {scene_idx} payload: prompt={prompt[:40]}...")

    def do_submit():
        return client.models.generate_videos(
            model=config.GEMINI_VIDEO_MODEL,
            source=types.GenerateVideosSource(
                prompt=prompt,
                image=image_part
            ),
            config=types.GenerateVideosConfig(
                aspect_ratio="9:16",
                number_of_videos=1,
            )
        )

    try:
        operation = await asyncio.to_thread(do_submit)
        return operation
    except Exception as e:
        raise RuntimeError(
            f"Gemini Veo submit failed for scene {scene_idx}: {e}"
        ) from e

    return op_name


async def _poll_until_complete(
    client: genai.Client,
    operation,
    scene_idx: int,
) -> str:
    """
    Poll the task endpoint until the status is 'done'.
    Returns the MP4 download URL on success.

    Raises:
        RuntimeError: On task failure or timeout.
        asyncio.TimeoutError: If the total wait exceeds POLLING_TIMEOUT_S.
    """
    deadline = asyncio.get_event_loop().time() + config.POLLING_TIMEOUT_S
    attempt = 0

    while True:
        # Check timeout BEFORE sleeping so we fail fast if already past deadline
        now = asyncio.get_event_loop().time()
        if now >= deadline:
            raise asyncio.TimeoutError(
                f"[Motion] Scene {scene_idx} timed out after "
                f"{config.POLLING_TIMEOUT_S}s."
            )

        attempt += 1
        await asyncio.sleep(config.POLLING_INTERVAL_S)

        def do_poll():
            return client.operations.get(operation=operation)

        try:
            operation = await asyncio.to_thread(do_poll)
        except Exception as e:
            logger.warning(
                f"[Motion] Poll attempt {attempt} for scene {scene_idx} failed: {e}"
            )
            continue

        if not operation.done:
            logger.info(
                f"[Motion] Scene {scene_idx} operation — "
                f"status: polling (attempt {attempt})"
            )
            continue

        # Completed
        logger.info(
            f"[Motion] Scene {scene_idx} operation — "
            f"status: done (attempt {attempt})"
        )

        if getattr(operation, "error", None):
            error_msg = operation.error.message
            raise RuntimeError(f"[Motion] Scene {scene_idx} operation failed: {error_msg}")
            
        try:
            video_obj = operation.result.generated_videos[0].video
            logger.info(f"[Motion] Scene {scene_idx} complete → {video_obj.uri[:80]}...")
            return video_obj
        except Exception as e:
            raise RuntimeError(
                f"Failed to find generated video URL in result for scene {scene_idx}: {e}\n{operation.result}"
            )


# ──────────────────────────────────────────────────────────────────────────────
# Download helper (sync — run via asyncio.to_thread)
# ──────────────────────────────────────────────────────────────────────────────

def _download_mp4(url_or_video_obj, scene_idx: int) -> Path:
    """
    Download an MP4 from the given URL or generated Video object.
    Returns the Path to the saved file.
    """
    out_path = config.VIDEOS_DIR / f"scene_{scene_idx}.mp4"
    logger.info(f"[Motion] Downloading MP4 for scene {scene_idx}...")

    if isinstance(url_or_video_obj, str) and url_or_video_obj.startswith("http"):
        response = requests.get(url_or_video_obj, stream=True, timeout=120)
        response.raise_for_status()

        with out_path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    else:
        # SDK Video object -> can save itself
        client = genai.Client(
            api_key=config.GOOGLE_API_KEY,
            http_options={"api_version": "v1beta"}
        )
        # Use the explicit boilerplate pattern:
        # 1. Download
        # 2. Save
        client.files.download(file=url_or_video_obj)
        logger.info(f"[Motion] Acquired video object. Saving to local path: {out_path}")
        url_or_video_obj.save(str(out_path))

    size_mb = out_path.stat().st_size / (1024 * 1024)
    logger.info(f"[Motion] Saved {out_path} ({size_mb:.2f} MB)")
    return out_path
