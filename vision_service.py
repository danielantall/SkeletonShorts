"""
vision_service.py — Module B: Scene Image Generation via Google GenAI

Takes the reference skeleton character image (as a filepath) plus the list of
scene prompts from Module A, and generates one static PNG per scene using the
Gemini multimodal image-generation model.

The reference image is passed alongside each prompt to keep the skeleton
character visually consistent across all scenes.
"""

import logging
from pathlib import Path

from google import genai
from google.genai import types

import config

# ──────────────────────────────────────────────────────────────────────────────
# Logger
# ──────────────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def generate_scene_images(
    character_img_path: str | Path,
    scenes: list[dict],
) -> list[Path]:
    """
    Generate one PNG image per scene prompt using Gemini image generation.

    The character reference image is included in every request so the model can
    maintain visual consistency for the skeleton protagonist.

    Args:
        character_img_path: Path to the skeleton reference PNG/JPG.
        scenes: The 'scenes' list from Module A — dicts with 'prompt' & 'duration'.

    Returns:
        A sorted list of Paths to the generated scene PNGs:
        [output/images/scene_0.png, output/images/scene_1.png, ...]

    Raises:
        FileNotFoundError: If the character image doesn't exist.
        RuntimeError: If the Gemini API returns no image for a scene.
    """
    character_img_path = Path(character_img_path)
    if not character_img_path.exists():
        raise FileNotFoundError(f"Character image not found: {character_img_path}")

    # Read reference image bytes once — reuse for every scene
    logger.info(f"[Vision] Loading reference character image: {character_img_path}")
    character_bytes: bytes = character_img_path.read_bytes()

    # Detect MIME type from file extension
    ext = character_img_path.suffix.lower()
    mime_type_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
    character_mime: str = mime_type_map.get(ext, "image/png")

    # Initialise the Google GenAI client
    client = genai.Client(api_key=config.GOOGLE_API_KEY)

    output_paths: list[Path] = []

    for idx, scene in enumerate(scenes):
        prompt: str = scene["prompt"]
        output_path: Path = config.IMAGES_DIR / f"scene_{idx}.png"

        logger.info(f"[Vision] Generating image {idx + 1}/{len(scenes)}: {prompt[:80]}...")

        try:
            # Build a multimodal request:
            #   Part 1 — reference character image (keeps the style consistent)
            #   Part 2 — text prompt for this specific scene
            reference_image_part = types.Part.from_bytes(
                data=character_bytes,
                mime_type=character_mime,
            )
            text_prompt_part = types.Part.from_text(
                text=(
                    "Using the skeleton character shown in the reference image as the "
                    "protagonist, generate the following scene:\n\n"
                    f"{prompt}\n\n"
                    "Style: Vibrant, stylised cartoon / animation art style. "
                    "9:16 portrait aspect ratio. Dramatic lighting. High detail."
                )
            )

            response = client.models.generate_content(
                model=config.GEMINI_IMAGE_MODEL,
                contents=[reference_image_part, text_prompt_part],
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                    # Safety settings — keep defaults; lower threshold for creative content
                ),
            )

            # Extract the image bytes from the response
            image_bytes = _extract_image_bytes(response, idx)

            # Write PNG to disk
            output_path.write_bytes(image_bytes)
            logger.info(f"[Vision] Saved scene {idx} → {output_path}")
            output_paths.append(output_path)

        except Exception as e:
            logger.error(f"[Vision] Failed to generate image for scene {idx}: {e}")
            raise RuntimeError(f"Image generation failed for scene {idx}: {e}") from e

    logger.info(f"[Vision] All {len(output_paths)} scene images generated.")
    return sorted(output_paths)  # ensure consistent ordering


# ──────────────────────────────────────────────────────────────────────────────
# Private helpers
# ──────────────────────────────────────────────────────────────────────────────

def _extract_image_bytes(response, scene_idx: int) -> bytes:
    """
    Pull the raw PNG/JPEG bytes out of a Gemini GenerateContentResponse.

    Gemini image responses store image data inside response.candidates[0]
    .content.parts — we search for the first Part that has inline_data.

    Raises:
        RuntimeError: If no image part is found in the response.
    """
    try:
        for candidate in response.candidates:
            for part in candidate.content.parts:
                # inline_data is a Blob object with .data (bytes) and .mime_type
                if hasattr(part, "inline_data") and part.inline_data is not None:
                    return part.inline_data.data
    except (AttributeError, IndexError) as e:
        raise RuntimeError(
            f"Unexpected Gemini response structure for scene {scene_idx}: {e}"
        ) from e

    raise RuntimeError(
        f"Gemini returned no image for scene {scene_idx}. "
        "Check your API key, quota, and model name in config.py."
    )
