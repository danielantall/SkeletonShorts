import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import motion_service
import config

# Setup basic logging to see the output
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")

async def generate_extra():
    image_paths = [Path("output/images/scene_7.png")]
    
    prompt = (
        "Place the skeleton character standing beside Andrew Tate on a stage facing a cheering crowd, doing the gangnam style dance. "
        "Keep the same 3D style. Use intense stage spotlights cutting through light smoke. "
        "Open with a sweeping crane shot from behind the crowd toward the stage, then transition into a low-angle push-in emphasizing confidence. "
        "Add flashing camera lights and slight lens distortion for energy. Rendered in clean cinematic 3D, highly detailed."
    )
    
    scenes = [{"prompt": prompt, "duration": 5.0}]
    
    print("Generating scene_9 video clip using scene_7 image and Sheikh of Dubai prompt...")
    # Start idx 9 means it will be saved as scene_9.mp4
    output_paths = await motion_service._run_all_tasks(image_paths, scenes, start_idx=9)
    print(f"Success! Video saved to: {output_paths[0]}")

if __name__ == "__main__":
    asyncio.run(generate_extra())
