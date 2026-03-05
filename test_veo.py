"""
Quick test: submit an image to Gemini Veo via motion_service and poll until done.
Usage:  python test_veo.py
"""
import asyncio
from pathlib import Path
from motion_service import _run_all_tasks

IMAGE = Path("output/images/scene_0.png")

if not IMAGE.exists():
    print(f"❌ Cannot find {IMAGE}. Run the pipeline first to generate an image.")
    exit(1)

print("[1] Testing Gemini Veo Image-to-Video generation...")
scenes = [{"prompt": "The skeleton character dances happily", "duration": 5}]

async def test():
    try:
        output_paths = await _run_all_tasks([IMAGE], scenes, start_idx=0)
        print(f"\n✅ Done! Video saved at: {output_paths[0]}")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test())
