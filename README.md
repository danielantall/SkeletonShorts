# 🦴 SkeletonShort - Become Viral 

## Description
Automated pipeline that generates viral "What If" skeleton animation YouTube Shorts from a single prompt.

## Motivation
Making videos is tedious, why not make it easy to print out content which actually reaches the right users efficiently. SkeletonShorts automates this process by fulfilling script generation, video generation, and video compilation all in one pipeline.

**Pipeline:** Script → Images → Video → Audio → Final Render

## Quick Start

### 1. Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Set up API keys

```bash
cp .env.example .env
# Fill in your keys:
#   GOOGLE_API_KEY    — Gemini (script, image + video gen)
#   ELEVENLABS_API_KEY — narration audio
```

### 3. Run

```bash
# Full pipeline — generates everything from a prompt
python main.py \
  --scenario "What if you never slept?" \
  --character_img ./assets/skeleton_base.png

# With hand-crafted scenes (skips scene generation)
python main.py \
  --scenario "What if you ate only sugar for 30 days?" \
  --character_img ./assets/skeleton_base.png \
  --scenes_md ./scenes/sugar_30_days.md
```

---

## CLI Flags

| Flag | What it does |
|---|---|
| `--scenario` | The "What if" topic (required) |
| `--character_img` | Path to skeleton reference image (required) |
| `--scenes_md` | Path to Markdown scene file (optional — LLM generates scenes if omitted) |
| `--dry-run` | Skip all API calls, use placeholder assets to test the render pipeline |
| `--no-render` | Run all generation steps but skip the final video stitch |

---

## Scene File Format

Create a `.md` file in `scenes/`. The first line can optionally be the scenario.

```markdown
What if you ate only sugar for 30 days?

Scene 1: Day 1 Sugar Rush
Place the skeleton character inside a bright colorful kitchen surrounded
by candy and soda. Use vibrant lighting and a low-angle close-up.

Scene 2: Energy Crash
Place the skeleton character slumped on a couch surrounded by empty
soda cans. Use dim afternoon lighting and a wide shot.

Scene 3: Hospital Visit [7s]
Place the skeleton character in a hospital bed with an IV drip.
Use dramatic low-key lighting.
```

**Header triggers** (any of these start a new scene):
- `Scene:` or `Scene N:`
- `🎬`
- `## ` or `### `
- `B-Roll`
- `Prompt N:`

**Duration:** Default 5s per scene. Override with `[7s]` in the header.

---

## Pipeline Steps

| Step | Service | What it does |
|---|---|---|
| **A** | `llm_service.py` | Generates voiceover script (Gemini) |
| **B** | `vision_service.py` | Generates scene images (Gemini) |
| **C** | `motion_service.py` | Generates video clips from images (Gemini Veo) |
| **D** | `audio_service.py` | Generates narration audio + word timestamps (ElevenLabs) |
| **E** | `render_service.py` | Composites video + audio + captions → `output/final_short.mp4` |

Steps C and D run **in parallel** to save time.

---

## Utilities

### Convert timestamps to SRT captions

```bash
python json_to_srt.py                          # default: 3 words per caption
python json_to_srt.py output/word_timestamps.json output/captions.srt 4
```

### Speed up clips with ffmpeg

```bash
for f in output/videos/scene_*.mp4; do
  ffmpeg -y -i "$f" -filter:v "setpts=PTS/1.6" -an "${f%.mp4}_fast.mp4"
  mv "${f%.mp4}_fast.mp4" "$f"
done
```

### Burn SRT captions into video

```bash
ffmpeg -i output/final_short.mp4 \
  -vf "subtitles=output/captions.srt:force_style='FontSize=24,FontName=LuckiestGuy'" \
  -c:a copy output/final_with_subs.mp4
```

---

## Output Structure

```
output/
├── images/          # Scene PNGs from Gemini
│   ├── scene_0.png
│   └── ...
├── videos/          # Scene MP4s from Gemini Veo
│   ├── scene_0.mp4
│   └── ...
├── narration.mp3    # ElevenLabs voiceover
├── word_timestamps.json
├── captions.srt     # Generated via json_to_srt.py
└── final_short.mp4  # Rendered final video
```

---

## Configuration

All settings are in [`config.py`](config.py):

- **Video:** 1080×1920 (9:16), 30fps
- **Veo:** veo-3.1-fast-generate-preview
- **Captions:** LuckiestGuy font, 100px, white fill + black stroke
- **Audio:** ElevenLabs Turbo v2.5, 1.2× speed
- **LLM:** Gemini 2.5 Flash, temperature 0.8
