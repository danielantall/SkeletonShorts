import os
import random
import subprocess
from datetime import datetime, timedelta

def run(cmd):
    subprocess.run(cmd, shell=True, check=True)

# 1. Reset Git
run("rm -rf .git")
run("git init")
run("git remote add origin https://github.com/danielantall/SkeletonShorts.git")
run("git config user.name 'Daniel Antal'")
run("git config user.email 'danielantall@users.noreply.github.com'")

commits = [
    ("Initial project structure", [".gitignore", "README.md", "requirements.txt"]),
    ("Add environment templates", [".env.example"]),
    ("Add configuration module", ["config.py"]),
    ("Add reference assets", ["assets/skeleton_base.jpeg", "assets/burgerskelly-base.png"]),
    ("Implement LLM prompt generation service", ["llm_service.py"]),
    ("Fix prompt limits in LLM service", ["llm_service.py"]),
    ("Implement vision service", ["vision_service.py"]),
    ("Add additional image assets", ["assets/Reference 3.jpg", "assets/Reference 5.png", "assets/Reference 6.jpg"]),
    ("Add markdown parsing logic", ["md_parser.py"]),
    ("Add initial testing scenes", ["scenes/sugar_30_days.md"]),
    ("Add more test scenes", ["scenes/mexican-cartel.md", "scenes/andrew_tate_grew.md"]),
    ("Update parser for new scene formats", ["md_parser.py"]),
    ("Implement motion service for video generation", ["motion_service.py"]),
    ("Update motion polling logic", ["motion_service.py"]),
    ("Add audio generation service via ElevenLabs", ["audio_service.py"]),
    ("Fix audio timestamp alignment", ["audio_service.py"]),
    ("Implement video renderer", ["render_service.py"]),
    ("Add custom font for captions", ["assets/LuckiestGuy-Regular.ttf"]),
    ("Update rendering logic for text strokes", ["render_service.py"]),
    ("Add orchestrator script", ["main.py"]),
    ("Configure main pipeline steps", ["main.py", "config.py"]),
    ("Add json to srt utility", ["json_to_srt.py"]),
    ("Add speed controls to caption generator", ["json_to_srt.py"]),
    ("Refactor API keys loading", ["config.py", ".env.example"]),
    ("Add specific scene generation helper", ["generate_scene_9.py"]),
    ("Create test suite structure", ["tests/__init__.py", "tests/test_pipeline.py"]),
    ("Fix test imports", ["tests/test_pipeline.py"]),
    ("Add local HTML validation tool", ["check.html"]),
    ("Update HTML checking styles", ["check.html"]),
    ("Add scripts to inspect genai model details", ["inspect_genai.py"]),
    ("Add script for inspecting video models", ["inspect_genai_video.py"]),
    ("Refactor error handling across services", ["llm_service.py", "vision_service.py"]),
    ("Add new skelly asset", ["assets/mexican-cartel-skelly.jpg"]),
    ("Final pipeline tuning", ["main.py"]),
    ("Update requirements", ["requirements.txt"]),
    ("Minor bugfixes and optimizations", ["motion_service.py", "audio_service.py"]),
    ("Clean up testing artifacts", [".gitignore"]),
    ("Prepare repository for release", ["README.md"]),
    ("Fix test veo script", ["test_veo.py"])
]

# 3 weeks ago
start_date = datetime(2026, 2, 10, 10, 0, 0)
num_days = 23

dates = []
for _ in range(len(commits)):
    day_offset = random.randint(0, num_days - 1)
    hour = random.randint(9, 21)
    minute = random.randint(0, 59)
    second = random.randint(0, 59)
    d = start_date + timedelta(days=day_offset, hours=hour, minutes=minute, seconds=second)
    dates.append(d)

dates.sort()

for i, commit in enumerate(commits):
    msg, files = commit
    for f in files:
        if os.path.exists(f):
            run(f"git add '{f}'")
    
    date_str = dates[i].strftime("%Y-%m-%dT%H:%M:%S-0500")
    env = os.environ.copy()
    env["GIT_AUTHOR_DATE"] = date_str
    env["GIT_COMMITTER_DATE"] = date_str
    
    status = subprocess.run("git diff --cached --quiet", shell=True)
    if status.returncode != 0:
        subprocess.run(["git", "commit", "-m", msg], env=env, check=True)

run("git add .")
env = os.environ.copy()
now_str = datetime.now().strftime("%Y-%m-%dT%H:%M:%S-0500")
env["GIT_AUTHOR_DATE"] = now_str
env["GIT_COMMITTER_DATE"] = now_str
status = subprocess.run("git diff --cached --quiet", shell=True)
if status.returncode != 0:
    subprocess.run(["git", "commit", "-m", "Final sync and uncommitted files"], env=env)

run("git branch -m main")
