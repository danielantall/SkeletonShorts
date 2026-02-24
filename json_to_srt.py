#!/usr/bin/env python3
"""Convert word_timestamps.json → captions.srt"""

import json
import sys
from pathlib import Path


def seconds_to_srt_time(s: float) -> str:
    """Convert seconds (float) to SRT timestamp format: HH:MM:SS,mmm"""
    hours = int(s // 3600)
    minutes = int((s % 3600) // 60)
    secs = int(s % 60)
    millis = int((s % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def json_to_srt(json_path: str, srt_path: str, words_per_caption: int = 3, speedup: float = 1.0) -> None:
    """
    Group words into captions of `words_per_caption` and write an SRT file.

    Args:
        json_path:          Path to word_timestamps.json
        srt_path:           Output .srt path
        words_per_caption:  How many words per subtitle line (default 3)
        speedup:            Factor to speed up the timestamps by (e.g. 1.1)
    """
    with open(json_path, "r", encoding="utf-8") as f:
        words = json.load(f)

    captions: list[dict] = []
    for i in range(0, len(words), words_per_caption):
        group = words[i : i + words_per_caption]
        text = " ".join(w["word"] for w in group)
        start = group[0]["start"] / speedup
        end = group[-1]["end"] / speedup
        captions.append({"start": start, "end": end, "text": text})

    with open(srt_path, "w", encoding="utf-8") as f:
        for idx, cap in enumerate(captions, start=1):
            f.write(f"{idx}\n")
            f.write(f"{seconds_to_srt_time(cap['start'])} --> {seconds_to_srt_time(cap['end'])}\n")
            f.write(f"{cap['text']}\n\n")

    print(f"✅ Wrote {len(captions)} captions → {srt_path}")


if __name__ == "__main__":
    json_in = sys.argv[1] if len(sys.argv) > 1 else "output/word_timestamps.json"
    srt_out = sys.argv[2] if len(sys.argv) > 2 else "output/captions.srt"
    words_n = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    spd_up  = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0

    if not Path(json_in).exists():
        print(f"❌ {json_in} not found")
        sys.exit(1)

    json_to_srt(json_in, srt_out, words_n, spd_up)
