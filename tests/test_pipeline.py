"""
tests/test_pipeline.py — Unit Tests for Skeleton Shorts Pipeline

Uses unittest.mock to stub all external API calls, verifying that:
  1. llm_service.generate_script parses JSON correctly and validates schema
  2. audio_service._parse_word_timestamps produces correct word entries
  3. main.py wires modules in the correct order
  4. Dry-run mode completes without errors

Run with:
  python -m pytest tests/test_pipeline.py -v
  # or
  python -m unittest tests/test_pipeline.py
"""

import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

# ── Ensure the project root is on the path ────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set fake env vars BEFORE importing any service module (config loads .env on import)
os.environ.setdefault("OPENAI_API_KEY", "sk-test")
os.environ.setdefault("GOOGLE_API_KEY", "google-test")
os.environ.setdefault("PIAPI_KEY", "piapi-test")
os.environ.setdefault("ELEVENLABS_API_KEY", "el-test")


# ──────────────────────────────────────────────────────────────────────────────
# Module A — llm_service
# ──────────────────────────────────────────────────────────────────────────────
class TestLLMService(unittest.TestCase):
    """Tests for script generation and JSON validation."""

    def _make_valid_script(self) -> dict:
        return {
            "voiceover": "What if you ate only sugar for 30 days? Day one seems fine...",
            "scenes": [
                {"prompt": "A cartoonish skeleton eating a mountain of candy", "duration": 5.0},
                {"prompt": "A cartoonish skeleton vibrating with sugar rush energy", "duration": 5.0},
                {"prompt": "A cartoonish skeleton dissolved into sugar powder", "duration": 5.0},
            ],
        }

    @patch("llm_service.openai.OpenAI")
    def test_generate_script_success(self, mock_openai_cls):
        """Happy path: valid JSON returned by GPT-4o is parsed correctly."""
        import llm_service

        valid = self._make_valid_script()
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value.choices[0].message.content = (
            json.dumps(valid)
        )

        script = llm_service.generate_script("What if you ate only sugar for 30 days?")
        self.assertIn("voiceover", script)
        self.assertIn("scenes", script)
        self.assertEqual(len(script["scenes"]), 3)

    @patch("llm_service.openai.OpenAI")
    def test_generate_script_retries_on_bad_json(self, mock_openai_cls):
        """Should retry and succeed on the second attempt after bad JSON."""
        import llm_service

        valid = self._make_valid_script()
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        # First call returns garbage, second returns valid JSON
        mock_client.chat.completions.create.return_value.choices[0].message.content = (
            "not valid json"
        )
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            mock_resp = MagicMock()
            if call_count[0] == 1:
                mock_resp.choices[0].message.content = "not json at all"
            else:
                mock_resp.choices[0].message.content = json.dumps(valid)
            return mock_resp

        mock_client.chat.completions.create.side_effect = side_effect

        script = llm_service.generate_script("test scenario")
        self.assertEqual(call_count[0], 2)
        self.assertIn("voiceover", script)

    def test_validate_script_missing_voiceover(self):
        """_validate_script should raise ValueError on missing 'voiceover'."""
        import llm_service
        with self.assertRaises(ValueError, msg="Should raise for missing voiceover"):
            llm_service._validate_script({"scenes": [{"prompt": "x", "duration": 5}]})

    def test_validate_script_missing_duration(self):
        """_validate_script should raise ValueError when a scene has no duration."""
        import llm_service
        with self.assertRaises(ValueError):
            llm_service._validate_script({
                "voiceover": "test",
                "scenes": [{"prompt": "x"}],
            })


# ──────────────────────────────────────────────────────────────────────────────
# Module D — audio_service timestamp parsing
# ──────────────────────────────────────────────────────────────────────────────
class TestAudioService(unittest.TestCase):
    """Tests for ElevenLabs timestamp parsing."""

    def test_parse_word_timestamps_basic(self):
        """Should correctly parse character-level data into words."""
        import audio_service

        alignment = {
            "characters":                    ["H", "i", " ", "t", "h", "e", "r", "e"],
            "character_start_times_seconds": [0.0, 0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5],
            "character_end_times_seconds":   [0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6],
        }
        result = audio_service._parse_word_timestamps(alignment)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["word"], "Hi")
        self.assertEqual(result[1]["word"], "there")
        self.assertAlmostEqual(result[0]["start"], 0.0)
        self.assertAlmostEqual(result[0]["end"],   0.2)
        self.assertAlmostEqual(result[1]["start"], 0.3)
        self.assertAlmostEqual(result[1]["end"],   0.6)

    def test_parse_word_timestamps_empty(self):
        """Should return empty list on empty alignment."""
        import audio_service

        result = audio_service._parse_word_timestamps({
            "characters": [],
            "character_start_times_seconds": [],
            "character_end_times_seconds": [],
        })
        self.assertEqual(result, [])

    def test_parse_word_timestamps_trailing_word(self):
        """Should flush the last word even without a trailing space."""
        import audio_service

        alignment = {
            "characters":                    ["o", "k"],
            "character_start_times_seconds": [1.0, 1.1],
            "character_end_times_seconds":   [1.1, 1.2],
        }
        result = audio_service._parse_word_timestamps(alignment)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["word"], "ok")


# ──────────────────────────────────────────────────────────────────────────────
# llm_service — _validate_script edge cases
# ──────────────────────────────────────────────────────────────────────────────
class TestValidateScript(unittest.TestCase):
    def test_empty_scenes_list(self):
        import llm_service
        with self.assertRaises(ValueError):
            llm_service._validate_script({"voiceover": "test", "scenes": []})

    def test_duration_coercion(self):
        """String durations should be coerced to float."""
        import llm_service
        script = {
            "voiceover": "test",
            "scenes": [{"prompt": "x", "duration": "5"}],
        }
        llm_service._validate_script(script)
        self.assertIsInstance(script["scenes"][0]["duration"], float)


# ──────────────────────────────────────────────────────────────────────────────
# Integration smoke test — dry-run
# ──────────────────────────────────────────────────────────────────────────────
class TestDryRun(unittest.TestCase):
    """
    Runs the full pipeline in dry-run mode to validate the render chain works
    end-to-end without any real API calls.
    """

    def test_dry_run_produces_output_file(self):
        """dry-run should produce a final_short.mp4 (or .mp4) output file."""
        import config

        # Import main after env vars are set
        import importlib
        import main as main_module

        # Patch argparse so we don't need CLI args
        with patch(
            "sys.argv",
            [
                "main.py",
                "--scenario", "What if you ate only sugar?",
                "--character_img", "./assets/skeleton_base.png",
                "--dry-run",
            ],
        ):
            # Run dry-run; should not raise
            try:
                main_module.main()
            except SystemExit as e:
                self.fail(f"main() raised SystemExit({e}) unexpectedly")

        # Check that the final video was created
        self.assertTrue(
            config.FINAL_VIDEO_PATH.exists(),
            f"Expected {config.FINAL_VIDEO_PATH} to exist after dry-run",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
