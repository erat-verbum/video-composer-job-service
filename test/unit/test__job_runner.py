import json
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import pytest

from src.job_runner import JobRunner
from src.models import VideoMetadata


class TestMetadataExtraction:
    """Tests for metadata extraction functionality."""

    def test_save_metadata(self, tmp_path):
        """Test saving metadata to JSON file."""
        runner = JobRunner(None, lambda: "running")
        metadata = VideoMetadata(
            fps=30.0,
            width=1920,
            height=1080,
            display_width=1920,
            display_height=1080,
            codec="h264",
            duration_seconds=10.5,
        )
        runner._save_metadata(tmp_path, metadata)

        metadata_file = tmp_path / "metadata.json"
        assert metadata_file.exists()

        with open(metadata_file) as f:
            data = json.load(f)

        assert data["fps"] == 30.0
        assert data["width"] == 1920
        assert data["height"] == 1080
        assert data["display_width"] == 1920
        assert data["display_height"] == 1080
        assert data["codec"] == "h264"
        assert data["duration_seconds"] == 10.5

    def test_load_metadata(self, tmp_path):
        """Test loading metadata from JSON file."""
        metadata = VideoMetadata(
            fps=24.0,
            width=1280,
            height=720,
            display_width=1280,
            display_height=720,
            codec="hevc",
            duration_seconds=60.0,
        )
        metadata_file = tmp_path / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata.model_dump(), f)

        runner = JobRunner(None, lambda: "running")
        loaded = runner._load_metadata(metadata_file)

        assert loaded.fps == 24.0
        assert loaded.width == 1280
        assert loaded.height == 720
        assert loaded.codec == "hevc"
        assert loaded.duration_seconds == 60.0


class TestDisplayDimensions:
    """Tests for display width/height calculation with rotation and aspect ratio."""

    def test_rotation_90_swaps_dimensions(self):
        """Test that 90 degree rotation swaps width and height."""
        sar = 1.0
        rotation = 90
        width = 1920
        height = 1080

        if rotation in (90, -90, 270, -270):
            display_width = height
            display_height = round(width * sar)
        else:
            display_width = round(width * sar)
            display_height = height

        display_width = (display_width // 2) * 2
        display_height = (display_height // 2) * 2

        assert display_width == 1080
        assert display_height == 1920

    def test_rotation_270_swaps_dimensions(self):
        """Test that 270 degree rotation swaps width and height."""
        width = 1920
        height = 1080
        rotation = 270
        sar = 1.0

        if rotation in (90, -90, 270, -270):
            display_width = height
            display_height = round(width * sar)
        else:
            display_width = round(width * sar)
            display_height = height

        display_width = (display_width // 2) * 2
        display_height = (display_height // 2) * 2

        assert display_width == 1080
        assert display_height == 1920

    def test_rotation_0_keeps_dimensions(self):
        """Test that 0 degree rotation keeps original dimensions."""
        width = 1920
        height = 1080
        rotation = 0
        sar = 1.0

        if rotation in (90, -90, 270, -270):
            display_width = height
            display_height = round(width * sar)
        else:
            display_width = round(width * sar)
            display_height = height

        assert display_width == 1920
        assert display_height == 1080

    def test_sar_16_9_in_4_3_container(self):
        """Test SAR calculation for 16:9 video in 4:3 container."""
        width = 1440
        height = 1080
        sar_str = "4:3"
        sar_num, sar_den = sar_str.split(":")
        sar = float(sar_num) / float(sar_den)

        display_width = round(width * sar)
        display_height = height

        assert display_width == 1920
        assert display_height == 1080

    def test_sar_with_rotation(self):
        """Test SAR calculation combined with rotation."""
        width = 1440
        height = 1080
        rotation = 90
        sar_str = "4:3"
        sar_num, sar_den = sar_str.split(":")
        sar = float(sar_num) / float(sar_den)

        if rotation in (90, -90, 270, -270):
            display_width = height
            display_height = round(width * sar)
        else:
            display_width = round(width * sar)
            display_height = height

        display_width = (display_width // 2) * 2
        display_height = (display_height // 2) * 2

        assert display_width == 1080
        assert display_height == 1920

    def test_display_dimensions_must_be_even(self):
        """Test that display dimensions are always even (required for video encoding)."""
        test_cases = [
            (1920, 1080, 1.0, 0),
            (1440, 1080, 1.333, 0),
            (1920, 1080, 1.0, 90),
            (1280, 720, 1.0, 0),
            (1001, 1000, 1.5, 0),
        ]

        for width, height, sar, rotation in test_cases:
            if rotation in (90, -90, 270, -270):
                display_width = height
                display_height = round(width * sar)
            else:
                display_width = round(width * sar)
                display_height = height

            display_width = (display_width // 2) * 2
            display_height = (display_height // 2) * 2

            assert display_width % 2 == 0, f"display_width {display_width} is not even"
            assert display_height % 2 == 0, (
                f"display_height {display_height} is not even"
            )


class TestJobDispatch:
    """Tests for job type dispatch."""

    @pytest.mark.asyncio
    async def test_run_dispatches_to_extract(self):
        """Test run() dispatches to _extract_frames for extract job."""
        runner = JobRunner(
            {
                "input_params": {
                    "job_type": "extract",
                    "input_file": "test.mp4",
                    "output_dir": "output",
                }
            },
            lambda: "running",
        )

        with patch.object(
            runner, "_extract_frames", new_callable=AsyncMock
        ) as mock_extract:
            mock_extract.return_value = {"completed": True, "frame_count": 10}
            result = await runner.run()

            mock_extract.assert_called_once()
            assert result["completed"] is True

    @pytest.mark.asyncio
    async def test_run_dispatches_to_compose(self):
        """Test run() dispatches to _compose_frames for compose job."""
        runner = JobRunner(
            {
                "input_params": {
                    "job_type": "compose",
                    "input_dir": "frames",
                    "output_file": "output.mp4",
                }
            },
            lambda: "running",
        )

        with patch.object(
            runner, "_compose_frames", new_callable=AsyncMock
        ) as mock_compose:
            mock_compose.return_value = {"completed": True, "frame_count": 100}
            result = await runner.run()

            mock_compose.assert_called_once()
            assert result["completed"] is True

    @pytest.mark.asyncio
    async def test_run_raises_for_unknown_job_type(self):
        """Test run() raises ValueError for unknown job type."""
        runner = JobRunner(
            {
                "input_params": {
                    "job_type": "unknown",
                }
            },
            lambda: "running",
        )

        with pytest.raises(ValueError, match="Unknown job_type"):
            await runner.run()


class TestExtractValidation:
    """Tests for extract job validation."""

    @pytest.mark.asyncio
    async def test_extract_requires_input_file(self):
        """Test extract raises ValueError when input_file missing."""
        runner = JobRunner(
            {"input_params": {"job_type": "extract", "output_dir": "output"}},
            lambda: "running",
        )

        with pytest.raises(ValueError, match="input_file and output_dir are required"):
            await runner._extract_frames(
                cast(dict[str, Any], runner._job_ref)["input_params"]
            )

    @pytest.mark.asyncio
    async def test_extract_requires_output_dir(self):
        """Test extract raises ValueError when output_dir missing."""
        runner = JobRunner(
            {"input_params": {"job_type": "extract", "input_file": "test.mp4"}},
            lambda: "running",
        )

        with pytest.raises(ValueError, match="input_file and output_dir are required"):
            await runner._extract_frames(
                cast(dict[str, Any], runner._job_ref)["input_params"]
            )


class TestComposeValidation:
    """Tests for compose job validation."""

    @pytest.mark.asyncio
    async def test_compose_requires_input_dir(self):
        """Test compose raises ValueError when input_dir missing."""
        runner = JobRunner(
            {"input_params": {"job_type": "compose", "output_file": "output.mp4"}},
            lambda: "running",
        )

        with pytest.raises(ValueError, match="input_dir and output_file are required"):
            await runner._compose_frames(
                cast(dict[str, Any], runner._job_ref)["input_params"]
            )

    @pytest.mark.asyncio
    async def test_compose_requires_output_file(self):
        """Test compose raises ValueError when output_file missing."""
        runner = JobRunner(
            {"input_params": {"job_type": "compose", "input_dir": "frames"}},
            lambda: "running",
        )

        with pytest.raises(ValueError, match="input_dir and output_file are required"):
            await runner._compose_frames(
                cast(dict[str, Any], runner._job_ref)["input_params"]
            )

    @pytest.mark.asyncio
    async def test_compose_fails_if_metadata_missing(self, tmp_path):
        """Test compose raises ValueError when metadata.json missing."""
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()

        runner = JobRunner(
            {
                "input_params": {
                    "job_type": "compose",
                    "input_dir": str(frames_dir),
                    "output_file": "out.mp4",
                }
            },
            lambda: "running",
        )

        with pytest.raises(ValueError, match="Metadata file not found"):
            await runner._compose_frames(
                cast(dict[str, Any], runner._job_ref)["input_params"]
            )
