import asyncio
import json
from pathlib import Path
from typing import Any, Callable, Optional

from .models import AudioTrack, JobType, SubtitleTrack, VideoMetadata

DATA_DIR = "/app/data"


def resolve_data_path(relative_path: str) -> Path:
    """Resolve a relative path to an absolute path within the data directory."""
    path = Path(relative_path)
    if path.is_absolute():
        return path
    return Path(DATA_DIR) / relative_path


class JobRunner:
    """Manages job execution with progress updates and cancellation support."""

    def __init__(
        self, job_ref: Optional[dict[str, Any]], get_status: Callable[[], str]
    ):
        self._job_ref = job_ref
        self._get_status = get_status

    async def run(self) -> dict[str, Any]:
        """
        Run a job based on job_type.

        Returns:
            dict: Result containing job completion details

        Raises:
            ValueError: If required parameters are missing or invalid
            RuntimeError: If ffmpeg/ffprobe fails
        """
        input_params = self._job_ref.get("input_params", {}) if self._job_ref else {}
        job_type = input_params.get("job_type", "extract")

        if job_type == JobType.EXTRACT:
            return await self._extract_frames(input_params)
        elif job_type == JobType.COMPOSE:
            return await self._compose_frames(input_params)
        else:
            raise ValueError(f"Unknown job_type: {job_type}")

    async def _extract_frames(self, input_params: dict[str, Any]) -> dict[str, Any]:
        """
        Extract all frames from a video file to PNG images.

        Args:
            input_params: Dictionary with input_file and output_dir

        Returns:
            dict: Result containing extraction status and frame count

        Raises:
            ValueError: If input file doesn't exist
            RuntimeError: If ffmpeg fails
        """
        input_file = input_params.get("input_file")
        output_dir = input_params.get("output_dir")

        if not input_file or not output_dir:
            raise ValueError("input_file and output_dir are required")

        input_path = resolve_data_path(input_file)
        output_path = resolve_data_path(output_dir)

        if not input_path.exists():
            raise ValueError(f"Input file not found: {input_path}")

        output_path.mkdir(parents=True, exist_ok=True)

        metadata = await self._extract_metadata(input_path)
        self._save_metadata(output_path, metadata)

        output_pattern = str(output_path / "frame_%04d.png")

        ffmpeg_args = ["ffmpeg", "-i", str(input_path), "-y"]

        ffmpeg_args.extend(
            [
                "-map",
                "0:v",
                "-vf",
                f"scale={metadata.display_width}:{metadata.display_height}",
                output_pattern,
            ]
        )

        for track in metadata.audio_tracks:
            ffmpeg_args.extend(
                [
                    "-map",
                    f"0:{track.stream_index}",
                    "-c:a",
                    "copy",
                    "-y",
                    str(output_path / track.filename),
                ]
            )

        for track in metadata.subtitle_tracks:
            ffmpeg_args.extend(
                [
                    "-map",
                    f"0:{track.stream_index}",
                    "-c:s",
                    "copy",
                    "-y",
                    str(output_path / track.filename),
                ]
            )

        process = await asyncio.create_subprocess_exec(
            *ffmpeg_args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        self._update_progress(10)

        try:
            await process.wait()
        except asyncio.CancelledError:
            process.kill()
            await process.wait()
            raise

        if process.returncode != 0:
            if process.stderr is not None:
                stderr = await process.stderr.read()
                error_msg = stderr.decode() if stderr else "Unknown error"
            else:
                error_msg = "Unknown error"
            raise RuntimeError(f"FFmpeg failed: {error_msg}")

        frame_files = sorted(
            output_path.glob("frame_*.png"), key=lambda p: int(p.stem.split("_")[1])
        )
        frame_count = len(frame_files)

        return {
            "completed": True,
            "job_type": "extract",
            "input_file": input_file,
            "output_dir": output_dir,
            "frame_count": frame_count,
            "metadata_file": f"{output_dir}/metadata.json",
            "audio_track_count": len(metadata.audio_tracks),
            "subtitle_track_count": len(metadata.subtitle_tracks),
        }

    async def _compose_frames(self, input_params: dict[str, Any]) -> dict[str, Any]:
        """
        Compose a video from PNG frames.

        Args:
            input_params: Dictionary with input_dir and output_file

        Returns:
            dict: Result containing composition status and details

        Raises:
            ValueError: If input directory doesn't exist or metadata missing
            RuntimeError: If ffmpeg fails
        """
        input_dir = input_params.get("input_dir")
        output_file = input_params.get("output_file")

        if not input_dir or not output_file:
            raise ValueError("input_dir and output_file are required")

        input_path = resolve_data_path(input_dir)
        output_path = resolve_data_path(output_file)

        if not input_path.exists():
            raise ValueError(f"Input directory not found: {input_path}")

        metadata_path = input_path / "metadata.json"
        if not metadata_path.exists():
            raise ValueError(f"Metadata file not found: {metadata_path}")

        metadata = self._load_metadata(metadata_path)

        frame_files = sorted(
            input_path.glob("frame_*.png"), key=lambda p: int(p.stem.split("_")[1])
        )
        if not frame_files:
            raise ValueError(f"No frame files found in: {input_path}")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        input_pattern = str(input_path / "frame_%04d.png")

        ffmpeg_args = [
            "ffmpeg",
            "-framerate",
            str(metadata.fps),
            "-i",
            input_pattern,
            "-y",
        ]

        audio_extensions = [
            "aac",
            "mp3",
            "m4a",
            "ac3",
            "eac3",
            "flac",
            "ogg",
            "opus",
            "wav",
        ]
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(sorted(input_path.glob(f"audio_*.{ext}")))
        audio_files.sort()

        subtitle_extensions = ["srt", "ass", "vtt"]
        subtitle_files = []
        for ext in subtitle_extensions:
            subtitle_files.extend(sorted(input_path.glob(f"subtitle_*.{ext}")))
        subtitle_files.sort()

        audio_index = 0
        for audio_file in audio_files:
            ffmpeg_args.extend(["-i", str(audio_file)])

        for subtitle_file in subtitle_files:
            ffmpeg_args.extend(["-i", str(subtitle_file)])

        ffmpeg_args.extend(
            [
                "-map",
                "0:v",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
            ]
        )

        for _ in audio_files:
            ffmpeg_args.extend(["-map", f"{audio_index + 1}:a", "-c:a", "copy"])
            audio_index += 1

        subtitle_input_offset = 1 + len(audio_files)
        for i in range(len(subtitle_files)):
            ffmpeg_args.extend(
                ["-map", f"{subtitle_input_offset + i}:s", "-c:s", "copy"]
            )

        ffmpeg_args.append(str(output_path))

        process = await asyncio.create_subprocess_exec(
            *ffmpeg_args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        self._update_progress(10)

        try:
            await process.wait()
        except asyncio.CancelledError:
            process.kill()
            await process.wait()
            raise

        if process.returncode != 0:
            if process.stderr is not None:
                stderr = await process.stderr.read()
                error_msg = stderr.decode() if stderr else "Unknown error"
            else:
                error_msg = "Unknown error"
            raise RuntimeError(f"FFmpeg failed: {error_msg}")

        return {
            "completed": True,
            "job_type": "compose",
            "input_dir": input_dir,
            "output_file": output_file,
            "frame_count": len(frame_files),
            "fps": metadata.fps,
            "audio_track_count": len(metadata.audio_tracks),
            "subtitle_track_count": len(metadata.subtitle_tracks),
        }

    async def _extract_metadata(self, video_path: Path) -> VideoMetadata:
        """
        Extract metadata from a video file using ffprobe.

        Args:
            video_path: Path to the video file

        Returns:
            VideoMetadata: Extracted metadata

        Raises:
            RuntimeError: If ffprobe fails
        """
        process = await asyncio.create_subprocess_exec(
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=r_frame_rate,width,height,codec_name,duration,sample_aspect_ratio,rotation",
            "-of",
            "json",
            str(video_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            await process.wait()
        except asyncio.CancelledError:
            process.kill()
            await process.wait()
            raise

        if process.returncode != 0:
            if process.stderr is not None:
                stderr = await process.stderr.read()
                error_msg = stderr.decode() if stderr else "Unknown error"
            else:
                error_msg = "Unknown error"
            raise RuntimeError(f"FFprobe failed: {error_msg}")

        if process.stdout is not None:
            stdout = await process.stdout.read()
            output = json.loads(stdout.decode())
        else:
            raise RuntimeError("FFprobe failed to return output")

        stream = output.get("streams", [{}])[0]

        fps_str = stream.get("r_frame_rate", "30/1")
        if "/" in fps_str:
            num, den = fps_str.split("/")
            fps = float(num) / float(den)
        else:
            fps = float(fps_str)

        width = int(stream.get("width", 0))
        height = int(stream.get("height", 0))
        codec = stream.get("codec_name", "unknown")
        duration = float(stream.get("duration", 0.0))

        sar_str = stream.get("sample_aspect_ratio", "1:1")
        if ":" in sar_str:
            sar_num, sar_den = sar_str.split(":")
            sar = float(sar_num) / float(sar_den)
        else:
            sar = float(sar_str)

        rotation = int(stream.get("rotation", 0))

        if rotation in (90, -90, 270, -270):
            display_width = height
            display_height = round(width * sar)
        else:
            display_width = round(width * sar)
            display_height = height

        display_width = (display_width // 2) * 2
        display_height = (display_height // 2) * 2

        audio_tracks = await self._extract_audio_streams(video_path)
        subtitle_tracks = await self._extract_subtitle_streams(video_path)

        return VideoMetadata(
            fps=fps,
            width=width,
            height=height,
            display_width=display_width,
            display_height=display_height,
            codec=codec,
            duration_seconds=duration,
            audio_tracks=audio_tracks,
            subtitle_tracks=subtitle_tracks,
        )

    async def _extract_audio_streams(self, video_path: Path) -> list[AudioTrack]:
        """
        Extract audio stream information from a video file.

        Args:
            video_path: Path to the video file

        Returns:
            list[AudioTrack]: List of audio tracks
        """
        process = await asyncio.create_subprocess_exec(
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index,codec_name,tags",
            "-of",
            "json",
            str(video_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            await process.wait()
        except asyncio.CancelledError:
            process.kill()
            await process.wait()
            raise

        if process.returncode != 0:
            return []

        if process.stdout is not None:
            stdout = await process.stdout.read()
            output = json.loads(stdout.decode())
        else:
            return []

        audio_tracks = []
        streams = output.get("streams", [])

        codec_to_ext = {
            "aac": "aac",
            "mp3": "mp3",
            "ac3": "ac3",
            "eac3": "eac3",
            "flac": "flac",
            "alac": "m4a",
            "opus": "opus",
            "vorbis": "ogg",
            "wav": "wav",
            "pcm_s16le": "wav",
        }

        for stream in streams:
            stream_index = int(stream.get("index", 0))
            codec = stream.get("codec_name", "unknown")
            tags = stream.get("tags", {})
            language = tags.get("language")

            ext = codec_to_ext.get(codec, "m4a")
            filename = f"audio_{stream_index}.{ext}"

            audio_tracks.append(
                AudioTrack(
                    stream_index=stream_index,
                    codec=codec,
                    language=language,
                    filename=filename,
                )
            )

        return audio_tracks

    async def _extract_subtitle_streams(self, video_path: Path) -> list[SubtitleTrack]:
        """
        Extract subtitle stream information from a video file.

        Args:
            video_path: Path to the video file

        Returns:
            list[SubtitleTrack]: List of subtitle tracks
        """
        process = await asyncio.create_subprocess_exec(
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "s",
            "-show_entries",
            "stream=index,codec_name,tags",
            "-of",
            "json",
            str(video_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            await process.wait()
        except asyncio.CancelledError:
            process.kill()
            await process.wait()
            raise

        if process.returncode != 0:
            return []

        if process.stdout is not None:
            stdout = await process.stdout.read()
            output = json.loads(stdout.decode())
        else:
            return []

        subtitle_tracks = []
        streams = output.get("streams", [])

        codec_to_ext = {
            "subrip": "srt",
            "srt": "srt",
            "ass": "ass",
            "ssa": "ass",
            "webvtt": "vtt",
            "vtt": "vtt",
        }

        for stream in streams:
            stream_index = int(stream.get("index", 0))
            codec = stream.get("codec_name", "unknown")
            tags = stream.get("tags", {})
            language = tags.get("language")

            ext = codec_to_ext.get(codec, "srt")
            filename = f"subtitle_{stream_index}.{ext}"

            subtitle_tracks.append(
                SubtitleTrack(
                    stream_index=stream_index,
                    codec=codec,
                    language=language,
                    filename=filename,
                )
            )

        return subtitle_tracks

    def _save_metadata(self, output_dir: Path, metadata: VideoMetadata) -> None:
        """Save metadata to a JSON file in the output directory."""
        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata.model_dump(), f)

    def _load_metadata(self, metadata_path: Path) -> VideoMetadata:
        """Load metadata from a JSON file."""
        with open(metadata_path, "r") as f:
            data = json.load(f)
        return VideoMetadata(**data)

    def _update_progress(self, progress: int) -> None:
        """Update job progress."""
        if self._job_ref:
            self._job_ref["progress"] = progress


async def run_job(
    job_ref: Optional[dict[str, Any]],
    get_status: Callable[[], str],
) -> dict[str, Any]:
    """Entry point for running a job."""
    runner = JobRunner(job_ref, get_status)
    return await runner.run()
