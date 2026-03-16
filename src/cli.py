import asyncio
import signal
import sys
from pathlib import Path
from typing import Any, Callable, Optional

import typer
from rich.console import Console

from .job_runner import JobRunner
from .models import JobStatus, JobType

app = typer.Typer(
    name="video-processing-job-service",
    help="CLI for video frame extraction and composition without Docker",
)
console = Console()

DATA_DIR = Path("/app/data") if Path("/app/data").exists() else Path("data")


class CliJobRunner(JobRunner):
    """Job runner with CLI progress output."""

    def __init__(
        self,
        job_ref: Optional[dict[str, Any]],
        get_status: Callable[[], str],
    ):
        super().__init__(job_ref, get_status)
        self._last_progress = -1

    async def run(self) -> dict[str, Any]:
        """Run the job with progress output."""
        input_params = self._job_ref.get("input_params", {}) if self._job_ref else {}
        job_type = input_params.get("job_type", "extract")

        if job_type == JobType.EXTRACT:
            return await self._run_extract(input_params)
        elif job_type == JobType.COMPOSE:
            return await self._run_compose(input_params)
        else:
            raise ValueError(f"Unknown job_type: {job_type}")

    async def _run_extract(self, input_params: dict[str, Any]) -> dict[str, Any]:
        """Run the frame extraction job with progress output."""
        input_file = input_params.get("input_file")
        output_dir = input_params.get("output_dir")

        if not input_file or not output_dir:
            raise ValueError("input_file and output_dir are required")

        input_path = Path(DATA_DIR) / input_file
        output_path = Path(DATA_DIR) / output_dir

        if not input_path.exists():
            raise ValueError(f"Input file not found: {input_path}")

        output_path.mkdir(parents=True, exist_ok=True)

        from .job_runner import JobRunner

        runner = JobRunner(None, lambda: "running")
        metadata = await runner._extract_metadata(input_path)
        runner._save_metadata(output_path, metadata)

        output_pattern = str(output_path / "frame_%04d.png")

        console.print(f"[cyan]Starting extraction:[/cyan] {input_file} -> {output_dir}")

        ffmpeg_args = [
            "ffmpeg",
            "-i",
            str(input_path),
            "-y",
            "-map",
            "0:v",
            "-vf",
            f"scale={metadata.display_width}:{metadata.display_height}",
            output_pattern,
        ]

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
        console.print("[cyan]Progress:[/cyan] 10% - Extracting frames...")

        try:
            await process.wait()
        except asyncio.CancelledError:
            process.kill()
            await process.wait()
            console.print("[yellow]Job cancelled[/yellow]")
            return {"cancelled": True}

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

        self._update_progress(100)
        console.print("[cyan]Progress:[/cyan] 100% - Complete!")

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

    async def _run_compose(self, input_params: dict[str, Any]) -> dict[str, Any]:
        """Run the frame composition job with progress output."""
        input_dir = input_params.get("input_dir")
        output_file = input_params.get("output_file")

        if not input_dir or not output_file:
            raise ValueError("input_dir and output_file are required")

        input_path = Path(DATA_DIR) / input_dir
        output_path = Path(DATA_DIR) / output_file

        if not input_path.exists():
            raise ValueError(f"Input directory not found: {input_path}")

        metadata_path = input_path / "metadata.json"
        if not metadata_path.exists():
            raise ValueError(f"Metadata file not found: {metadata_path}")

        from .job_runner import JobRunner

        runner = JobRunner(None, lambda: "running")
        metadata = runner._load_metadata(metadata_path)

        frame_files = sorted(
            input_path.glob("frame_*.png"), key=lambda p: int(p.stem.split("_")[1])
        )
        if not frame_files:
            raise ValueError(f"No frame files found in: {input_path}")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        input_pattern = str(input_path / "frame_%04d.png")

        console.print(
            f"[cyan]Starting composition:[/cyan] {input_dir} -> {output_file}"
        )

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
        console.print("[cyan]Progress:[/cyan] 10% - Composing frames...")

        try:
            await process.wait()
        except asyncio.CancelledError:
            process.kill()
            await process.wait()
            console.print("[yellow]Job cancelled[/yellow]")
            return {"cancelled": True}

        if process.returncode != 0:
            if process.stderr is not None:
                stderr = await process.stderr.read()
                error_msg = stderr.decode() if stderr else "Unknown error"
            else:
                error_msg = "Unknown error"
            raise RuntimeError(f"FFmpeg failed: {error_msg}")

        self._update_progress(100)
        console.print("[cyan]Progress:[/cyan] 100% - Complete!")

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


def run_cli_job(
    job_id: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Run a job synchronously for the CLI."""
    job_ref: dict[str, Any] = {
        "id": job_id,
        "status": JobStatus.RUNNING,
        "progress": 0,
        "input_params": params,
    }

    def get_status() -> str:
        return job_ref["status"]

    async def run_with_progress():
        runner = CliJobRunner(job_ref, get_status)
        try:
            result = await runner.run()
            if job_ref["status"] == JobStatus.RUNNING:
                job_ref["status"] = JobStatus.COMPLETED
                job_ref["result"] = result
        except Exception as e:
            job_ref["status"] = JobStatus.FAILED
            job_ref["error"] = str(e)

    original_handler = signal.signal(signal.SIGINT, signal.SIG_DFL)

    def signal_handler(sig, frame):
        job_ref["status"] = JobStatus.CANCELLED
        console.print("\n[yellow]Received interrupt, cancelling job...[/yellow]")
        sys.exit(130)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        asyncio.run(run_with_progress())
    finally:
        signal.signal(signal.SIGINT, original_handler)

    return job_ref


@app.command()
def run(
    job_id: Optional[str] = typer.Option(
        None,
        "--job-id",
        "-j",
        help="Job identifier (auto-generated if not provided)",
    ),
    job_type: JobType = typer.Option(
        ...,
        "--job-type",
        "-t",
        help="Job type: extract or compose",
    ),
    input_file: Optional[str] = typer.Option(
        None,
        "--input",
        "-i",
        help="Input video file path (for extract)",
    ),
    output_dir: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory for frames or output video file",
    ),
    input_dir: Optional[str] = typer.Option(
        None,
        "--input-dir",
        help="Input directory containing frames (for compose)",
    ),
    output_file: Optional[str] = typer.Option(
        None,
        "--output-file",
        help="Output video file path (for compose)",
    ),
) -> None:
    """
    Run a video processing job (extract or compose).

    Extract: Extract frames from a video to PNG images.
        video-processing-job-service run -t extract -i video.mp4 -o output_frames

    Compose: Create a video from PNG frames.
        video-processing-job-service run -t compose -i output_frames -o composed.mp4
    """
    import uuid

    if job_id is None:
        job_id = str(uuid.uuid4())

    if job_type == JobType.EXTRACT:
        if not input_file or not output_dir:
            console.print("[red]Error:[/red] extract requires --input and --output")
            raise typer.Exit(code=1)
        param_dict = {
            "job_type": job_type.value,
            "input_file": input_file,
            "output_dir": output_dir,
        }
        console.print(f"[bold]Starting extract job:[/bold] {job_id}")
    elif job_type == JobType.COMPOSE:
        if not input_dir or not output_file:
            console.print(
                "[red]Error:[/red] compose requires --input-dir and --output-file"
            )
            raise typer.Exit(code=1)
        param_dict = {
            "job_type": job_type.value,
            "input_dir": input_dir,
            "output_file": output_file,
        }
        console.print(f"[bold]Starting compose job:[/bold] {job_id}")
    else:
        console.print(f"[red]Error:[/red] unknown job type: {job_type}")
        raise typer.Exit(code=1)

    job_result = run_cli_job(job_id, param_dict)

    if job_result["status"] == JobStatus.COMPLETED:
        console.print("[green]Job completed successfully![/green]")
        console.print(f"Result: {job_result.get('result')}")
    elif job_result["status"] == JobStatus.CANCELLED:
        console.print("[yellow]Job was cancelled[/yellow]")
        raise typer.Exit(code=130)
    else:
        console.print(f"[red]Job failed:[/red] {job_result.get('error')}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
