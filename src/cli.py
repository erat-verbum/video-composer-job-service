import asyncio
import signal
import sys
from typing import Any, Callable, Optional

import typer
from rich.console import Console

from .job_runner import JobRunner
from .models import JobStatus

app = typer.Typer(
    name="video-processing-job-service",
    help="CLI for extracting video frames without Docker",
)
console = Console()


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
        """Run the frame extraction job with progress output."""
        input_params = self._job_ref.get("input_params", {}) if self._job_ref else {}
        input_file = input_params.get("input_file")
        output_dir = input_params.get("output_dir")

        if not input_file or not output_dir:
            raise ValueError("input_file and output_dir are required")

        from pathlib import Path
        import os

        input_path = Path(input_file)
        if not input_path.exists():
            raise ValueError(f"Input file not found: {input_file}")

        os.makedirs(output_dir, exist_ok=True)

        output_pattern = os.path.join(output_dir, "frame_%04d.png")

        console.print(f"[cyan]Starting extraction:[/cyan] {input_file} -> {output_dir}")

        process = await asyncio.create_subprocess_exec(
            "ffmpeg",
            "-i",
            str(input_path),
            "-y",
            output_pattern,
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

        frame_files = list(Path(output_dir).glob("frame_*.png"))
        frame_count = len(frame_files)

        self._update_progress(100)
        console.print("[cyan]Progress:[/cyan] 100% - Complete!")

        return {
            "completed": True,
            "input_file": input_file,
            "output_dir": output_dir,
            "frame_count": frame_count,
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
    input_file: str = typer.Option(
        ...,
        "--input",
        "-i",
        help="Input video file path",
    ),
    output_dir: str = typer.Option(
        ...,
        "--output",
        "-o",
        help="Output directory for PNG frames",
    ),
) -> None:
    """
    Extract frames from a video file to PNG images.

    Example:

        video-processing-job-service run -i video.mp4 -o output_frames
    """
    import uuid

    from .models import JobStatus

    if job_id is None:
        job_id = str(uuid.uuid4())

    param_dict = {
        "input_file": input_file,
        "output_dir": output_dir,
    }

    console.print(f"[bold]Starting job:[/bold] {job_id}")

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
