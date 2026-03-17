# FFmpeg Service

FastAPI service for extracting and composing video frames using FFmpeg.

## Features

### Extract Job
- Extracts frames from a video to `PNG` images
- Automatically detects and crops black bars from video (enabled by default)
  - Uses FFmpeg's `cropdetect` filter to find black borders
  - Recalculates display dimensions based on cropped internal resolution + SAR
- Saves metadata (resolution, display dimensions, frame rate, duration, crop info, etc.) for reconstitution
  - Accounts for sample aspect ratio (SAR) and rotation when calculating display dimensions
  - Stores output frames at display resolution with rotation baked in
- Extracts all audio tracks to separate files
  - Preserves original format: `aac`, `mp3`, `ac3`, `flac`, `opus`, `ogg`, `wav`, `m4a`, etc.
- Extracts all subtitle tracks to separate files
  - Text formats: `srt`, `ass`, `ssa`, `vtt`, `webvtt`, etc. (preserved as-is)
  - Bitmap formats: `dvd_subtitle`, `vobsub` (extracted to `.sub` + `.idx` files), `dvbsub`, `hdmv_pgs_subtitle` (extracted to `.sup` files)
  - Other text formats (jacosub, microdvd, mpl2, etc.) converted to `srt`
- Output:
  - `frame/` directory with `PNG` frames
  - `audio/` directory with audio tracks
  - `subtitle/` directory with subtitle tracks
  - `metadata.json` at root

### Compose Job
- Creates a video from `PNG` frames using saved metadata
  - Uses display dimensions from metadata (rotation is already baked into frames)
  - No need to apply SAR or rotation when composing
- Re-muxes all extracted audio tracks into the output video
  - Preserves original formats
- Re-muxes all extracted subtitle tracks into the output video
  - Preserves original formats
- Output:
  - Reconstructed video file with all original audio/subtitle tracks

## Quick Start

```bash
make install  # Install dependencies
make run      # Start the service
```

## Commands

- `make install` - Install dependencies (uv venv/sync) and setup pre-commit hook
- `make lint` - Run linters (ruff) and auto-fix issues
- `make check` - Type checking (ty)
- `make test` - Run all tests
- `make test-unit` - Run unit tests only
- `make test-int` - Run integration tests only
- `make run` - Start the service with uvicorn
- `make run-cli` - Run frame extraction via CLI (requires ffmpeg installed locally)

## API Endpoints

- `GET /health` - Health check
- `POST /job` - Start a job (returns 409 if a job is already running)
- `GET /job` - Get job status
- `POST /job/cancel` - Cancel running job

## File Structure

```
service-name/
├── Dockerfile      # Container configuration
├── .venv/          # Virtual environment (created by uv)
├── pyproject.toml  # Project configuration
├── Makefile        # Common commands
├── src/
│   ├── main.py     # FastAPI application
│   ├── models.py   # Pydantic models and data structures
│   ├── job_runner.py  # Job execution logic
│   └── cli.py      # CLI for running without Docker
└── test/
    ├── unit/       # Unit tests
    │   └── test__<name_of_file_being_tested>__<name_of_feature_being_tested>.py
    └── integration/ # Integration tests
        └── test__<name_of_file_being_tested>__<name_of_feature_being_tested>.py
```

### Source Files

- **main.py**: FastAPI application with endpoints:
  - `GET /health` - Health check
  - `POST /job` - Start a frame extraction or composition job
  - `GET /job` - Get current job status
  - `POST /job/cancel` - Cancel running job

- **models.py**: Pydantic models including `Job`, `JobStatus`, `JobType`, `StartJobRequest`, `VideoMetadata`, `AudioTrack`, `SubtitleTrack`

- **job_runner.py**: FFmpeg job execution. Extracts frames from video to `PNG` files or composes video from frames. Saves metadata for reconstitution. Preserves all audio and subtitle tracks.

- **cli.py**: Command-line interface for running frame extraction or composition without Docker. Use `-t` for job type, `-i` for input file, `-o` for output directory.

## Service Components

- **Dockerfile**: Container configuration for the service
- **uv**: Package manager (installed in local `.venv`)
- **pre-commit**: Git hook framework (runs lint, type check, and unit tests on commit)
- **ruff**: Linting and formatting
- **ty**: Type checking
- **FastAPI**: Interface with other services
- **pyproject.toml**: Project configuration
- **Python `requests` library**: For making HTTP API calls to external services
- **Pydantic**: For data validation and serialization - Pydantic models should be defined for every non-simple object
- **Type annotations**: All method parameters and return types must be annotated for better code quality and IDE support
- **Docstrings**: Each method must include a docstring with: a description, Args section (parameter names, types, descriptions), Returns section (return type and description), and Raises section (exceptions and when they're raised). Format example:
  ```
  def method_name(self, param1: Type) -> ReturnType:
      """
      Brief description of the method.
      
      Args:
          param1 (Type): Description of parameter
      
      Returns:
          ReturnType: Description of return value
      
      Raises:
          ExceptionType: Description of when this exception is raised
      ```
- **Class method organization**: Public methods in a class should always be written at the bottom of the class AFTER all of the private methods (those starting with underscore). This improves code readability by grouping implementation details together.
- **Test naming**: Test files should be named:
  - `test__<name_of_file_being_tested>.py` for simple cases where the file contains tests for a single feature
  - `test__<name_of_file_being_tested>__<name_of_feature_being_tested>.py` when the file would become too large or contain tests for multiple distinct features

## Input Parameters

The job requires a `job_type` parameter to specify the operation:

### Extract Job
Extracts frames from a video to `PNG` images and saves metadata for reconstitution. Also extracts all audio and subtitle tracks.
- `job_type`: `"extract"` (required)
- `input_file` (required): Path to input video file (e.g., `data/input.mp4`)
- `output_dir` (required): Directory for output `PNG` frames (e.g., `data/output_frames`)
- `auto_crop` (optional): Whether to automatically crop black bars from video (default: `true`)

### Compose Job
Creates a video from `PNG` frames using saved metadata. Preserves all extracted audio and subtitle tracks.
- `job_type`: `"compose"` (required)
- `input_dir` (required): Directory containing `PNG` frames and `metadata.json` (e.g., `data/output_frames`)
- `output_file` (required): Path to output video file (e.g., `data/composed.mp4`)

All paths are relative to `/app/data/` in the container.

## HTTP Interface

### Health Check

- `GET /health`
  - **Response**:
    ```json
    {
      "status": "healthy|unhealthy|degraded",
      "message": "string",
      "timestamp": "ISO timestamp",
      "service_name": "video-processing-job-service"
    }
    ```

### Job Management

- `POST /job`
  - **Request Body**:
    ```json
    {
      "job_id": "string",
      "job_type": "extract" | "compose",
      "input_params": {
        // For extract:
        "input_file": "data/input.mp4",
        "output_dir": "data/output_frames",
        "auto_crop": true
        // OR for compose:
        "input_dir": "data/output_frames",
        "output_file": "data/composed.mp4"
      }
    }
    ```
  - **Response**: Job object with status, progress, timestamps
  - **Errors**: 
    - 400 if job_type missing
    - 400 if required params missing for job type
    - 400 if input file/directory doesn't exist
    - 409 if job already running

- `GET /job`
  - **Response**: Job object or `null` if no job exists
    ```json
    {
      "id": "string",
      "job_type": "extract" | "compose",
      "status": "running|completed|failed|cancelled",
      "progress": 0-100,
      "result": { "completed": true, "frame_count": 300, "audio_track_count": 2, "subtitle_track_count": 1 },
      "error": "string",
      "created_at": "ISO timestamp",
      "started_at": "ISO timestamp",
      "finished_at": "ISO timestamp"
    }
    ```

- `POST /job/cancel`
  - **Request Body**: `{}` (empty)
  - **Response**: `{"message": "Job cancelled"}`
  - **Errors**: 404 if no job, 400 if job not running

## Docker

The service includes a data volume mount at `/app/data` for input/output files.

```bash
# Build and run
make up-build

# The service will be available at http://localhost:8001

# Example: Extract frames from a video
# 1. Put video at data/input.mp4
# 2. Start job:
curl -X POST http://localhost:8001/job \
  -H "Content-Type: application/json" \
  -d '{
    "job_id": "extract-1",
    "job_type": "extract",
    "input_params": {
      "input_file": "data/input.mp4",
      "output_dir": "data/output_frames",
      "auto_crop": true
    }
  }'

# 3. Check status
curl http://localhost:8001/job

# 4. Frames will be at `data/output_frames/frame/frame_0001.png`, etc.
# 5. Audio tracks at `data/output_frames/audio/audio_0.aac`, etc.
# 6. Subtitle tracks at `data/output_frames/subtitle/subtitle_0.srt`, etc.
#    (bitmap subtitles: `.sub` + `.idx` files)
# 7. Metadata saved at `data/output_frames/metadata.json` (includes crop info if auto_crop enabled)

# Example: Compose a video from frames
# 1. Start compose job:
curl -X POST http://localhost:8001/job \
  -H "Content-Type: application/json" \
  -d '{
    "job_id": "compose-1",
    "job_type": "compose",
    "input_params": {
      "input_dir": "data/output_frames",
      "output_file": "data/composed.mp4"
    }
  }'
```

## CLI

Run video processing without Docker (requires ffmpeg installed locally):

```bash
make run-cli

# Or directly:
uv run python -m src.cli run -t extract -i video.mp4 -o output_frames

# Extract without auto-crop:
uv run python -m src.cli run -t extract -i video.mp4 -o output_frames --no-auto-crop

# Compose a video from frames:
uv run python -m src.cli run -t compose --input-dir output_frames -o composed.mp4
```

### CLI Options

- `-t, --job-type` (required): Job type: `extract` or `compose`
- `-i, --input`: Input video file path (for extract)
- `-o, --output`: Output directory for frames or output video file (for extract)
- `--input-dir`: Input directory containing frames (for compose)
- `--output-file`: Output video file path (for compose)
- `-j, --job-id`: Job identifier (auto-generated if not provided)
- `--auto-crop/--no-auto-crop`: Automatically crop black bars from video (default: enabled)

## Dockerfile Requirements

The Dockerfile must include:
```dockerfile
RUN make install && \
    make lint && \
    make check && \
    make test-unit
```
