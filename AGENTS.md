# FFmpeg Service

FastAPI service for extracting video frames to PNG images using FFmpeg.

## Features

- FastAPI-based REST API
- Job management (start, status, cancel)
- Single job at a time (returns 409 Conflict if a job is already running)
- Progress tracking
- Health check endpoint
- Pydantic models for data validation
- Docker support
- Unit and integration tests
- Pre-commit hook (runs lint, type check, and unit tests on commit)

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
  - `POST /job` - Start a frame extraction job
  - `GET /job` - Get current job status
  - `POST /job/cancel` - Cancel running job

- **models.py**: Pydantic models including `Job`, `JobStatus`, `StartJobRequest`, `ExtractFramesRequest`

- **job_runner.py**: FFmpeg job execution. Extracts all frames from video to PNG files.

- **cli.py**: Command-line interface for running frame extraction without Docker. Use `-i` for input file and `-o` for output directory.

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

The job accepts these input parameters:
- `input_file` (required): Path to input video file (e.g., `data/input.mp4`)
- `output_dir` (required): Directory for output PNG frames (e.g., `data/output_frames`)

## HTTP Interface

### Health Check

- `GET /health`
  - **Response**:
    ```json
    {
      "status": "healthy|unhealthy|degraded",
      "message": "string",
      "timestamp": "ISO timestamp",
      "service_name": "video-composer-job-service"
    }
    ```

### Job Management

- `POST /job`
  - **Request Body**:
    ```json
    {
      "job_id": "string",
      "input_params": {
        "input_file": "data/input.mp4",
        "output_dir": "data/output_frames"
      }
    }
    ```
  - **Response**: Job object with status, progress, timestamps
  - **Errors**: 
    - 400 if missing input_file or output_dir
    - 400 if input file doesn't exist
    - 409 if job already running

- `GET /job`
  - **Response**: Job object or `null` if no job exists
    ```json
    {
      "id": "string",
      "status": "running|completed|failed|cancelled",
      "progress": 0-100,
      "result": { "completed": true, "frame_count": 300 },
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
    "input_params": {
      "input_file": "data/input.mp4",
      "output_dir": "data/output_frames"
    }
  }'

# 3. Check status
curl http://localhost:8001/job

# 4. Frames will be at data/output_frames/frame_0001.png, etc.
```

## CLI

Run frame extraction without Docker (requires ffmpeg installed locally):

```bash
make run-cli

# Or directly:
uv run python -m src.cli run -i video.mp4 -o output_frames
```

### CLI Options

- `-i, --input` (required): Input video file path
- `-o, --output` (required): Output directory for PNG frames
- `-j, --job-id`: Job identifier (auto-generated if not provided)

## Dockerfile Requirements

The Dockerfile must include:
```dockerfile
RUN make install && \
    make lint && \
    make check && \
    make test-unit
```
