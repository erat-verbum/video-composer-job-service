import os
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

import httpx
import pytest

DATA_PATH = Path(__file__).parent.parent.parent / "data"
POLL_INTERVAL = 0.5
POLL_TIMEOUT = 5


BASE_URL = os.environ.get("INTEGRATION_TEST_URL", "http://localhost:8001")


def is_video_composer_job_service(client: httpx.Client) -> bool:
    """Check if the service is the video-processing-job-service."""
    try:
        response = client.get("/health")
        if response.status_code == 200:
            data = response.json()
            return data.get("service_name") == "video-processing-job-service"
    except Exception:
        pass
    return False


@pytest.fixture
def client():
    return httpx.Client(base_url=BASE_URL, timeout=30.0)


@pytest.fixture(autouse=True)
def check_service(client):
    """Skip tests if not running video-processing-job-service."""
    if not is_video_composer_job_service(client):
        pytest.skip("Not running video-processing-job-service container")
    yield


@pytest.fixture(autouse=True)
def reset_job_state(client):
    """Ensure no running job before each test."""
    response = client.get("/job")
    if response.status_code == 200 and response.json() is not None:
        job = response.json()
        if job["status"] == "running":
            client.post("/job/cancel")
    yield


def wait_for_job_completion(
    client: httpx.Client, job_id: str
) -> Optional[dict[str, Any]]:
    """Poll GET /job until status is terminal or timeout."""
    start_time = time.time()
    job: Optional[dict[str, Any]] = None
    while time.time() - start_time < POLL_TIMEOUT:
        response = client.get("/job")
        assert response.status_code == 200
        job = response.json()
        if job is None:
            return None
        if job["status"] in ("completed", "failed", "cancelled"):
            return job
        time.sleep(POLL_INTERVAL)
    return job


def test_health_check_in_container(client):
    """Test GET /health returns healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert data["service_name"] == "video-processing-job-service"


def test_get_job_empty_in_container(client):
    """Test GET /job returns null when no job exists."""
    response = client.get("/job")
    assert response.status_code == 200
    data = response.json()
    if data is not None:
        if data["status"] in ("running", "completed", "failed", "cancelled"):
            pytest.skip("Job from previous test still exists in memory")


def test_start_job_in_container(client):
    """Test POST /job starts a new job."""
    response = client.post(
        "job",
        json={
            "job_id": "test-job-1",
            "job_type": "extract",
            "input_params": {
                "input_file": "test/input/test_clip_1.mkv",
                "output_dir": "test/output",
            },
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "test-job-1"
    assert data["status"] == "running"
    assert data["progress"] == 0


def test_get_job_returns_running(client):
    """Test GET /job returns current running job with progress."""
    client.post(
        "job",
        json={
            "job_id": "running-job",
            "job_type": "extract",
            "input_params": {
                "input_file": "test/input/test_clip_1.mkv",
                "output_dir": "test/output/2",
            },
        },
    )

    response = client.get("/job")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "running-job"
    assert data["status"] == "running"
    assert "progress" in data
    assert data["progress"] >= 0


def test_job_with_input_params(client):
    """Test POST /job with input_params stores and returns them."""
    input_params = {
        "input_file": "test/input/test_clip_1.mkv",
        "output_dir": "test/output/params",
    }
    response = client.post(
        "job",
        json={
            "job_id": "params-job",
            "job_type": "extract",
            "input_params": input_params,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["input_params"]["input_file"] == input_params["input_file"]
    assert data["input_params"]["output_dir"] == input_params["output_dir"]

    response = client.get("/job")
    assert response.status_code == 200
    stored_params = response.json()["input_params"]
    assert stored_params["input_file"] == input_params["input_file"]
    assert stored_params["output_dir"] == input_params["output_dir"]


def test_cannot_start_job_while_running(client):
    """Test POST /job while job is running returns 409."""
    client.post(
        "job",
        json={
            "job_id": "first-job",
            "job_type": "extract",
            "input_params": {
                "input_file": "test/input/test_clip_1.mkv",
                "output_dir": "test/output/3",
            },
        },
    )

    response = client.post(
        "job",
        json={
            "job_id": "second-job",
            "job_type": "extract",
            "input_params": {
                "input_file": "test/input/test_clip_1.mkv",
                "output_dir": "test/output/4",
            },
        },
    )
    assert response.status_code == 409
    assert "already running" in response.json()["detail"]


def test_cancel_running_job(client):
    """Test POST /job/cancel cancels running job."""
    client.post(
        "job",
        json={
            "job_id": "cancel-job",
            "job_type": "extract",
            "input_params": {
                "input_file": "test/input/test_clip_1.mkv",
                "output_dir": "test/output/5",
            },
        },
    )

    response = client.post("/job/cancel")
    assert response.status_code == 200

    response = client.get("/job")
    assert response.status_code == 200
    job = response.json()
    assert job["status"] == "cancelled"
    assert job["finished_at"] is not None


def test_cancel_no_job(client):
    """Test POST /job/cancel when no running job returns 404 or 400."""
    response = client.post("/job/cancel")
    assert response.status_code in (400, 404)


def test_cancel_completed_job(client):
    """Test POST /job/cancel on completed job returns 400."""
    client.post(
        "job",
        json={
            "job_id": "already-done",
            "job_type": "extract",
            "input_params": {
                "input_file": "test/input/test_clip_1.mkv",
                "output_dir": "test/output/6",
            },
        },
    )

    wait_for_job_completion(client, "already-done")

    response = client.post("/job/cancel")
    assert response.status_code == 400


def get_video_fps(video_path: str) -> float:
    """Get the FPS of a video file using ffprobe."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=r_frame_rate",
            "-of",
            "json",
            video_path,
        ],
        capture_output=True,
        text=True,
    )
    import json

    output = json.loads(result.stdout)
    fps_str = output["streams"][0]["r_frame_rate"]
    if "/" in fps_str:
        num, den = fps_str.split("/")
        return float(num) / float(den)
    return float(fps_str)


def test_roundtrip_extract_and_compose(client):
    """Test extracting frames and composing them back preserves FPS."""
    original_video = "/home/nelson/Development/video-processing-job-service/data/test/input/test_clip_1.mkv"
    extract_dir = "test/output/roundtrip_frames"
    composed_video = "test/output/roundtrip_composed.mp4"

    original_fps = get_video_fps(original_video)

    extract_response = client.post(
        "job",
        json={
            "job_id": "roundtrip-extract",
            "job_type": "extract",
            "input_params": {
                "input_file": "test/input/test_clip_1.mkv",
                "output_dir": extract_dir,
            },
        },
    )
    assert extract_response.status_code == 200

    extract_job = wait_for_job_completion(client, "roundtrip-extract")
    assert extract_job is not None
    assert extract_job["status"] == "completed"
    assert extract_job["result"]["frame_count"] > 0

    compose_response = client.post(
        "job",
        json={
            "job_id": "roundtrip-compose",
            "job_type": "compose",
            "input_params": {
                "input_dir": extract_dir,
                "output_file": composed_video,
            },
        },
    )
    assert compose_response.status_code == 200

    compose_job = wait_for_job_completion(client, "roundtrip-compose")
    assert compose_job is not None
    assert compose_job["status"] == "completed"

    composed_fps = get_video_fps(
        "/home/nelson/Development/video-processing-job-service/data/" + composed_video
    )

    assert composed_fps == original_fps, (
        f"FPS mismatch: original={original_fps}, composed={composed_fps}"
    )


def test_rotated_video_roundtrip(client):
    """Test extracting and composing a rotated video preserves correct dimensions."""
    rotated_video = "test/input/test_clip_3.mkv"
    extract_dir = "test/output/rotated_frames"
    composed_video = "test/output/rotated_composed.mp4"

    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "json",
            "/home/nelson/Development/video-processing-job-service/data/"
            + rotated_video,
        ],
        capture_output=True,
        text=True,
    )
    import json

    rotated_info = json.loads(result.stdout)
    rotated_width = rotated_info["streams"][0]["width"]
    rotated_height = rotated_info["streams"][0]["height"]

    assert rotated_width == 576
    assert rotated_height == 720

    extract_response = client.post(
        "job",
        json={
            "job_id": "rotated-extract",
            "job_type": "extract",
            "input_params": {
                "input_file": rotated_video,
                "output_dir": extract_dir,
            },
        },
    )
    assert extract_response.status_code == 200

    extract_job = wait_for_job_completion(client, "rotated-extract")
    assert extract_job is not None
    assert extract_job["status"] == "completed"
    assert extract_job["result"]["frame_count"] > 0

    metadata_response = client.get("/job")
    assert metadata_response.status_code == 200

    metadata_file = str(DATA_PATH / extract_dir / "metadata.json")
    with open(metadata_file) as f:
        metadata = json.load(f)

    assert metadata["width"] == 576
    assert metadata["height"] == 720
    assert metadata["display_width"] == 404
    assert metadata["display_height"] == 720

    compose_response = client.post(
        "job",
        json={
            "job_id": "rotated-compose",
            "job_type": "compose",
            "input_params": {
                "input_dir": extract_dir,
                "output_file": composed_video,
            },
        },
    )
    assert compose_response.status_code == 200

    compose_job = wait_for_job_completion(client, "rotated-compose")
    assert compose_job is not None
    assert compose_job["status"] == "completed"

    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "json",
            "/home/nelson/Development/video-processing-job-service/data/"
            + composed_video,
        ],
        capture_output=True,
        text=True,
    )
    composed_info = json.loads(result.stdout)
    composed_width = composed_info["streams"][0]["width"]
    composed_height = composed_info["streams"][0]["height"]

    assert composed_width == 404, f"Expected width 404, got {composed_width}"
    assert composed_height == 720, f"Expected height 720, got {composed_height}"


def test_auto_crop_detects_and_applies_black_bar_crop(client):
    """Test that auto-crop detects black bars and crops them from frames."""
    video_with_black_bars = "test/input/test_clip_2.mkv"
    extract_dir = "test/output/autocrop_frames"

    extract_response = client.post(
        "job",
        json={
            "job_id": "autocrop-test",
            "job_type": "extract",
            "input_params": {
                "input_file": video_with_black_bars,
                "output_dir": extract_dir,
                "auto_crop": True,
            },
        },
    )
    assert extract_response.status_code == 200

    extract_job = wait_for_job_completion(client, "autocrop-test")
    assert extract_job is not None
    assert extract_job["status"] == "completed"
    assert extract_job["result"]["frame_count"] > 0

    import json

    metadata_file = str(DATA_PATH / extract_dir / "metadata.json")
    with open(metadata_file) as f:
        metadata = json.load(f)

    assert metadata["crop_width"] is not None
    assert metadata["crop_height"] is not None
    assert metadata["crop_x"] is not None
    assert metadata["crop_y"] is not None

    assert metadata["crop_height"] < metadata["height"], (
        f"Crop height ({metadata['crop_height']}) should be less than "
        f"original height ({metadata['height']})"
    )

    assert metadata["display_height"] == metadata["crop_height"], (
        f"Display height ({metadata['display_height']}) should equal crop height ({metadata['crop_height']})"
    )

    assert metadata["display_height"] < metadata["height"], (
        f"Display height ({metadata['display_height']}) should be less than "
        f"original height ({metadata['height']}) due to cropping"
    )

    assert metadata["display_width"] == 1024, (
        f"Display width should always be 1024 for this SAR, got {metadata['display_width']}"
    )

    frame_files = sorted((DATA_PATH / extract_dir / "frame").glob("frame_*.png"))
    assert len(frame_files) > 0, "No frames extracted"

    frame_path = frame_files[0]
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "json",
            str(frame_path),
        ],
        capture_output=True,
        text=True,
    )
    frame_info = json.loads(result.stdout)
    frame_width = frame_info["streams"][0]["width"]
    frame_height = frame_info["streams"][0]["height"]

    assert frame_width == metadata["display_width"], (
        f"Frame width {frame_width} should match display_width {metadata['display_width']}"
    )
    assert frame_height == metadata["display_height"], (
        f"Frame height {frame_height} should match display_height {metadata['display_height']}"
    )


def test_bitmap_subtitle_roundtrip(client):
    """Test that bitmap subtitles (DVD subtitles) are correctly preserved in roundtrip."""
    original_video = "test/input/test_clip_1.mkv"
    extract_dir = "test/output/bitmap_subtitle_frames"
    composed_video = "test/output/bitmap_subtitle_composed.mp4"

    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "s",
            "-show_entries",
            "stream=index,codec_name",
            "-of",
            "json",
            "/home/nelson/Development/video-processing-job-service/data/"
            + original_video,
        ],
        capture_output=True,
        text=True,
    )
    import json

    original_subtitle_info = json.loads(result.stdout)
    original_subtitle_streams = original_subtitle_info.get("streams", [])
    assert len(original_subtitle_streams) > 0, (
        "Original video should have subtitle streams"
    )
    original_codec = original_subtitle_streams[0]["codec_name"]
    assert original_codec == "dvd_subtitle", (
        f"Expected dvd_subtitle, got {original_codec}"
    )

    extract_response = client.post(
        "job",
        json={
            "job_id": "bitmap-subtitle-extract",
            "job_type": "extract",
            "input_params": {
                "input_file": original_video,
                "output_dir": extract_dir,
            },
        },
    )
    assert extract_response.status_code == 200

    extract_job = wait_for_job_completion(client, "bitmap-subtitle-extract")
    assert extract_job is not None
    assert extract_job["status"] == "completed"
    assert extract_job["result"]["subtitle_track_count"] >= 1

    subtitle_files = sorted((DATA_PATH / extract_dir / "subtitle").glob("subtitle_*"))
    assert len(subtitle_files) > 0, "No subtitle files extracted"
    subtitle_file = subtitle_files[0]
    assert subtitle_file.suffix in (".sub", ".idx"), (
        f"Expected .sub or .idx file, got {subtitle_file.suffix}"
    )

    metadata_file = str(DATA_PATH / extract_dir / "metadata.json")
    with open(metadata_file) as f:
        metadata = json.load(f)

    assert len(metadata["subtitle_tracks"]) >= 1
    extracted_track = metadata["subtitle_tracks"][0]
    assert extracted_track["codec"] == "dvd_subtitle"

    compose_response = client.post(
        "job",
        json={
            "job_id": "bitmap-subtitle-compose",
            "job_type": "compose",
            "input_params": {
                "input_dir": extract_dir,
                "output_file": composed_video,
            },
        },
    )
    assert compose_response.status_code == 200

    compose_job = wait_for_job_completion(client, "bitmap-subtitle-compose")
    assert compose_job is not None
    assert compose_job["status"] == "completed"

    composed_result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "s",
            "-show_entries",
            "stream=index,codec_name",
            "-of",
            "json",
            "/home/nelson/Development/video-processing-job-service/data/"
            + composed_video,
        ],
        capture_output=True,
        text=True,
    )
    composed_subtitle_info = json.loads(composed_result.stdout)
    composed_subtitle_streams = composed_subtitle_info.get("streams", [])
    assert len(composed_subtitle_streams) >= 1, (
        f"Composed video should have at least 1 subtitle stream, got {len(composed_subtitle_streams)}"
    )


def test_auto_crop_disabled_keeps_original_dimensions(client):
    """Test that auto_crop=false does not crop black bars."""
    video_with_black_bars = "test/input/test_clip_2.mkv"
    extract_dir = "test/output/no_autocrop_frames"

    extract_response = client.post(
        "job",
        json={
            "job_id": "no-autocrop-test",
            "job_type": "extract",
            "input_params": {
                "input_file": video_with_black_bars,
                "output_dir": extract_dir,
                "auto_crop": False,
            },
        },
    )
    assert extract_response.status_code == 200

    extract_job = wait_for_job_completion(client, "no-autocrop-test")
    assert extract_job is not None
    assert extract_job["status"] == "completed"

    import json

    metadata_file = str(DATA_PATH / extract_dir / "metadata.json")
    with open(metadata_file) as f:
        metadata = json.load(f)

    assert metadata.get("crop_width") is None
    assert metadata.get("crop_height") is None
    assert metadata.get("crop_x") is None
    assert metadata.get("crop_y") is None

    assert metadata["display_width"] == 1024, (
        f"Display width should always be 1024 for this SAR, got {metadata['display_width']}"
    )

    frame_files = sorted((DATA_PATH / extract_dir / "frame").glob("frame_*.png"))
    assert len(frame_files) > 0, "No frames extracted"

    frame_path = frame_files[0]
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "json",
            str(frame_path),
        ],
        capture_output=True,
        text=True,
    )
    frame_info = json.loads(result.stdout)
    frame_width = frame_info["streams"][0]["width"]
    frame_height = frame_info["streams"][0]["height"]

    assert frame_width == metadata["display_width"], (
        f"Frame width {frame_width} should match display_width {metadata['display_width']}"
    )
    assert frame_height == metadata["display_height"], (
        f"Frame height {frame_height} should match display_height {metadata['display_height']}"
    )
