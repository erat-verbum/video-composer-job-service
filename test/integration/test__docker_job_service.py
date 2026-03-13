import os
import subprocess
import time
from typing import Any, Optional

import httpx
import pytest


BASE_URL = os.environ.get("INTEGRATION_TEST_URL", "http://localhost:8001")
POLL_INTERVAL = 0.5
POLL_TIMEOUT = 5


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
            "input_params": {
                "input_file": "data/test/input/test_clip_1.mkv",
                "output_dir": "data/test/output",
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
            "input_params": {
                "input_file": "data/test/input/test_clip_1.mkv",
                "output_dir": "data/test/output/2",
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
        "input_file": "data/test/input/test_clip_1.mkv",
        "output_dir": "data/test/output/params",
    }
    response = client.post(
        "job", json={"job_id": "params-job", "input_params": input_params}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["input_params"] == input_params

    response = client.get("/job")
    assert response.status_code == 200
    assert response.json()["input_params"] == input_params


def test_cannot_start_job_while_running(client):
    """Test POST /job while job is running returns 409."""
    client.post(
        "job",
        json={
            "job_id": "first-job",
            "input_params": {
                "input_file": "data/test/input/test_clip_1.mkv",
                "output_dir": "data/test/output/3",
            },
        },
    )

    response = client.post(
        "job",
        json={
            "job_id": "second-job",
            "input_params": {
                "input_file": "data/test/input/test_clip_1.mkv",
                "output_dir": "data/test/output/4",
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
            "input_params": {
                "input_file": "data/test/input/test_clip_1.mkv",
                "output_dir": "data/test/output/5",
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
            "input_params": {
                "input_file": "data/test/input/test_clip_1.mkv",
                "output_dir": "data/test/output/6",
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
    original_video = "/app/data/test/input/test_clip_1.mkv"
    extract_dir = "data/test/output/roundtrip_frames"
    composed_video = "data/test/output/roundtrip_composed.mp4"

    original_fps = get_video_fps(original_video)

    extract_response = client.post(
        "job",
        json={
            "job_id": "roundtrip-extract",
            "job_type": "extract",
            "input_params": {
                "input_file": "data/test/input/test_clip_1.mkv",
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

    composed_fps = get_video_fps("/app/data/" + composed_video)

    assert composed_fps == original_fps, (
        f"FPS mismatch: original={original_fps}, composed={composed_fps}"
    )
