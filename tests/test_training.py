import io
import time
import zipfile
import pytest
from fastapi.testclient import TestClient
from api.server import app

client = TestClient(app)
AUTH_HEADERS = {"x-api-key": "dev-secret-key"}

def create_valid_zip():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        # Create dummy image content
        zf.writestr("img1.jpg", b"fake image content")
        zf.writestr("img2.png", b"fake image content")
    return buffer.getvalue()

def create_invalid_zip():
    return b"not a zip file"

def create_empty_folder_zip():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("dummy.txt", b"txt")
    return buffer.getvalue()

def test_training_submission_success():
    zip_content = create_valid_zip()
    files = {"file": ("dataset.zip", zip_content, "application/zip")}
    data = {"trigger_word": "sks", "generator_name": "mock"}
    
    response = client.post("/train", files=files, data=data, headers=AUTH_HEADERS)
    assert response.status_code == 200
    
    job_id = response.json()["job_id"]
    assert job_id
    
    # Poll for completion
    max_retries = 10
    for _ in range(max_retries):
        resp = client.get(f"/jobs/{job_id}")
        status = resp.json()["status"]
        if status in ["training_completed", "training_failed"]:
            break
        time.sleep(1)
        
    assert status == "training_completed"
    result = resp.json()["result"]
    assert "adapter_id" in result

def test_training_invalid_zip():
    content = create_invalid_zip()
    files = {"file": ("dataset.zip", content, "application/zip")}
    data = {"trigger_word": "sks"}
    
    response = client.post("/train", files=files, data=data, headers=AUTH_HEADERS)
    assert response.status_code == 400
    assert "Invalid ZIP" in response.json()["detail"]

def test_training_no_images():
    # If extraction works but validate_dataset fails
    content = create_empty_folder_zip()
    files = {"file": ("dataset.zip", content, "application/zip")}
    data = {"trigger_word": "sks"}
    
    response = client.post("/train", files=files, data=data, headers=AUTH_HEADERS)
    # The endpoint returns 200 (Accepted), validation happens in background
    assert response.status_code == 200
    job_id = response.json()["job_id"]
    
    # Poll -> should fail
    for _ in range(5):
        resp = client.get(f"/jobs/{job_id}")
        if resp.json()["status"] == "training_failed":
            break
        time.sleep(1)
    
    as_json = resp.json()
    assert as_json["status"] == "training_failed"
    assert "No valid images" in as_json["error"]
