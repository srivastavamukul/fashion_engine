import time
import pytest
from fastapi.testclient import TestClient
from api.server import app

# Setup client with API Key for all requests if possible, or per request
client = TestClient(app)
AUTH_HEADERS = {"x-api-key": "dev-secret-key"}

def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "Fashion Engine API is running"

def test_static_web_serving():
    # If index.html exists, / should serve it (200 OK)
    # If not, it might return 404, but we check if module loading worked
    response = client.get("/")
    # We expect 200 if web dir exists, else 404. 
    # Since we created web/, it should be 200.
    assert response.status_code == 200

def test_image_upload():
    # Mock file upload
    file_content = b"fake image content"
    files = {"file": ("test.jpg", file_content, "image/jpeg")}
    response = client.post("/upload", files=files, headers=AUTH_HEADERS)
    
    assert response.status_code == 200
    data = response.json()
    assert "url" in data
    assert data["filename"].endswith(".jpg")

def test_job_submission_and_polling():
    # 1. Submit
    payload = {
        "product_name": "Test Product",
        "category": "hoodie",
        "features": ["Blue"],
        "images": [] # No images for simple test
    }
    response = client.post("/jobs", json=payload, headers=AUTH_HEADERS)
    assert response.status_code == 200
    job_id = response.json()["job_id"]
    assert job_id

    # 2. Poll until complete
    # Since it uses background tasks, we can't easily wait in sync test without sleep 
    # unless using TestClient with a context manager or explicit sleep
    # But for unit/integ test speed, we might just check it exists
    
    response = client.get(f"/jobs/{job_id}")
    assert response.status_code == 200
    assert response.json()["status"] in ["pending", "processing", "completed"]

    # Optional: wait a bit to see completion (Mock generator is fast)
    max_retries = 10
    for _ in range(max_retries):
        response = client.get(f"/jobs/{job_id}")
        status = response.json()["status"]
        if status in ["completed", "failed"]:
            break
        time.sleep(0.5)
    
    # Assert final success
    assert status == "completed"
    result = response.json()["result"]
    assert result["total_returned"] > 0
