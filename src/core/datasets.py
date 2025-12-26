import os
import shutil
import zipfile
import uuid
from typing import List
from fastapi import UploadFile

DATASET_ROOT = "datasets"

def setup_datasets_dir():
    os.makedirs(DATASET_ROOT, exist_ok=True)

async def save_and_extract_dataset(file: UploadFile) -> str:
    """
    Saves the uploaded zip file, extracts it, and returns the path to the extracted folder.
    """
    setup_datasets_dir()
    
    # Create unique ID for this dataset upload
    upload_id = str(uuid.uuid4())
    upload_dir = os.path.join(DATASET_ROOT, upload_id)
    os.makedirs(upload_dir, exist_ok=True)
    
    zip_path = os.path.join(upload_dir, "dataset.zip")
    
    # Save Zip
    with open(zip_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # Extract
    extract_path = os.path.join(upload_dir, "extracted")
    os.makedirs(extract_path, exist_ok=True)
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
    except zipfile.BadZipFile:
        # Cleanup and raise
        shutil.rmtree(upload_dir)
        raise ValueError("Invalid ZIP file provided.")
        
    return extract_path

def validate_dataset(path: str) -> int:
    """
    Checks if dataset contains valid images. Returns count of valid images.
    """
    valid_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    count = 0
    for root, _, files in os.walk(path):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in valid_extensions:
                count += 1
    
    if count == 0:
        raise ValueError("No valid images found in dataset (jpg, png, webp).")
        
    return count
