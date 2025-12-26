# Antigravity Fashion Engine

**Premium AI Video Generation Pipeline for Fashion Campaigns.**

This project is a high-performance engine that transforms product images and descriptions into high-fidelity fashion videos. It features a robust async backend, multi-model support (Stability/Luma/Runway), fine-tuning capabilities, and an AI-powered evaluation system.

## Table of Contents
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Quick Start](#quick-start)
- [Advanced Usage](#advanced-usage)
  - [Fine-Tuning](#fine-tuning)
  - [AI Evaluation](#ai-evaluation)
- [API Documentation](#api-documentation)
- [Testing](#testing)

## Features
- **Multi-Model Pipeline**: Parallel generation using Stability AI, Luma Dream Machine, and Runway Gen-3 (simulated interfaces included).
- **Fine-Tuning API**: Upload custom datasets to train LoRA adapters for specific styles.
- **AI Quality Audit**: Automated Vision-Language Models (GPT-4o/VLLM) score videos on Realism, Brand Alignment, and Product Visibility.
- **Overfitting Protection**: Smart training architecture with Early Stopping and optimized hyperparameters.
- **Premium Web UI**: Dark-mode dashboard with real-time logging and detailed quality metrics.

## Prerequisites
- Python 3.10+
- (Optional) Docker & Docker Compose
- (Optional) OpenAI API Key (for Vision Evaluation)

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/antigravity/fashion-engine.git
   cd fashion_engine
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration
The engine is configured via Environment Variables. You can set these in your shell or a `.env` file.

| Variable | Description | Default |
|----------|-------------|---------|
| `FASHION_OUTPUT_ROOT` | Directory for generated videos | `api_runs` |
| `FASHION_ENABLE_STABILITY` | Enable Stability AI Generator | `False` |
| `FASHION_ENABLE_VISION_EVALUATOR` | Enable GPT-4o Scoring | `False` |
| `OPENAI_API_KEY` | Key for Vision Evaluation | `None` |
| `STABILITY_API_KEY` | Key for Stability AI | `None` |

## Quick Start

1. **Start the Server**:
   ```bash
   python run_server.py
   ```
   The API will launch at `http://localhost:8000`.

2. **Access the Web UI**:
   Open [http://localhost:8000](http://localhost:8000) in your browser.
   - Upload a product image.
   - Enter a prompt (e.g., "Silk Summer Dress").
   - Click **GENERATE CAMPAIGN**.

## Advanced Usage

### Fine-Tuning
Train a custom model on your brand's dataset.
1. Prepare a `.zip` file containing 10-50 high-quality product images.
2. Call the Training Endpoint:
   ```bash
   curl -X POST "http://localhost:8000/train" \
        -H "x-api-key: dev-secret-key" \
        -F "file=@./my_dataset.zip" \
        -F "trigger_word=sks_dress"
   ```
3. The engine simulates a training loop with **Early Stopping**.

### AI Evaluation
To enable "AI Eyes" that watch and score your videos:
```bash
export FASHION_ENABLE_VISION_EVALUATOR=true
export OPENAI_API_KEY=sk-your-key
python run_server.py
```
Generated videos will include detailed metrics: **Realism, Brand Alignment, Consistency, Motion**.

## API Documentation

### POST `/jobs` (Generate)
```json
{
  "product_name": "Test Product",
  "category": "hoodie",
  "features": ["Blue fabric"],
  "images": ["uploads/front.jpg"]
}
```

### GET `/jobs/{job_id}` (Poll)
Returns status and detailed metrics:
```json
{
  "status": "completed",
  "result": {
    "videos": [
      {
        "score": 9.2,
        "metrics": {
            "realism": 9.5,
            "brand_alignment": 9.0
        },
        "file_path": "/artifacts/video.mp4"
      }
    ]
  }
}
```

## Testing
Run the comprehensive test suite to verify all modules:
```bash
python -m pytest tests/
```
