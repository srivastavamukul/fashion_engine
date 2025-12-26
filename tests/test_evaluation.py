import sys
import pytest
from unittest.mock import MagicMock, patch

from src.core.models import VideoArtifact, Intent, Shot
from src.evaluation.vision import VisionQualityEvaluator
from src.evaluation import vision # To patch module level vars

@pytest.fixture
def mock_artifact():
    return VideoArtifact(
        file_path="dummy_vid.mp4",
        video_id="v1",
        seed=123,
        duration=1.0,
        model_used="test",
        prompt="A fashion video",
        prompt_id="p1"
    )

@pytest.fixture
def mock_intent():
    m = MagicMock()
    # Configure nested attributes for Scorer logic
    m.brand_identity.name = "TestBrand"
    m.brand_identity.vibe = "Cool"
    m.brand_identity.palette = ["Red"]
    m.guardrails.avoid = []
    m.meta.reference_image_paths = []
    # Configure Shot
    return m

@pytest.fixture
def mock_shot():
    s = MagicMock()
    s.focus_points = ["overall"]
    s.camera_action = "slow pan"
    return s

def test_vision_evaluator_extraction_failure(mock_artifact, mock_intent, mock_shot):
    evaluator = VisionQualityEvaluator()
    
    # Mock extract_frames to return empty
    with patch.object(evaluator, "_extract_frames", return_value=[]):
        # Should fallback to mock score (which returns random float > 0)
        # But wait, VisionQualityEvaluator calls super().evaluate() if extraction fails.
        # MockQualityEvaluator returns random score.
        score = evaluator.evaluate(mock_artifact, mock_intent, mock_shot, "prompt", [])
        assert score.overall >= 0

def test_vision_evaluator_success(mock_artifact, mock_intent, mock_shot):
    # Patch module level dependencies
    with patch.object(vision, "OpenAI", MagicMock()) as MockOpenAI, \
         patch.object(vision, "cv2", MagicMock()):
        
        # Setup mock client
        mock_client = MockOpenAI.return_value
        mock_response = MagicMock()
        mock_response.choices[0].message.content = '{"overall": 8.5, "realism": 9.0}'
        mock_client.chat.completions.create.return_value = mock_response

        evaluator = VisionQualityEvaluator()
        # Verify client was created
        assert evaluator.client is not None

        # Mock extraction
        with patch.object(evaluator, "_extract_frames", return_value=["base64img"]):
            score = evaluator.evaluate(mock_artifact, mock_intent, mock_shot, "prompt", [])
            
            assert score.overall == 8.5
            assert score.realism == 9.0

def test_vision_evaluator_api_failure(mock_artifact, mock_intent, mock_shot):
    with patch.object(vision, "OpenAI", MagicMock()) as MockOpenAI, \
         patch.object(vision, "cv2", MagicMock()):
         
        mock_client = MockOpenAI.return_value
        # Mock API error
        mock_client.chat.completions.create.side_effect = Exception("API Down")
        
        evaluator = VisionQualityEvaluator()

        with patch.object(evaluator, "_extract_frames", return_value=["base64img"]):
            score = evaluator.evaluate(mock_artifact, mock_intent, mock_shot, "prompt", [])
            
            # Should return 0.0 on failure (fallback to mock with broken random? no mock uses base config)
            # wait mock evaluator returns > 0 usually
            # But the Exception caught in vision.py returns a Zero-score object directly!
            assert score.overall == 0.0
