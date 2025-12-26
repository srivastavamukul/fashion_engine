import logging
from src.pipeline.manager import FashionPipeline

# Configure Logging Globally
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

if __name__ == "__main__":
    # 1. Initialize
    engine = FashionPipeline(output_root="campaign_runs")

    # 2. Input Data (Simulating UI)
    inputs = {
        "product_name": "Resilience Hoodie",
        "category": "hoodie",
        "features": ["Text: 'I AM STRONGER'", "Logo on chest"],
        "images": ["uploads/front.jpg"] # Ensure these exist or mock them
    }

    # 3. Execute
    try:
        print("🚀 Starting Engine...")
        winners = engine.run(**inputs)
        
        print("\n🏆 FINAL SELECTION:")
        for artifact, score, _ in winners:
            print(f"Video: {artifact.file_path} | Score: {score.overall}")
            
    except Exception as e:
        logging.critical(f"🔥 Critical Failure: {e}")