import logging
import asyncio
from src.pipeline.manager import FashionPipeline
from src.utils.logger import setup_logging

# Configure Logging
logger = setup_logging()

async def main():
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
        winners = await engine.run_async(**inputs)
        
        print("\n🏆 FINAL SELECTION:")
        for artifact, score, _ in winners:
            print(f"Video: {artifact.file_path} | Score: {score.overall}")
            
    except Exception as e:
        logging.critical(f"🔥 Critical Failure: {e}")

if __name__ == "__main__":
    asyncio.run(main())