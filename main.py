import asyncio
import logging

from src.pipeline.manager import FashionPipeline
from src.utils.logger import setup_logging

# Configure Logging
logger = setup_logging()

async def main():
    """
    Main entry point for the Fashion Engine.
    """
    # 1. Initialize
    # Config is handled internally via settings, can be overridden here if needed
    engine = FashionPipeline()

    # 2. Input Data (Simulating UI or API Request)
    inputs = {
        "product_name": "Resilience Hoodie",
        "category": "hoodie",
        "features": ["Text: 'I AM STRONGER'", "Logo on chest"],
        "images": ["uploads/front.jpg"], 
    }

    # 3. Execute
    try:
        logger.info("🚀 Starting Engine Main Loop...")
        winners = await engine.run_async(**inputs)

        logger.info("🏆 FINAL SELECTION:")
        for res in winners:
            logger.info(f"Video: {res.artifact.file_path} | Score: {res.score.overall}")

    except Exception as e:
        logger.critical(f"🔥 Critical Failure: {e}", exc_info=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Captured KeyboardInterrupt. Exiting...")
