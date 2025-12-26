import time
import logging
from functools import wraps

logger = logging.getLogger("FashionEngine")

def smart_retry(retries=3, delay=1, backoff=2):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            for i in range(retries):
                try:
                    return func(*args, **kwargs)
                except (ConnectionError, TimeoutError) as e:
                    logger.warning(f"⚠️ [Retry {i+1}/{retries}] Transient error: {e}. Waiting {current_delay}s...")
                    time.sleep(current_delay)
                    current_delay *= backoff
                except (ValueError, RuntimeError, TypeError) as e:
                    logger.critical(f"🛑 Non-retryable failure detected: {e}")
                    raise e
                except Exception as e:
                    logger.error(f"⚠️ Unexpected error: {e}. Retrying...")
                    time.sleep(current_delay)
            logger.error(f"❌ Operation failed after {retries} attempts.")
            raise ConnectionError("Max retries exceeded")
        return wrapper
    return decorator