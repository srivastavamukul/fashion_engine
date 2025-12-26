import uvicorn
import os
from src.config import settings

if __name__ == "__main__":
    # Ensure run directories exist
    os.makedirs("api_runs", exist_ok=True)
    
    print(f"🚀 Starting Fashion Engine API on {settings.api_host}:{settings.api_port}...")
    uvicorn.run(
        "api.server:app", 
        host=settings.api_host, 
        port=settings.api_port, 
        reload=True
    )
