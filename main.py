import uvicorn
from src.config import CUDA_VISIBLE_DEVICES, MODEL_NAME
from src.api import app

if __name__ == "__main__":
    print("Starting Spark TTS FastAPI server with WebSocket support...")
    print(f"Model: {MODEL_NAME}")
    print(f"Device: {CUDA_VISIBLE_DEVICES}")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8002,
        reload=False,
        ws_ping_interval=20,
        ws_ping_timeout=20,
        timeout_keep_alive=300
    )
