import os
import sys
import torch
from vllm import LLM
from huggingface_hub import snapshot_download
from .config import (
    MODEL_NAME,
    TOKENIZER_REPO,
    TOKENIZER_CACHE_DIR,
    SPARK_TTS_REPO_PATH
)

# Global model variables
vllm_model = None
audio_tokenizer = None
device = None


def initialize_models():
    global vllm_model, audio_tokenizer, device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if os.path.exists(SPARK_TTS_REPO_PATH):
        sys.path.append(SPARK_TTS_REPO_PATH)
        print(f"Added {SPARK_TTS_REPO_PATH} to Python path")
    else:
        print(f"Warning: {SPARK_TTS_REPO_PATH} not found. Clone it with:")
        print(f"git clone https://github.com/SparkAudio/Spark-TTS")

    print(f"Loading Spark TTS model: {MODEL_NAME}...")
    vllm_model = LLM(
        MODEL_NAME,
        enforce_eager=False,
        gpu_memory_utilization=0.5,
        tensor_parallel_size=1
    )
    print("✅ Model loaded successfully!")

    if not os.path.exists(TOKENIZER_CACHE_DIR) or not os.path.exists(f"{TOKENIZER_CACHE_DIR}/config.yaml"):
        print(f"Downloading tokenizer from {TOKENIZER_REPO}...")
        snapshot_download(repo_id=TOKENIZER_REPO, local_dir=TOKENIZER_CACHE_DIR)
        print(f"✅ Tokenizer downloaded to {TOKENIZER_CACHE_DIR}")
    else:
        print(f"✅ Tokenizer already exists at {TOKENIZER_CACHE_DIR}")

    try:
        from sparktts.models.audio_tokenizer import BiCodecTokenizer
        print("Initializing audio tokenizer...")
        audio_tokenizer = BiCodecTokenizer(TOKENIZER_CACHE_DIR, device)
        print("✅ Audio tokenizer initialized!")
    except ImportError:
        print("Error: Could not import BiCodecTokenizer. Make sure Spark-TTS repo is available.")
        raise
