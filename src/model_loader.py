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
end_semantic_token_id = None


def initialize_models():
    global vllm_model, audio_tokenizer, device, end_semantic_token_id

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if os.path.exists(SPARK_TTS_REPO_PATH):
        sys.path.append(SPARK_TTS_REPO_PATH)
        print(f"Added {SPARK_TTS_REPO_PATH} to Python path")
    else:
        print(f"Warning: {SPARK_TTS_REPO_PATH} not found. Clone it with:")
        print(f"git clone https://github.com/SparkAudio/Spark-TTS")

    gpu_memory_util = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.65"))
    print(f"Loading Spark TTS model: {MODEL_NAME}...")
    print(f"GPU memory utilization: {gpu_memory_util}")
    vllm_model = LLM(
        MODEL_NAME,
        enforce_eager=False,
        gpu_memory_utilization=gpu_memory_util,
        tensor_parallel_size=1
    )
    print("✅ Model loaded successfully!")

    # Discover end_semantic_token_id from the model's tokenizer
    try:
        tokenizer = vllm_model.get_tokenizer()
        token_id = tokenizer.convert_tokens_to_ids("<|end_semantic_token|>")
        if token_id is not None and token_id != tokenizer.unk_token_id:
            end_semantic_token_id = token_id
            print(f"✅ Found end_semantic_token_id: {end_semantic_token_id}")
        else:
            # Try alternative: search vocabulary for the token
            vocab = tokenizer.get_vocab()
            for token_str, tid in vocab.items():
                if "end_semantic" in token_str:
                    end_semantic_token_id = tid
                    print(f"✅ Found end_semantic_token_id via vocab search: {end_semantic_token_id} ('{token_str}')")
                    break
            if end_semantic_token_id is None:
                print("⚠️ Could not find end_semantic_token_id - will rely on max_tokens limit")
    except Exception as e:
        print(f"⚠️ Error looking up end_semantic_token_id: {e}")
        print("Will rely on max_tokens limit for generation stop")

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
