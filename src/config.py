import os

# Configuration
CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1")
MODEL_NAME = os.environ.get("MODEL_NAME", "crestai/spark-tts-nexvox_v20")
TOKENIZER_REPO = os.environ.get("TOKENIZER_REPO", "unsloth/Spark-TTS-0.5B")
TOKENIZER_CACHE_DIR = os.environ.get("TOKENIZER_CACHE_DIR", "Spark-TTS-0.5B")
SPARK_TTS_REPO_PATH = os.environ.get("SPARK_TTS_REPO_PATH", "Spark-TTS")

# Set NCCL environment variables for multi-GPU communication
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
os.environ["NCCL_DEBUG"] = "WARN"
os.environ["NCCL_SOCKET_IFNAME"] = "lo"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_NET_GDR_LEVEL"] = "0"
os.environ["NCCL_SHM_DISABLE"] = "1"
os.environ["NCCL_TREE_THRESHOLD"] = "0"
os.environ["NCCL_RING_THRESHOLD"] = "8388608"

# Audio configuration
AUDIO_SAMPLERATE = 16000
AUDIO_BITS_PER_SAMPLE = 16
AUDIO_CHANNELS = 1

# Default parameters
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 2048
DEFAULT_SPEAKER_ID = "pcm_female_1"  # Runyankore female
