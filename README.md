# Spark TTS Streaming Server

A high-performance FastAPI server for streaming text-to-speech using the Spark TTS model with vLLM. Supports WebSocket and HTTP streaming with multiple African language voices.

## Features

- **Multi-language Support**: African languages including Acholi, Ateso, Runyankore, Luganda, Swahili, and more
- **Voice Cloning**: Clone voices from reference audio
- **Fast Streaming**: Sentence-by-sentence generation for low latency
- **Dual Protocols**: WebSocket and HTTP streaming endpoints
- **vLLM Backend**: Efficient inference with GPU acceleration
- **Modular Architecture**: Clean separation of concerns for maintainability
- **Docker Support**: Easy deployment with Docker and Docker Compose

## Project Structure

```
nexvoxai_spark-tts-vllm/
├── src/                      # Modularized source code
│   ├── __init__.py
│   ├── config.py            # Configuration constants
│   ├── speakers.py          # Speaker ID mappings
│   ├── speaker_tokens.py    # Precomputed global tokens
│   ├── models.py            # Pydantic models for API requests
│   ├── text_processing.py   # Text chunking functions
│   ├── audio_processing.py  # Audio processing functions
│   ├── model_loader.py      # Model initialization
│   ├── audio_generation.py  # Audio generation functions
│   └── api.py               # FastAPI app and endpoints
├── main.py                  # Application entry point
├── Dockerfile               # Docker configuration
├── docker-compose.yml       # Docker Compose configuration
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Quick Start

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (tested on RTX 4090, 24GB VRAM)
- Hugging Face account with access token

### Local Installation

1. **Clone the repository:**

```bash
git clone <repository-url>
cd nexvoxai_spark-tts-vllm
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

3. **Clone Spark-TTS repository:**

```bash
git clone https://github.com/SparkAudio/Spark-TTS
```

4. **Set up Hugging Face authentication:**

```bash
huggingface-cli login
# Or set token directly
export HF_TOKEN=your_token_here
```

5. **Run the server:**

```bash
python main.py
```

The server will start on `http://0.0.0.0:8002`

## Docker Deployment

### Prerequisites

1. **Docker Engine** (20.10+)
2. **Docker Compose** (2.0+)
3. **NVIDIA Container Toolkit** (for GPU support)
4. **NVIDIA GPU** with CUDA support

### Install NVIDIA Container Toolkit

```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### Verify GPU Support

```bash
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

### Using Docker Compose (Recommended)

```bash
# Build and start the service
docker compose up --build

# Run in detached mode
docker compose up -d --build

# View logs
docker compose logs -f

# Stop the service
docker compose down
```

### Using Docker Directly

```bash
# Build the image
docker build -t spark-tts-streaming .

# Run the container
docker run --gpus all -p 8002:8002 --name spark-tts spark-tts-streaming

# Run in detached mode
docker run -d --gpus all -p 8002:8002 --name spark-tts spark-tts-streaming
```
```bash
# 1. Build the image with v1.4 tag
docker build -t spark-tts-streaming:v1.4 .

# 2. Tag it for Docker Hub
docker tag spark-tts-streaming:v1.4 simonallanachuka/spark-tts-streaming:v1.4

# 3. Push to Docker Hub
docker push simonallanachuka/spark-tts-streaming:v1.4
```

## API Reference

### Available Voices

The server supports multiple African language voices. Use the `/v1/voices` endpoint to get the full list.

Available languages include:
- English (eng)
- Acholi (ach)
- Ateso (teo)
- Runyankore (nyn)
- Swahili (swa)
- Luganda (lug)
- Lusoga (xog)
- Kinyarwanda (kin)
- Luo (luo)
- Kikuyu (kik)
- Hausa (hau)
- Igbo (ibo)
- Twi (twi)
- Yoruba (yor)
- Wolof (wol)
- Nigerian Pidgin (pcm)
- Fula (fat)

### Endpoints

#### 1. WebSocket Streaming

**Endpoint:** `ws://localhost:8002/v1/audio/speech/stream/ws`

**Protocol:**

**Client → Server:**
```json
{
  "input": "Your text here",
  "speaker_id": "lug_female_4",
  "temperature": 0.7,
  "segment_id": "unique_id",
  "continue": true
}
```

**Server → Client:**
```json
// Start message
{"type": "start", "segment_id": "unique_id", "speaker_id": "lug_female_4"}

// Binary audio chunks (PCM16, 16kHz, mono)
<binary data>

// End message
{"type": "end", "segment_id": "unique_id"}

// Error message (if any)
{"type": "error", "message": "error description"}
```

#### 2. HTTP Streaming

**Endpoint:** `POST /v1/audio/speech/stream`

**Request Body:**
```json
{
  "text": "Your text here",
  "speaker_id": "lug_female_4",
  "temperature": 0.7,
  "max_tokens": 2048
}
```

**Response:**
- Content-Type: `audio/pcm`
- Headers:
  - `X-Sample-Rate: 16000`
  - `X-Bit-Depth: 16`
  - `X-Channels: 1`
- Body: Streaming PCM audio data

#### 3. Voice Cloning (HTTP)

**Endpoint:** `POST /v1/audio/speech/clone`

**Request Body:**
```json
{
  "text": "Your text here",
  "reference_audio_path": "/path/to/reference.wav",
  "reference_text": "Optional reference text",
  "temperature": 0.7
}
```

#### 4. Voice Cloning (Upload)

**Endpoint:** `POST /v1/audio/speech/clone/upload`

**Request:** Multipart form data with:
- `text`: Text to synthesize
- `reference_audio`: Audio file (WAV, MP3, M4A, FLAC, OGG)
- `reference_text`: Optional reference text
- `temperature`: Temperature (default: 0.7)

#### 5. Voice Cloning (WebSocket)

**Endpoint:** `ws://localhost:8002/v1/audio/speech/clone/ws`

**Protocol:**
```json
{
  "input": "Your text here",
  "reference_audio_path": "/path/to/reference.wav",
  "reference_text": "Optional reference text",
  "temperature": 0.7,
  "segment_id": "unique_id"
}
```

#### 6. List Available Voices

**Endpoint:** `GET /v1/voices`

**Response:**
```json
{
  "speaker_ids": [
    {"id": "lug_female_4", "language": "lug"},
    {"id": "swa_male_3", "language": "swa"}
  ]
}
```

#### 7. Server Info

**Endpoint:** `GET /`

Returns server information, available voices, and usage examples.

## Usage Examples

### Python - WebSocket Client

```python
import asyncio
import websockets
import json
import wave

async def generate_speech():
    uri = "ws://localhost:8002/v1/audio/speech/stream/ws"

    async with websockets.connect(uri) as ws:
        # Send text for generation
        await ws.send(json.dumps({
            "input": "Nze Prosi Nafula. Ndi musawo akola ku bantu abalina kookolo.",
            "speaker_id": "lug_female_4",
            "temperature": 0.7,
            "segment_id": "test_1"
        }))
        
        audio_data = bytearray()
        
        while True:
            message = await ws.recv()
            
            if isinstance(message, bytes):
                # Audio chunk received
                audio_data.extend(message)
                print(f"Received {len(message)} bytes")
            else:
                # Control message
                data = json.loads(message)
                print(f"Control: {data}")
                
                if data.get("type") == "end":
                    break
                elif data.get("type") == "error":
                    print(f"Error: {data.get('message')}")
                    break
        
        # Save to WAV file
        with wave.open("output.wav", "wb") as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(16000)  # 16kHz
            wav_file.writeframes(bytes(audio_data))
        
        print("Audio saved to output.wav")

asyncio.run(generate_speech())
```

### Python - HTTP Client

```python
import requests
import wave

def generate_speech_http():
    url = "http://localhost:8001/v1/audio/speech/stream"
    
    response = requests.post(
        url,
        json={
            "text": "Habari, naitwa Prosi Nafula. Mimi ni muuguzi.",
            "voice": "swahili_male",
            "temperature": 0.7
        },
        stream=True
    )
    
    audio_data = bytearray()
    
    for chunk in response.iter_content(chunk_size=8192):
        if chunk:
            audio_data.extend(chunk)
            print(f"Received {len(chunk)} bytes")
    
    # Save to WAV file
    with wave.open("output_http.wav", "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(bytes(audio_data))
    
    print("Audio saved to output_http.wav")

generate_speech_http()
```

### JavaScript - WebSocket Client

```javascript
const ws = new WebSocket('ws://localhost:8001/v1/audio/speech/stream/ws');

const audioChunks = [];

ws.onopen = () => {
    ws.send(JSON.stringify({
        input: "Nze Prosi Nafula. Ndi musawo.",
        voice: "luganda_female",
        temperature: 0.7,
        segment_id: "js_test_1"
    }));
};

ws.onmessage = (event) => {
    if (event.data instanceof Blob) {
        // Audio chunk
        audioChunks.push(event.data);
        console.log(`Received audio chunk: ${event.data.size} bytes`);
    } else {
        // Control message
        const data = JSON.parse(event.data);
        console.log('Control message:', data);
        
        if (data.type === 'end') {
            // Combine chunks and play
            const audioBlob = new Blob(audioChunks, { type: 'audio/pcm' });
            // Process audioBlob...
            ws.close();
        }
    }
};

ws.onerror = (error) => {
    console.error('WebSocket error:', error);
};
```

### cURL - HTTP Endpoint

```bash
curl -X POST http://localhost:8001/v1/audio/speech/stream \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Nze Prosi Nafula. Ndi musawo.",
    "voice": "luganda_female",
    "temperature": 0.7
  }' \
  --output output.pcm

# Convert PCM to WAV
ffmpeg -f s16le -ar 16000 -ac 1 -i output.pcm output.wav
```

## Configuration

### Environment Variables

```bash
# GPU selection
export CUDA_VISIBLE_DEVICES=0

# Model configuration
export MODEL_NAME=crestai/spark-tts-nexvox
export TOKENIZER_REPO=unsloth/Spark-TTS-0.5B
export TOKENIZER_CACHE_DIR=Spark-TTS-0.5B

# Spark-TTS repository path
export SPARK_TTS_REPO_PATH=Spark-TTS

# Server configuration
export HOST=0.0.0.0
export PORT=8001
```

### Server Parameters

Edit in `spark_tts_streaming_server.py`:

```python
# Default parameters
DEFAULT_TEMPERATURE = 0.7  # 0.1 (conservative) to 1.0 (creative)
DEFAULT_MAX_TOKENS = 2048  # Maximum tokens per generation
DEFAULT_SPEAKER_ID = 248   # Default voice

# Audio configuration
AUDIO_SAMPLERATE = 16000   # 16kHz sample rate
AUDIO_BITS_PER_SAMPLE = 16 # 16-bit audio
AUDIO_CHANNELS = 1         # Mono audio
```

## Performance Tips

### Memory Management

- **GPU Memory**: Server uses 50% GPU memory by default (`gpu_memory_utilization=0.5`)
- **Concurrent Requests**: WebSocket allows multiple concurrent connections
- **Sentence Chunking**: Automatic text splitting for efficient streaming

### Optimization

1. **Temperature Settings:**
   - `0.1-0.3`: More consistent, less varied
   - `0.5-0.7`: Balanced (recommended)
   - `0.8-1.0`: More creative, potentially less stable

2. **Text Length:**
   - Short texts (1-3 sentences): ~1-2 seconds
   - Medium texts (5-10 sentences): ~3-5 seconds
   - Long texts: Automatically chunked

3. **Hardware Requirements:**
   - Minimum: 16GB VRAM
   - Recommended: 24GB VRAM (RTX 4090, A5000)
   - CPU: Multi-core for async processing

## Audio Format

**Output Specification:**
- Format: PCM (uncompressed)
- Sample Rate: 16,000 Hz
- Bit Depth: 16-bit
- Channels: Mono (1)
- Byte Order: Little-endian
- Encoding: Signed integer

**Converting to Common Formats:**

```bash
# PCM to WAV
ffmpeg -f s16le -ar 16000 -ac 1 -i input.pcm output.wav

# PCM to MP3
ffmpeg -f s16le -ar 16000 -ac 1 -i input.pcm -b:a 128k output.mp3

# PCM to OGG
ffmpeg -f s16le -ar 16000 -ac 1 -i input.pcm -c:a libvorbis output.ogg
```

## Troubleshooting

### Common Issues

**1. "Could not import BiCodecTokenizer"**
```bash
# Make sure Spark-TTS is cloned
git clone https://github.com/SparkAudio/Spark-TTS
export SPARK_TTS_REPO_PATH=./Spark-TTS
```

**2. "CUDA out of memory"**
```python
# Reduce GPU memory utilization in code
vllm_model = LLM(
    MODEL_NAME,
    gpu_memory_utilization=0.3  # Reduce from 0.5
)
```

**3. "No semantic tokens found"**
- Increase `max_tokens` parameter
- Check if input text is valid
- Try different temperature values

**4. WebSocket connection refused**
```bash
# Check if server is running
curl http://localhost:8001/

# Check firewall settings
sudo ufw allow 8001
```

## Development

### Running in Development Mode

```bash
# With auto-reload (slower startup)
uvicorn spark_tts_streaming_server:app --reload --host 0.0.0.0 --port 8001

# Production mode
python spark_tts_streaming_server.py
```

### Testing

```python
# Test voices endpoint
curl http://localhost:8001/v1/voices

# Test health check
curl http://localhost:8001/

# Test streaming
python test_client.py
```

## License

This project uses:
- **Spark TTS**: [License](https://github.com/SparkAudio/Spark-TTS)
- **vLLM**: Apache 2.0 License
- **FastAPI**: MIT License

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Support

For issues and questions:
- GitHub Issues: [Create an issue]
- Spark TTS: https://github.com/SparkAudio/Spark-TTS
- vLLM Docs: https://docs.vllm.ai/

## Acknowledgments

- Spark Audio team for the Spark TTS model
- vLLM team for efficient LLM inference
- Anthropic for Claude assistance

---

