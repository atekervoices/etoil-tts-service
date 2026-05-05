import os
import asyncio
import time
import json
import tempfile
import soundfile as sf
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from typing import Optional

from .config import (
    MODEL_NAME,
    AUDIO_SAMPLERATE,
    AUDIO_BITS_PER_SAMPLE,
    AUDIO_CHANNELS,
    DEFAULT_TEMPERATURE,
    DEFAULT_SPEAKER_ID
)
from .models import AudioRequest, VoiceCloningRequest
from .speaker_tokens import GLOBAL_IDS_BY_SPEAKER
from .audio_processing import convert_to_pcm16_bytes
from .audio_generation import text_to_speech_cloned, generate_audio_segment
from .model_loader import initialize_models
from . import model_loader

app = FastAPI()


async def generate_audio_chunks_async(text: str, speaker_id: str, temperature: float):
    """Async generator for audio chunks."""
    loop = asyncio.get_running_loop()
    texts = text.split('. ')
    texts = [t.strip() for t in texts if t.strip()]
    
    for i, chunk in enumerate(texts):
        print(f"Generating audio chunk {i+1}/{len(texts)}: '{chunk[:50]}...'")
        try:
            wav_np = await loop.run_in_executor(
                None,
                generate_audio_segment,
                chunk,
                speaker_id,
                temperature
            )
            pcm_bytes = convert_to_pcm16_bytes(wav_np)
            print(f"Generated chunk {i+1}: {len(pcm_bytes)} bytes")
            yield pcm_bytes
        except Exception as e:
            print(f"Error generating chunk {i+1}: {e}")
            raise


@app.on_event("startup")
async def startup_event():
    print("Initializing Spark TTS models...")
    initialize_models()
    print("Server ready!")


@app.websocket("/v1/audio/speech/stream/ws")
async def websocket_audio_stream(websocket: WebSocket):
    """
    WebSocket endpoint for streaming audio generation.

    Protocol:
    - Client sends JSON: {"input": "text", "speaker_id": "pcm_female_2", "temperature": 0.7, "continue": true/false, "segment_id": "id"}
    - Server sends: {"type": "start", "segment_id": "id"} followed by binary audio chunks
    - Server sends: {"type": "end", "segment_id": "id"} when segment complete
    """
    await websocket.accept()
    print("WebSocket connection established")

    ping_task = None
    last_activity = time.time()

    async def ping_loop():
        nonlocal last_activity
        while True:
            try:
                await asyncio.sleep(30)
                if time.time() - last_activity > 60:
                    print("Connection idle, sending ping")
                    await websocket.send_json({"type": "ping"})
                    last_activity = time.time()
            except Exception:
                break

    try:
        ping_task = asyncio.create_task(ping_loop())

        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=300)
                last_activity = time.time()
                message = json.loads(data)

                text = message.get("input", "")
                speaker_id = message.get("speaker_id", DEFAULT_SPEAKER_ID)
                temperature = message.get("temperature", DEFAULT_TEMPERATURE)
                continue_stream = message.get("continue", True)
                segment_id = message.get("segment_id", "default")

                if not text and not continue_stream:
                    print("Received end signal, closing stream")
                    break

                if text:
                    await websocket.send_json({
                        "type": "start",
                        "segment_id": segment_id,
                        "speaker_id": speaker_id
                    })

                    chunk_count = 0
                    try:
                        async_generator = generate_audio_chunks_async(
                            text=text,
                            speaker_id=speaker_id,
                            temperature=temperature
                        )

                        async for audio_chunk in async_generator:
                            chunk_count += 1
                            print(f"Immediately sending audio chunk {chunk_count}: {len(audio_chunk)} bytes")
                            await websocket.send_bytes(audio_chunk)
                            print(f"Audio chunk {chunk_count} sent and played immediately")
                            last_activity = time.time()

                    except Exception as e:
                        print(f"Error during audio generation: {e}")
                        await websocket.send_json({
                            "type": "error",
                            "message": f"Audio generation error: {str(e)}"
                        })
                        continue

                    print(f"Finished streaming {chunk_count} audio chunks, sending end message")
                    await websocket.send_json({
                        "type": "end",
                        "segment_id": segment_id
                    })

            except asyncio.TimeoutError:
                print("Client timeout, closing connection")
                break
            except WebSocketDisconnect:
                print("Client disconnected")
                break
            except json.JSONDecodeError:
                print("Invalid JSON received")
                await websocket.send_json({"type": "error", "message": "Invalid JSON"})
            except Exception as e:
                print(f"Error processing message: {e}")
                await websocket.send_json({"type": "error", "message": str(e)})

    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        if ping_task:
            ping_task.cancel()
            try:
                await ping_task
            except asyncio.CancelledError:
                pass
        print("WebSocket connection closed")


@app.websocket("/v1/audio/speech/clone/ws")
async def websocket_voice_cloning(websocket: WebSocket):
    """
    WebSocket endpoint for voice cloning streaming.

    Protocol:
    - Client sends JSON: {"input": "text", "reference_audio_path": "path/to/audio.wav", "reference_text": "optional", "temperature": 0.7, "segment_id": "id"}
    - Server sends: {"type": "start", "segment_id": "id"} followed by binary audio chunks
    - Server sends: {"type": "end", "segment_id": "id"} when segment complete
    """
    await websocket.accept()
    print("Voice cloning WebSocket connection established")

    ping_task = None
    last_activity = time.time()

    async def ping_loop():
        nonlocal last_activity
        while True:
            try:
                await asyncio.sleep(30)
                if time.time() - last_activity > 60:
                    print("Connection idle, sending ping")
                    await websocket.send_json({"type": "ping"})
                    last_activity = time.time()
            except Exception:
                break

    try:
        ping_task = asyncio.create_task(ping_loop())

        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=300)
                last_activity = time.time()
                message = json.loads(data)

                text = message.get("input", "")
                reference_audio_path = message.get("reference_audio_path", "")
                reference_text = message.get("reference_text")
                temperature = message.get("temperature", DEFAULT_TEMPERATURE)
                segment_id = message.get("segment_id", "default")

                if not text:
                    print("Received empty text, skipping")
                    continue

                if not reference_audio_path:
                    await websocket.send_json({
                        "type": "error",
                        "message": "reference_audio_path is required for voice cloning"
                    })
                    continue

                if not os.path.exists(reference_audio_path):
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Reference audio file not found: {reference_audio_path}"
                    })
                    continue

                await websocket.send_json({
                    "type": "start",
                    "segment_id": segment_id,
                    "reference_audio": reference_audio_path
                })

                try:
                    loop = asyncio.get_running_loop()
                    result_wav = await loop.run_in_executor(
                        None,
                        text_to_speech_cloned,
                        text,
                        model_loader.audio_tokenizer,
                        model_loader.vllm_model,
                        reference_audio_path,
                        reference_text,
                        temperature,
                        model_loader.device
                    )

                    pcm_bytes = convert_to_pcm16_bytes(result_wav)
                    print(f"Generated cloned audio: {len(result_wav)} samples, {len(pcm_bytes)} bytes")

                    if len(pcm_bytes) > 0:
                        await websocket.send_bytes(pcm_bytes)
                        print(f"Cloned audio sent for segment {segment_id}")
                    else:
                        print("Warning: Generated empty cloned audio data")

                except Exception as e:
                    print(f"Error during voice cloning: {e}")
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Voice cloning error: {str(e)}"
                    })
                    continue

                await websocket.send_json({
                    "type": "end",
                    "segment_id": segment_id
                })

            except asyncio.TimeoutError:
                print("Client timeout, closing connection")
                break
            except WebSocketDisconnect:
                print("Client disconnected")
                break
            except json.JSONDecodeError:
                print("Invalid JSON received")
                await websocket.send_json({"type": "error", "message": "Invalid JSON"})
            except Exception as e:
                print(f"Error processing message: {e}")
                await websocket.send_json({"type": "error", "message": str(e)})

    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        if ping_task:
            ping_task.cancel()
            try:
                await ping_task
            except asyncio.CancelledError:
                pass
        print("Voice cloning WebSocket connection closed")


@app.post("/v1/audio/speech/stream")
async def http_audio_stream(request: AudioRequest):
    """
    HTTP endpoint for streaming audio as raw PCM bytes.

    Body:
        text: Text to synthesize (required)
        speaker_id: Speaker string ID e.g. "pcm_female_2" (default: nyn_female_248)
        temperature: Controls randomness 0.0-1.0 (default: 0.7)
        max_tokens: Max tokens to generate (default: 2048)
    """
    print(f"Received HTTP streaming request for: '{request.text[:50]}...'")
    print(f"Speaker ID: {request.speaker_id}")

    if request.speaker_id not in GLOBAL_IDS_BY_SPEAKER:
        return {
            "error": f"Unknown speaker_id: '{request.speaker_id}'.",
            "valid_speaker_ids": list(GLOBAL_IDS_BY_SPEAKER.keys())
        }

    async def stream_pcm():
        async for chunk in generate_audio_chunks_async(
            text=request.text,
            speaker_id=request.speaker_id,
            temperature=request.temperature
        ):
            yield chunk

    return StreamingResponse(
        stream_pcm(),
        media_type="audio/pcm",
        headers={
            "X-Sample-Rate": str(AUDIO_SAMPLERATE),
            "X-Bit-Depth": str(AUDIO_BITS_PER_SAMPLE),
            "X-Channels": str(AUDIO_CHANNELS),
        }
    )


@app.post("/v1/audio/speech/clone/upload")
async def voice_cloning_upload(
    text: str = Form(...),
    reference_audio: UploadFile = File(...),
    reference_text: Optional[str] = Form(None),
    temperature: float = Form(DEFAULT_TEMPERATURE)
):
    """
    Voice cloning endpoint that accepts reference audio as file upload.
    """
    print(f"Received voice cloning upload request for: '{text[:50]}...'")
    print(f"Reference audio file: {reference_audio.filename}, size: {reference_audio.size}")

    if not text or not text.strip():
        return {"error": "Text parameter is required and cannot be empty"}

    if not reference_audio or not reference_audio.filename:
        return {"error": "Reference audio file is required"}

    allowed_extensions = ['.wav', '.mp3', '.m4a', '.flac', '.ogg']
    file_ext = os.path.splitext(reference_audio.filename)[1].lower()
    if file_ext not in allowed_extensions:
        return {"error": f"Invalid audio file format: {file_ext}. Allowed: {', '.join(allowed_extensions)}"}

    if not 0.0 <= temperature <= 1.0:
        return {"error": f"Temperature must be between 0.0 and 1.0, got: {temperature}"}

    temp_fd, temp_path = tempfile.mkstemp(suffix='.wav')
    os.close(temp_fd)
    defer_cleanup = False

    try:
        content = await reference_audio.read()
        if not content:
            return {"error": "Uploaded audio file is empty"}

        with open(temp_path, "wb") as f:
            f.write(content)

        if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
            return {"error": "Failed to save temporary audio file"}

        try:
            test_wav, test_sr = sf.read(temp_path)
            print(f"Successfully read test audio: shape={test_wav.shape}, sr={test_sr}")
        except Exception as e:
            return {"error": f"Cannot read saved audio file: {str(e)}"}

        defer_cleanup = True

        async def stream_cloned_pcm():
            loop = asyncio.get_running_loop()
            try:
                result_wav = await loop.run_in_executor(
                    None,
                    text_to_speech_cloned,
                    text,
                    model_loader.audio_tokenizer,
                    model_loader.vllm_model,
                    temp_path,
                    reference_text,
                    temperature,
                    model_loader.device
                )
                pcm_bytes = convert_to_pcm16_bytes(result_wav)
                if len(pcm_bytes) > 0:
                    yield pcm_bytes
                else:
                    raise ValueError("Generated empty cloned audio data")
            except Exception as e:
                print(f"ERROR: {str(e)}")
                import traceback
                traceback.print_exc()
                yield json.dumps({"error": str(e)}).encode()
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    print(f"Cleaned up temporary file: {temp_path}")

        return StreamingResponse(
            stream_cloned_pcm(),
            media_type="audio/pcm",
            headers={
                "X-Sample-Rate": str(AUDIO_SAMPLERATE),
                "X-Bit-Depth": str(AUDIO_BITS_PER_SAMPLE),
                "X-Channels": str(AUDIO_CHANNELS),
                "X-Voice-Cloning": "true"
            }
        )

    except Exception as e:
        print(f"ERROR processing upload: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}
    finally:
        if not defer_cleanup and os.path.exists(temp_path):
            os.unlink(temp_path)


@app.post("/v1/audio/speech/clone/debug")
async def voice_cloning_debug(
    text: Optional[str] = Form(None),
    reference_audio: Optional[UploadFile] = File(None),
    reference_text: Optional[str] = Form(None),
    temperature: Optional[float] = Form(None)
):
    return {
        "message": "Debug endpoint received data",
        "text": text,
        "reference_audio_filename": reference_audio.filename if reference_audio else None,
        "reference_audio_size": reference_audio.size if reference_audio else None,
        "reference_text": reference_text,
        "temperature": temperature
    }


@app.post("/v1/audio/speech/clone")
async def voice_cloning_http(request: VoiceCloningRequest):
    """
    HTTP endpoint for voice cloning using reference audio path.
    """
    print(f"Received voice cloning request for: '{request.text[:50]}...'")
    print(f"Reference audio: {request.reference_audio_path}")

    if not os.path.exists(request.reference_audio_path):
        return {"error": f"Reference audio file not found: {request.reference_audio_path}"}

    async def stream_cloned_pcm():
        loop = asyncio.get_running_loop()
        try:
            result_wav = await loop.run_in_executor(
                None,
                text_to_speech_cloned,
                request.text,
                model_loader.audio_tokenizer,
                model_loader.vllm_model,
                request.reference_audio_path,
                request.reference_text,
                request.temperature,
                model_loader.device
            )
            pcm_bytes = convert_to_pcm16_bytes(result_wav)
            if len(pcm_bytes) > 0:
                yield pcm_bytes
            else:
                raise ValueError("Generated empty cloned audio data")
        except Exception as e:
            print(f"ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            yield json.dumps({"error": str(e)}).encode()

    return StreamingResponse(
        stream_cloned_pcm(),
        media_type="audio/pcm",
        headers={
            "X-Sample-Rate": str(AUDIO_SAMPLERATE),
            "X-Bit-Depth": str(AUDIO_BITS_PER_SAMPLE),
            "X-Channels": str(AUDIO_CHANNELS),
            "X-Voice-Cloning": "true"
        }
    )


@app.get("/")
async def read_root():
    return {
        "message": "Spark TTS Streaming API with Voice Cloning",
        "model": MODEL_NAME,
        "sample_rate": AUDIO_SAMPLERATE,
        "available_speaker_ids": list(GLOBAL_IDS_BY_SPEAKER.keys()),
        "features": ["text-to-speech", "voice-cloning", "streaming"],
        "endpoints": {
            "websocket": "/v1/audio/speech/stream/ws",
            "http": "/v1/audio/speech/stream",
            "voice_cloning_websocket": "/v1/audio/speech/clone/ws",
            "voice_cloning_http": "/v1/audio/speech/clone",
            "voice_cloning_upload": "/v1/audio/speech/clone/upload",
            "speaker_ids": "/v1/voices"
        },
        "example_usage": {
            "http": {
                "url": "POST /v1/audio/speech/stream",
                "body": {
                    "text": "Your text here",
                    "speaker_id": "pcm_female_2",
                    "temperature": 0.7
                }
            },
            "websocket": {
                "connect": "ws://localhost:8002/v1/audio/speech/stream/ws",
                "send": {
                    "input": "Your text here",
                    "speaker_id": "pcm_female_2",
                    "segment_id": "segment_1"
                }
            }
        }
    }


@app.get("/v1/voices")
async def list_voices():
    """List all available speaker IDs."""
    return {
        "speaker_ids": [
            {"id": sid, "language": sid.split("_")[0]}
            for sid in GLOBAL_IDS_BY_SPEAKER.keys()
        ]
    }
