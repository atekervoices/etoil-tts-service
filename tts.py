from __future__ import annotations

import asyncio
import json
import weakref
import time
from dataclasses import dataclass, replace

import aiohttp

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    tokenize,
    tts,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

AUDIO_FRAME_SIZE_MS = 20


@dataclass
class _TTSOptions:
    voice: str
    speaker_id: int | None
    temperature: float
    base_url: str

    def _normalize_url(self, url: str) -> str:
        """Ensure the URL ends with exactly one slash."""
        return url.rstrip('/') + '/'

    def get_ws_url(self, path: str) -> str:
        """Get WebSocket URL for a given path."""
        path = path.lstrip('/')
        base = self._normalize_url(self.base_url)
        # Convert http to ws or https to wss
        ws_base = base.replace('http://', 'ws://').replace('https://', 'wss://')
        return f"{ws_base}{path}"


class SparkTTS(tts.TTS):
    """
    LiveKit TTS plugin for Spark TTS real-time streaming synthesis.
    
    Supports multiple African language voices including Acholi, Ateso, 
    Runyankore, Lugbara, Swahili, and Luganda.
    
    Connects to our real-time WebSocket streaming endpoint for optimal performance.
    
    Example usage:
        from spark_tts_plugin import SparkTTS
        
        # Using voice name
        tts_instance = SparkTTS(
            base_url="http://35.203.124.213:8000/",
            voice="luganda_female",
            temperature=0.7
        )
        
        # Using speaker ID directly
        tts_instance = SparkTTS(
            base_url="http://35.203.124.213:8000/",
            speaker_id=248,
            temperature=0.7
        )
    """
    
    # Available voices mapping
    VOICES = {
        "acholi_female": 241,
        "ateso_female": 242,
        "runyankore_female": 243,
        "lugbara_female": 245,
        "swahili_male": 246,
        "luganda_female": 248,
    }
    
    def __init__(
        self,
        *,
        voice: str = "luganda_female",
        speaker_id: int | None = None,
        temperature: float = 0.7,
        http_session: aiohttp.ClientSession | None = None,
        base_url: str = "http://35.203.124.213:8000/",
    ) -> None:
        """
        Initialize Spark TTS plugin with real-time streaming support.
        
        Args:
            voice: Voice name (e.g., "luganda_female", "swahili_male")
            speaker_id: Direct speaker ID (overrides voice if provided)
            temperature: Generation temperature (0.1-1.0, default 0.7)
            http_session: Optional aiohttp session
            base_url: Spark TTS server URL
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=16000,  # Spark TTS uses 16kHz
            num_channels=1,
        )

        # Validate voice or speaker_id
        if speaker_id is None and voice not in self.VOICES:
            raise ValueError(
                f"Invalid voice '{voice}'. Available voices: {list(self.VOICES.keys())}"
            )

        self._opts = _TTSOptions(
            voice=voice,
            speaker_id=speaker_id,
            temperature=temperature,
            base_url=base_url,
        )
        self._session = http_session
        self._pool = utils.ConnectionPool[aiohttp.ClientWebSocketResponse](
            connect_cb=self._connect_ws,
            close_cb=self._close_ws,
            max_session_duration=300,
            mark_refreshed_on_get=True,
        )
        self._streams = weakref.WeakSet[SynthesizeStream]()

    async def _connect_ws(self, timeout: float) -> aiohttp.ClientWebSocketResponse:
        """Connect to WebSocket endpoint."""
        session = self._ensure_session()
        url = self._opts.get_ws_url("/v1/audio/speech/stream/ws")
        return await asyncio.wait_for(session.ws_connect(url), timeout)

    async def _close_ws(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        """Close WebSocket connection."""
        await ws.close()

    def _ensure_session(self) -> aiohttp.ClientSession:
        """Ensure HTTP session exists."""
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    def prewarm(self) -> None:
        """Prewarm the connection pool."""
        self._pool.prewarm()

    def update_options(
        self,
        *,
        voice: NotGivenOr[str] = NOT_GIVEN,
        speaker_id: NotGivenOr[int] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
    ) -> None:
        """
        Update TTS options dynamically.
        
        Args:
            voice: New voice name
            speaker_id: New speaker ID
            temperature: New temperature value
        """
        if is_given(voice):
            if voice not in self.VOICES:
                raise ValueError(
                    f"Invalid voice '{voice}'. Available: {list(self.VOICES.keys())}"
                )
            self._opts.voice = voice
            
        if is_given(speaker_id):
            self._opts.speaker_id = speaker_id
            
        if is_given(temperature):
            if not 0.1 <= temperature <= 1.0:
                raise ValueError("Temperature must be between 0.1 and 1.0")
            self._opts.temperature = temperature

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> SynthesizeStream:
        """
        Create a real-time streaming synthesis session using WebSocket.
        
        This is the primary method for LLM-generated text synthesis.
        
        Args:
            conn_options: Connection options
            
        Returns:
            SynthesizeStream for real-time synthesis
        """
        stream = SynthesizeStream(tts=self, conn_options=conn_options)
        self._streams.add(stream)
        return stream

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> "ChunkedStream":
        """
        Synthesize text using HTTP endpoint for non-streaming use cases.
        
        Args:
            text: Text to synthesize
            conn_options: Connection options
            
        Returns:
            ChunkedStream for audio output
        """
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    async def aclose(self) -> None:
        """Close all streams and connections."""
        for stream in list(self._streams):
            await stream.aclose()

        self._streams.clear()
        await self._pool.aclose()


class ChunkedStream(tts.ChunkedStream):
    """Synthesize chunked text using the HTTP streaming endpoint."""

    def __init__(
        self, *, tts: SparkTTS, input_text: str, conn_options: APIConnectOptions
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: SparkTTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        """Run the HTTP streaming synthesis."""
        # Determine speaker_id
        speaker_id = self._opts.speaker_id
        if speaker_id is None:
            speaker_id = SparkTTS.VOICES.get(self._opts.voice)

        json_data = {
            "text": self._input_text,
            "voice": self._opts.voice,
            "speaker_id": speaker_id,
            "temperature": self._opts.temperature,
        }

        try:
            session = self._tts._ensure_session()
            http_url = self._opts._normalize_url(self._opts.base_url) + "v1/audio/speech/stream"
            
            async with session.post(
                http_url,
                json=json_data,
                timeout=aiohttp.ClientTimeout(
                    total=120, sock_connect=self._conn_options.timeout
                ),
            ) as resp:
                resp.raise_for_status()

                output_emitter.initialize(
                    request_id=utils.shortuuid(),
                    sample_rate=16000,  # Spark TTS uses 16kHz
                    num_channels=1,
                    mime_type="audio/pcm",
                    frame_size_ms=AUDIO_FRAME_SIZE_MS,
                )

                async for data, _ in resp.content.iter_chunks():
                    if data:
                        output_emitter.push(data)

                output_emitter.flush()

        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message, status_code=e.status, request_id=None, body=None
            ) from None
        except Exception as e:
            raise APIConnectionError(f"Failed to connect to Spark TTS: {e}") from e


class SynthesizeStream(tts.SynthesizeStream):
    """
    Real-time streaming synthesis using WebSocket for LLM-generated text.
    
    This class is designed to work with LiveKit agents where text comes from LLM responses.
    The LLM pushes text fragments as they're generated, and this plugin converts them
    to real-time audio using our Spark TTS WebSocket streaming endpoint.
    
    Flow:
    1. LLM generates text fragments -> push_text()
    2. Agent calls flush() when LLM is done
    3. Complete text sent to Spark TTS server
    4. Server chunks by sentences and streams audio in real-time
    5. Audio chunks played back as they arrive
    """

    def __init__(self, *, tts: SparkTTS, conn_options: APIConnectOptions):
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: SparkTTS = tts
        self._opts = replace(tts._opts)
        self._start_time: float | None = None

    def push_text(self, text: str) -> None:
        """
        Push LLM-generated text to be synthesized.
        
        Args:
            text: Text fragment from LLM to synthesize
        """
        if self._start_time is None:
            self._start_time = time.perf_counter()
        return super().push_text(text)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        """Run the WebSocket streaming synthesis for LLM text."""
        request_id = utils.shortuuid()
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=16000,  # Spark TTS uses 16kHz
            num_channels=1,
            mime_type="audio/pcm",
            stream=True,
            frame_size_ms=AUDIO_FRAME_SIZE_MS,
        )

        # Determine speaker_id
        speaker_id = self._opts.speaker_id
        if speaker_id is None:
            speaker_id = SparkTTS.VOICES.get(self._opts.voice)

        async def _llm_text_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            """Task to send LLM-generated text to WebSocket for real-time streaming."""
            base_pkt = {
                "voice": self._opts.voice,
                "speaker_id": speaker_id,
                "temperature": self._opts.temperature,
            }
            
            # Buffer LLM text and send only COMPLETE sentences for natural speech
            # No character limits - wait for actual sentence endings
            text_buffer = ""
            sentence_endings = {'.', '!', '?'}
            sentences_sent = 0
            
            try:
                async for data in self._input_ch:
                    if isinstance(data, self._FlushSentinel):
                        # LLM finished, send any remaining text as final sentence
                        if text_buffer.strip():
                            segment_id = utils.shortuuid()
                            token_pkt = base_pkt.copy()
                            token_pkt["input"] = text_buffer.strip()
                            token_pkt["continue"] = False  # Final segment
                            token_pkt["segment_id"] = segment_id
                            self._mark_started()
                            
                            print(f"[Spark TTS] Sending final sentence #{sentences_sent + 1}: '{text_buffer.strip()}'")
                            await ws.send_str(json.dumps(token_pkt))
                            sentences_sent += 1
                        
                        print(f"[Spark TTS] All LLM text sent ({sentences_sent} sentences total), waiting for server audio generation...")
                        break
                    
                    # Accumulate LLM text fragments
                    text_buffer += data
                    
                    # Check for complete sentences - send ONLY when we find a sentence ending
                    while True:
                        # Find the earliest sentence ending
                        sentence_end_pos = -1
                        
                        for ending in sentence_endings:
                            pos = text_buffer.find(ending)
                            if pos != -1 and (sentence_end_pos == -1 or pos < sentence_end_pos):
                                sentence_end_pos = pos
                        
                        if sentence_end_pos == -1:
                            # No complete sentence yet, wait for more text from LLM
                            # Log buffer status
                            print(f"[Spark TTS] Buffering... (buffer has {len(text_buffer)} chars, waiting for sentence ending)")
                            break
                        
                        # Extract complete sentence including punctuation
                        send_text = text_buffer[:sentence_end_pos + 1].strip()
                        text_buffer = text_buffer[sentence_end_pos + 1:].lstrip()
                        
                        if send_text:  # Only send if not empty
                            segment_id = utils.shortuuid()
                            token_pkt = base_pkt.copy()
                            token_pkt["input"] = send_text
                            token_pkt["continue"] = bool(text_buffer.strip())  # More text if buffer has content
                            token_pkt["segment_id"] = segment_id
                            self._mark_started()
                            
                            sentences_sent += 1
                            remaining = len(text_buffer)
                            print(f"[Spark TTS] Sending sentence #{sentences_sent} ({len(send_text)} chars): '{send_text[:80]}{'...' if len(send_text) > 80 else ''}'")
                            print(f"[Spark TTS]   Remaining buffer: {remaining} chars")
                            await ws.send_str(json.dumps(token_pkt))
            
            except asyncio.CancelledError:
                print("[Spark TTS] LLM text task cancelled")
            except Exception as e:
                print(f"[Spark TTS] Error in LLM text task: {e}")
                import traceback
                traceback.print_exc()
            finally:
                print(f"[Spark TTS] LLM text task completed - sent {sentences_sent} sentences total")

        async def _recv_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            """Task to receive real-time audio from WebSocket."""
            segment_started = False
            first_chunk = True
            current_segment_id = None
            segments_received = 0
            last_message_time = time.perf_counter()
            message_timeout = 180  # Wait up to 180 seconds for a message (server may take time to generate audio)
            server_finished = False

            try:
                while not server_finished:
                    try:
                        # Use wait_for with timeout to handle slow/stalled messages
                        # Timeout is long to allow server time to generate and send audio
                        msg = await asyncio.wait_for(ws.__anext__(), timeout=message_timeout)
                    except asyncio.TimeoutError:
                        print(f"[Spark TTS] No message received for {message_timeout}s - connection may have stalled")
                        # Timeout waiting for data - likely server issue
                        break
                    except StopAsyncIteration:
                        print(f"[Spark TTS] WebSocket connection closed after {segments_received} segments received")
                        # This means the WebSocket connection was closed by the server
                        break
                    
                    last_message_time = time.perf_counter()
                    
                    if msg.type in (
                        aiohttp.WSMsgType.CLOSED,
                        aiohttp.WSMsgType.CLOSE,
                        aiohttp.WSMsgType.CLOSING,
                    ):
                        print(f"[Spark TTS] WebSocket closed by server after {segments_received} segments")
                        break

                    if msg.type == aiohttp.WSMsgType.BINARY:
                        # Binary audio data - real-time streaming chunk
                        # Start segment if not already started
                        if not segment_started:
                            segment_id = utils.shortuuid()
                            output_emitter.start_segment(segment_id=segment_id)
                            current_segment_id = segment_id
                            segment_started = True
                            print(f"[Spark TTS] Started implicit segment: {segment_id}")

                        if first_chunk and self._start_time:
                            ttfb = time.perf_counter() - self._start_time
                            print(f"[Spark TTS] TTFB: {ttfb*1000:.2f} ms - Real-time streaming started!")
                            first_chunk = False

                        chunk_size = len(msg.data)
                        print(f"[Spark TTS] Received audio chunk: {chunk_size} bytes (segment {segments_received + 1})")
                        output_emitter.push(msg.data)

                    elif msg.type == aiohttp.WSMsgType.TEXT:
                        # Control messages
                        try:
                            data = json.loads(msg.data)
                        except json.JSONDecodeError:
                            print(f"[Spark TTS] Invalid JSON message: {msg.data}")
                            continue
                            
                        msg_type = data.get("type")
                        print(f"[Spark TTS] Received control message: type={msg_type}")

                        if msg_type == "start":
                            # End previous segment if any
                            if segment_started:
                                output_emitter.end_segment()
                                segments_received += 1
                                print(f"[Spark TTS] Segment {segments_received} ended, next segment starting...")
                            
                            current_segment_id = data.get("segment_id", utils.shortuuid())
                            print(f"[Spark TTS] Starting segment {segments_received + 1}: {current_segment_id}")
                            output_emitter.start_segment(segment_id=current_segment_id)
                            segment_started = True

                        elif msg_type == "end":
                            if segment_started:
                                print(f"[Spark TTS] Ending segment {segments_received + 1}: {current_segment_id}")
                                output_emitter.end_segment()
                                segments_received += 1
                                segment_started = False
                                current_segment_id = None
                                print(f"[Spark TTS] Segment {segments_received} complete, waiting for next...")
                            else:
                                print(f"[Spark TTS] Received 'end' but no segment was active")

                        elif msg_type == "finish":
                            # All segments complete - NOW we can close
                            print(f"[Spark TTS] Server sent finish: {segments_received} total segments")
                            if segment_started:
                                output_emitter.end_segment()
                                segments_received += 1
                                segment_started = False
                            server_finished = True
                            break

                        elif msg_type == "error":
                            error_msg = data.get("message", "Unknown error")
                            print(f"[Spark TTS] Server error: {error_msg}")
                            if segment_started:
                                output_emitter.end_segment()
                            raise APIConnectionError(f"TTS server error: {error_msg}")
                        else:
                            print(f"[Spark TTS] Unknown control message type: {msg_type}")

            except asyncio.CancelledError:
                print("[Spark TTS] Receive task cancelled")
                if segment_started:
                    output_emitter.end_segment()
            except Exception as e:
                print(f"[Spark TTS] Error in receive task: {e}")
                if segment_started:
                    output_emitter.end_segment()
                raise
            finally:
                if segment_started:
                    print("[Spark TTS] Finalizing: ending open segment")
                    output_emitter.end_segment()
                print(f"[Spark TTS] Receive task completed - received {segments_received} total segments with audio")

        ws = None
        try:
            ws = await self._tts._connect_ws(self._conn_options.timeout)
            print("[Spark TTS] WebSocket connected, starting text and audio tasks...")
            
            # Create both tasks
            llm_task = asyncio.create_task(_llm_text_task(ws))
            recv_task = asyncio.create_task(_recv_task(ws))
            
            try:
                # Wait for BOTH tasks to complete
                # LLM task will finish when all text is sent
                # Recv task will finish when server sends "finish" or closes
                done, pending = await asyncio.wait(
                    [llm_task, recv_task],
                    return_when=asyncio.ALL_COMPLETED,
                    timeout=300  # 5 minute timeout for entire stream
                )
                
                if pending:
                    print(f"[Spark TTS] Timeout: {len(pending)} task(s) still pending after 5 minutes")
                    for task in pending:
                        print(f"[Spark TTS] Cancelling pending task: {task.get_name()}")
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass
                
                # Check for exceptions in completed tasks
                for task in done:
                    if task.exception():
                        exc = task.exception()
                        print(f"[Spark TTS] Task {task.get_name()} failed: {exc}")
                        raise exc
                
                print("[Spark TTS] Both tasks completed successfully")
                
            except asyncio.CancelledError:
                print("[Spark TTS] Main task orchestration cancelled")
                # Cancel remaining tasks
                for task in [llm_task, recv_task]:
                    if not task.done():
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass
                raise
            except Exception as e:
                print(f"[Spark TTS] Error during task execution: {e}")
                # Cancel remaining tasks
                for task in [llm_task, recv_task]:
                    if not task.done():
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass
                raise

        except asyncio.TimeoutError:
            print("[Spark TTS] Connection timeout")
            raise APITimeoutError() from None
        except (aiohttp.ClientResponseError, aiohttp.ClientConnectionError) as e:
            print(f"[Spark TTS] Connection error: {e}")
            raise APIConnectionError(f"Failed to connect to Spark TTS: {e}") from e
        except Exception as e:
            print(f"[Spark TTS] Unexpected error: {e}")
            raise APIConnectionError(f"Spark TTS error: {e}") from e
        finally:
            if ws is not None and not ws.closed:
                print("[Spark TTS] Closing WebSocket connection")
                try:
                    await self._tts._close_ws(ws)
                except Exception as e:
                    print(f"[Spark TTS] Error closing WebSocket: {e}")
            print("[Spark TTS] Stream processing completed")