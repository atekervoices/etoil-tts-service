from typing import Optional
from pydantic import BaseModel


class AudioRequest(BaseModel):
    text: str
    speaker_id: str = "pcm_female_1"
    temperature: float = 0.7
    max_tokens: int = 2048


class VoiceCloningRequest(BaseModel):
    text: str
    reference_audio_path: str
    reference_text: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2048
