import os
import tempfile
import soundfile as sf
import librosa
import torch


def extract_speaker_from_reference(
    audio_path: str,
    audio_tokenizer,
    reference_text: str = None,
    device="cuda"
) -> tuple[torch.Tensor, torch.Tensor]:
    wav, sr = sf.read(audio_path)
    if sr != 16000:
        print(f"Resampling audio from {sr}Hz to 16000Hz...")
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
        temp_fd, temp_path = tempfile.mkstemp(suffix='.wav')
        os.close(temp_fd)
        sf.write(temp_path, wav, 16000)
        audio_path_to_use = temp_path
    else:
        audio_path_to_use = audio_path

    try:
        global_ids, semantic_ids = audio_tokenizer.tokenize(audio_path_to_use)
        if not isinstance(global_ids, torch.Tensor):
            global_ids = torch.tensor(global_ids).long()
        if not isinstance(semantic_ids, torch.Tensor):
            semantic_ids = torch.tensor(semantic_ids).long()
        if global_ids.dim() > 1:
            global_ids = global_ids.squeeze()
        if semantic_ids.dim() > 1:
            semantic_ids = semantic_ids.squeeze()
        return global_ids, semantic_ids
    finally:
        if sr != 16000 and os.path.exists(temp_path):
            os.unlink(temp_path)


def convert_to_pcm16_bytes(audio_np) -> bytes:
    import numpy as np
    audio_int16 = (audio_np * 32767).astype(np.int16)
    return audio_int16.tobytes()
