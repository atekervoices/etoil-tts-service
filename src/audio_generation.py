import re
import numpy as np
import torch
from vllm.sampling_params import SamplingParams
from .config import DEFAULT_MAX_TOKENS
from .speaker_tokens import GLOBAL_IDS_BY_SPEAKER
from .text_processing import chunk_text_simple
from .audio_processing import convert_to_pcm16_bytes
from . import model_loader


def text_to_speech_cloned(
    text: str,
    audio_tokenizer,
    model,
    reference_audio_path: str,
    reference_text: str = None,
    temperature: float = 0.7,
    device="cuda"
):
    try:
        print(f"Starting voice cloning for text: '{text[:50]}...'")
        print(f"Reference audio path: {reference_audio_path}")

        texts = chunk_text_simple(text)
        texts = [t.strip() for t in texts if len(t.strip()) > 0]
        print(f"Text split into {len(texts)} chunks: {texts}")

        if not texts:
            raise ValueError("No valid text chunks found after processing")

        sampling_kwargs = {
            "temperature": temperature,
            "max_tokens": 2048,
            "repetition_penalty": 1.05,
        }
        if model_loader.end_semantic_token_id is not None:
            sampling_kwargs["stop_token_ids"] = [model_loader.end_semantic_token_id]
        sampling_params = SamplingParams(**sampling_kwargs)

        print("Extracting speaker features from reference audio...")
        try:
            from .audio_processing import extract_speaker_from_reference
            global_ids_ref, semantic_ids_ref = extract_speaker_from_reference(
                reference_audio_path, audio_tokenizer, reference_text, device
            )
            print(f"Successfully extracted speaker features")
            print(f"Global IDs shape: {global_ids_ref.shape}, Semantic IDs shape: {semantic_ids_ref.shape}")
        except Exception as e:
            raise ValueError(f"Failed to extract speaker features from reference audio: {str(e)}")

        global_ids_list = global_ids_ref.cpu().tolist()
        if isinstance(global_ids_list, int):
            global_ids_list = [global_ids_list]

        print(f"Extracted {len(global_ids_list)} global tokens from reference")

        prompts = []
        for i, chunk in enumerate(texts):
            prompt = f"<|task_tts|><|start_content|>{chunk}<|end_content|><|start_global_token|>"
            prompt += ''.join([f'<|bicodec_global_{t}|>' for t in global_ids_list])
            prompt += '<|end_global_token|><|start_semantic_token|>'
            prompts.append(prompt)
            print(f"Generated prompt {i+1}/{len(texts)}: {prompt[:100]}...")

        print("Generating speech with model...")
        try:
            outputs = model.generate(prompts=prompts, sampling_params=sampling_params)
            print(f"Model generation completed. Generated {len(outputs)} outputs")
        except Exception as e:
            raise ValueError(f"Model generation failed: {str(e)}")

        speech_segments = []
        for i, output in enumerate(outputs):
            print(f"Processing output {i+1}/{len(outputs)}")
            predicted_tokens = output.outputs[0].text
            print(f"Raw model output: {predicted_tokens[:200]}...")

            semantic_matches = re.findall(r"<\|bicodec_semantic_(\d+)\|>", predicted_tokens)
            print(f"Found {len(semantic_matches)} semantic tokens")

            if not semantic_matches:
                raise ValueError(f"No semantic tokens found in output {i+1}. Raw output: {predicted_tokens}")

            try:
                pred_semantic_ids = torch.tensor([int(t) for t in semantic_matches]).long().unsqueeze(0)
                pred_global_ids = torch.tensor([global_ids_list]).long()

                print(f"Detokenizing audio: semantic shape={pred_semantic_ids.shape}, global shape={pred_global_ids.shape}")

                wav_np = audio_tokenizer.detokenize(
                    pred_global_ids.to(device),
                    pred_semantic_ids.to(device)
                )

                print(f"Generated audio segment {i+1}: shape={wav_np.shape}")
                speech_segments.append(wav_np)

            except Exception as e:
                raise ValueError(f"Audio detokenization failed for segment {i+1}: {str(e)}")

        if not speech_segments:
            raise ValueError("No speech segments were generated")

        result_wav = np.concatenate(speech_segments)
        print(f"Successfully concatenated {len(speech_segments)} segments. Final shape: {result_wav.shape}")
        return result_wav

    except Exception as e:
        print(f"ERROR in text_to_speech_cloned: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


def generate_audio_segment(text: str, speaker_id: str, temperature: float) -> np.ndarray:
    """Generate audio for a single text segment using a string speaker_id."""
    if model_loader.vllm_model is None:
        raise RuntimeError("Model not initialized. Please wait for server startup to complete.")
    if model_loader.audio_tokenizer is None:
        raise RuntimeError("Audio tokenizer not initialized. Please wait for server startup to complete.")
    if speaker_id not in GLOBAL_IDS_BY_SPEAKER:
        raise ValueError(f"Unknown speaker_id: '{speaker_id}'. Valid IDs: {list(GLOBAL_IDS_BY_SPEAKER.keys())}")

    global_tokens = GLOBAL_IDS_BY_SPEAKER[speaker_id]

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    prompt = f"<|task_tts|><|start_content|>{speaker_id}: {text}<|end_content|><|start_global_token|>"
    prompt += ''.join([f'<|bicodec_global_{t}|>' for t in global_tokens])
    prompt += '<|end_global_token|><|start_semantic_token|>'

    # Scale max_tokens based on text length to prevent runaway generation
    # ~15 semantic tokens per character is a generous upper bound
    text_based_max = max(256, min(len(text) * 15, DEFAULT_MAX_TOKENS))
    
    # Build sampling params with stop token if available
    sampling_kwargs = {
        "temperature": temperature,
        "max_tokens": text_based_max,
        "repetition_penalty": 1.05,  # Slight penalty to prevent token loops
    }
    
    if model_loader.end_semantic_token_id is not None:
        sampling_kwargs["stop_token_ids"] = [model_loader.end_semantic_token_id]
    
    sampling_params = SamplingParams(**sampling_kwargs)
    outputs = model_loader.vllm_model.generate(prompts=[prompt], sampling_params=sampling_params)

    predicted_tokens = outputs[0].outputs[0].text
    finish_reason = getattr(outputs[0].outputs[0], 'finish_reason', 'unknown')
    semantic_matches = re.findall(r"<\|bicodec_semantic_(\d+)\|>", predicted_tokens)

    if not semantic_matches:
        raise ValueError("No semantic tokens found in the generated output.")

    print(f"Generated {len(semantic_matches)} semantic tokens for '{text[:40]}...' (finish_reason={finish_reason}, max_allowed={text_based_max})")

    # Safety cap - should rarely trigger now with proper stop tokens
    MAX_SEMANTIC_TOKENS = 800
    if len(semantic_matches) > MAX_SEMANTIC_TOKENS:
        semantic_matches = semantic_matches[:MAX_SEMANTIC_TOKENS]
        print(f"⚠️ Limited semantic tokens to {MAX_SEMANTIC_TOKENS} for memory safety (text: '{text[:40]}...')")

    pred_semantic_ids = (
        torch.tensor([int(token) for token in semantic_matches]).long().unsqueeze(0)
    )
    pred_global_ids = torch.tensor([global_tokens]).long()

    with torch.no_grad():
        wav_np = model_loader.audio_tokenizer.detokenize(
            pred_global_ids.to(model_loader.device), pred_semantic_ids.to(model_loader.device)
        )

    del pred_semantic_ids, pred_global_ids
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        import gc
        gc.collect()

    return wav_np
