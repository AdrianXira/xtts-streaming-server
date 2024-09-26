# # Corrijo el error de indentación causado por el bloque de comentarios multilinea

# code = '''
import base64
import io
import os
import tempfile
import wave
import torch
import numpy as np
from typing import List
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, Body
from fastapi.responses import StreamingResponse
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.utils.generic_utils import get_user_data_dir
from TTS.utils.manage import ModelManager


# Obtén la variable de entorno USE_CPU
use_cpu = os.getenv('USE_CPU', '1')  # Si no está definida, por defecto será '0'

# Set number of threads based on environment or fallback to os.cpu_count()
num_threads = os.environ.get("NUM_THREADS")
if num_threads is None or not num_threads.isdigit():
    torch.set_num_threads(os.cpu_count())
else:
    torch.set_num_threads(int(num_threads))

# device = torch.device("cuda" if os.environ.get("USE_CPU", "0") == "0" else "cpu")
device = "cpu"
if not torch.cuda.is_available() and device == "cuda":
    raise RuntimeError("CUDA device unavailable, please use Dockerfile.cpu instead.")

# Model path setup
custom_model_path = os.environ.get("CUSTOM_MODEL_PATH", "/app/tts_models")

if os.path.exists(custom_model_path) and os.path.isfile(custom_model_path + "/config.json"):
    model_path = custom_model_path
    print("Loading custom model from", model_path, flush=True)
else:
    print("Loading default model", flush=True)
    model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
    print("Downloading XTTS Model:", model_name, flush=True)
    ModelManager().download_model(model_name)
    model_path = os.path.join(get_user_data_dir("tts"), model_name.replace("/", "--"))
    print("XTTS Model downloaded", flush=True)

# Loading XTTS model
print("Loading XTTS", flush=True)
config = XttsConfig()
try:
    config.load_json(os.path.join(model_path, "config.json"))
except FileNotFoundError as e:
    raise RuntimeError("Config file not found") from e

model = Xtts.init_from_config(config)
model.load_checkpoint(
    config,
    checkpoint_dir=model_path,
    eval=True,
    use_deepspeed=True if device == "cuda" else False
)
model.to(device)
print("XTTS Loaded.", flush=True)

print("Running XTTS Server ...", flush=True)
# FastAPI app initialization
app = FastAPI(
    title="XTTS Streaming server",
    description="API for generating text-to-speech (TTS) using XTTS models. Supports real-time streaming and non-streaming TTS generation.",
    version="0.0.1",
    docs_url="/",
)

@app.post("/clone_speaker", summary="Clone speaker from an audio file")
def predict_speaker(wav_file: UploadFile):
    """
    This endpoint extracts conditioned latents and speaker embeddings from a reference audio file.

    **Parameter**:
    - `wav_file` (UploadFile): A `.wav` audio file used as the reference for cloning the speaker's voice characteristics.

    **Response**:
    - Returns a JSON structure with `gpt_cond_latent` (conditioned latents) and `speaker_embedding` (speaker embedding).
    """
    with tempfile.NamedTemporaryFile(delete=False) as temp, torch.inference_mode():
        temp.write(io.BytesIO(wav_file.file.read()).getbuffer())
        temp_audio_name = temp.name
    
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(temp_audio_name)
    
    return {
        "gpt_cond_latent": gpt_cond_latent.cpu().squeeze().half().tolist(),
        "speaker_embedding": speaker_embedding.cpu().squeeze().half().tolist(),
    }

def postprocess(wav):
    """Post-processes the output waveform."""
    if isinstance(wav, list):
        wav = torch.cat(wav, dim=0)
    wav = wav.clone().detach().cpu().numpy()
    wav = wav[None, : int(wav.shape[0])]
    wav = np.clip(wav, -1, 1)
    wav = (wav * 32767).astype(np.int16)
    return wav

def encode_audio_common(
    frame_input, encode_base64=True, sample_rate=24000, sample_width=2, channels=1
):
    """Return base64 encoded audio."""
    wav_buf = io.BytesIO()
    with wave.open(wav_buf, "wb") as vfout:
        vfout.setnchannels(channels)
        vfout.setsampwidth(sample_width)
        vfout.setframerate(sample_rate)
        vfout.writeframes(frame_input)

    wav_buf.seek(0)
    if encode_base64:
        return base64.b64encode(wav_buf.getbuffer()).decode("utf-8")
    return wav_buf.read()

class StreamingInputs(BaseModel):
    """Input model for real-time text-to-speech streaming."""
    speaker_embedding: List[float]
    gpt_cond_latent: List[List[float]]
    text: str
    language: str
    add_wav_header: bool = True
    stream_chunk_size: str = "20"

def predict_streaming_generator(parsed_input: dict = Body(...)):
    speaker_embedding = torch.tensor(parsed_input.speaker_embedding).unsqueeze(0).unsqueeze(-1)
    gpt_cond_latent = torch.tensor(parsed_input.gpt_cond_latent).reshape(-1, 1024).unsqueeze(0)
    text = parsed_input.text
    language = parsed_input.language

    stream_chunk_size = int(parsed_input.stream_chunk_size)
    add_wav_header = parsed_input.add_wav_header

    chunks = model.inference_stream(
        text,
        language,
        gpt_cond_latent,
        speaker_embedding,
        stream_chunk_size=stream_chunk_size,
        enable_text_splitting=True
    )

    for i, chunk in enumerate(chunks):
        chunk = postprocess(chunk)
        if i == 0 and add_wav_header:
            yield encode_audio_common(b"", encode_base64=False)
            yield chunk.tobytes()
        else:
            yield chunk.tobytes()

@app.post("/tts_ulaw", summary="Real-time TTS streaming in uLaw format")
def predict_ulaw_streaming_endpoint(parsed_input: StreamingInputs):
    """
    Generates real-time text-to-speech audio in uLaw format at 8kHz based on text and speaker parameters.

    **Parameters**:
    - `speaker_embedding` (List[float]): Speaker embedding vector.
    - `gpt_cond_latent` (List[List[float]]): Conditioned latents that inform the model about the style and content of the speech.
    - `text` (str): The text to be converted to audio.
    - `language` (str): The language in which the speech will be generated.
    - `add_wav_header` (bool): Whether to add a WAV header to the first streamed chunk.
    - `stream_chunk_size` (str): The size of the audio chunk for each stream (in milliseconds).

    **Response**:
    - A streaming audio response in uLaw format (encoded as Base64) at 8kHz.
    """

    def ulaw_encode(audio, sample_rate=8000):
        """Encode the audio in uLaw format with 8kHz sampling rate."""
        wav_buf = io.BytesIO()

        # Open the buffer as a wave file with 8kHz sampling and 8-bit width
        with wave.open(wav_buf, "wb") as vfout:
            vfout.setnchannels(1)  # Mono channel
            vfout.setsampwidth(1)  # 8-bit (uLaw encoded)
            vfout.setframerate(sample_rate)  # 8kHz sampling rate
            vfout.writeframes(audio)

        # Return the uLaw encoded audio
        return base64.b64encode(wav_buf.getvalue()).decode("utf-8")

    def predict_ulaw_streaming_generator(parsed_input):
        speaker_embedding = torch.tensor(parsed_input["speaker_embedding"]).unsqueeze(0).unsqueeze(-1)
        gpt_cond_latent = torch.tensor(parsed_input["gpt_cond_latent"]).reshape(-1, 1024).unsqueeze(0)
        text = parsed_input["text"]
        language = parsed_input["language"]

        stream_chunk_size = int(parsed_input["stream_chunk_size"])
        add_wav_header = parsed_input["add_wav_header"]

        chunks = model.inference_stream(
            text,
            language,
            gpt_cond_latent,
            speaker_embedding,
            stream_chunk_size=stream_chunk_size,
            enable_text_splitting=True
        )

        for i, chunk in enumerate(chunks):
            chunk = postprocess(chunk)
            if i == 0 and add_wav_header:
                yield ulaw_encode(b"", sample_rate=8000)
                yield ulaw_encode(chunk.tobytes(), sample_rate=8000)
            else:
                yield ulaw_encode(chunk.tobytes(), sample_rate=8000)

    return StreamingResponse(
        predict_ulaw_streaming_generator(parsed_input.dict()),
        media_type="audio/ulaw"
    )

@app.post("/tts_stream", summary="Real-time TTS streaming")
def predict_streaming_endpoint(parsed_input: StreamingInputs):
    """
    Generates real-time text-to-speech audio based on text and speaker parameters.

    **Parameters**:
    - `speaker_embedding` (List[float]): Speaker embedding vector.
    - `gpt_cond_latent` (List[List[float]]): Conditioned latents that inform the model about the style and content of the speech.
    - `text` (str): The text to be converted to audio.
    - `language` (str): The language in which the speech will be generated.
    - `add_wav_header` (bool): Whether to add a WAV header to the first streamed chunk.
    - `stream_chunk_size` (str): The size of the audio chunk for each stream (in milliseconds).

    **Response**:
    - A streaming audio response in `audio/wav` format.
    """
    return StreamingResponse(
        predict_streaming_generator(parsed_input.dict()),
        media_type="audio/wav",
    )

class TTSInputs(BaseModel):
    """Input model for non-streaming text-to-speech."""
    speaker_embedding: List[float]
    gpt_cond_latent: List[List[float]]
    text: str
    language: str

@app.post("/tts", summary="Generate audio from text")
def predict_speech(parsed_input: TTSInputs):
    """
    Generates a speech audio file from text and speaker parameters.

    **Parameters**:
    - `speaker_embedding` (List[float]): Speaker embedding vector.
    - `gpt_cond_latent` (List[List[float]]): Conditioned latents that inform the model about the style and content of the speech.
    - `text` (str): The text to be converted to audio.
    - `language` (str): The language in which the speech will be generated.

    **Response**:
    - Returns a Base64-encoded WAV audio file.
    """
    speaker_embedding = torch.tensor(parsed_input.speaker_embedding).unsqueeze(0).unsqueeze(-1)
    gpt_cond_latent = torch.tensor(parsed_input.gpt_cond_latent).reshape(-1, 1024).unsqueeze(0)
    text = parsed_input.text
    language = parsed_input.language

    out = model.inference(
        text,
        language,
        gpt_cond_latent,
        speaker_embedding,
    )

    wav = postprocess(torch.tensor(out["wav"]))
    return encode_audio_common(wav.tobytes())

@app.get("/studio_speakers", summary="Get preconfigured speakers")
def get_speakers():
    """
    Returns the available preconfigured speakers in the system.

    **Response**:
    - A dictionary of speaker names and their respective `speaker_embedding` and `gpt_cond_latent`.
    """
    if hasattr(model, "speaker_manager") and hasattr(model.speaker_manager, "speakers"):
        return {
            speaker: {
                "speaker_embedding": model.speaker_manager.speakers[speaker]["speaker_embedding"].cpu().squeeze().half().tolist(),
                "gpt_cond_latent": model.speaker_manager.speakers[speaker]["gpt_cond_latent"].cpu().squeeze().half().tolist(),
            }
            for speaker in model.speaker_manager.speakers.keys()
        }
    return {}

@app.get("/languages", summary="Get available languages")
def get_languages():
    """
    Returns a list of languages supported by the TTS model.

    **Response**:
    - A list of available languages for speech synthesis.
    """
    return config.languages
