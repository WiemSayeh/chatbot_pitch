# app.py
import os
import tempfile
import pathlib
import shutil
import numpy as np
import traceback
import logging
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Audio processing
from pydub import AudioSegment  # requires ffmpeg installed
import soundfile as sf
import noisereduce as nr

# Whisper
import whisper
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ---------- Configuration ----------
MODEL_SIZE = os.environ.get("MODEL_SIZE", "small")
TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1

# Whisper decoding parameters
DEFAULT_TEMPERATURE = 0.0
DEFAULT_BEAM_SIZE = 5
CONDITION_ON_PREVIOUS_TEXT = False

# Denoise default
DEFAULT_DENOISE = True

# File size limit (50MB default)
MAX_FILE_SIZE_MB = int(os.environ.get("MAX_FILE_SIZE_MB", 50))
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# Supported audio formats
SUPPORTED_FORMATS = {
    'audio/mpeg', 'audio/mp3', 'audio/wav', 'audio/x-wav',
    'audio/ogg', 'audio/webm', 'audio/flac', 'audio/m4a',
    'audio/x-m4a', 'audio/mp4', 'video/mp4'
}

# Global model variable
model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    global model
    logger.info(f"Loading Whisper model '{MODEL_SIZE}'...")
    try:
        model = whisper.load_model(MODEL_SIZE)
        logger.info(f"Model '{MODEL_SIZE}' loaded successfully")
        # Log device being used
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    
    yield
    
    # Cleanup on shutdown
    logger.info("Shutting down...")


app = FastAPI(
    title="Whisper Transcription Service",
    description="Audio transcription API using OpenAI Whisper",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Utility functions ----------

def validate_audio_file(file: UploadFile, content: bytes) -> None:
    """Validate uploaded audio file."""
    # Check file size
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {MAX_FILE_SIZE_MB}MB"
        )
    
    # Check content type
    if file.content_type and file.content_type not in SUPPORTED_FORMATS:
        logger.warning(f"Unsupported content type: {file.content_type}")
        # Don't reject - pydub might still handle it


def convert_to_wav_16k_mono(input_path: str, output_path: str) -> None:
    """Convert arbitrary audio to 16kHz mono WAV using pydub."""
    try:
        audio = AudioSegment.from_file(input_path)
        audio = (audio
                .set_frame_rate(TARGET_SAMPLE_RATE)
                .set_channels(TARGET_CHANNELS)
                .set_sample_width(2))
        audio.export(output_path, format="wav")
        logger.info(f"Converted audio: {len(audio)/1000:.2f}s duration")
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        raise


def denoise_audio(input_path: str, output_path: str) -> None:
    """Apply noise reduction to audio file."""
    try:
        data, sr = sf.read(input_path)
        
        # Convert stereo to mono if needed
        if data.ndim == 2:
            data = np.mean(data, axis=1)
        
        # Apply noise reduction with conservative settings
        reduced = nr.reduce_noise(
            y=data,
            sr=sr,
            stationary=True,
            prop_decrease=0.8
        )
        
        sf.write(output_path, reduced, sr)
        logger.info("Noise reduction applied successfully")
    except Exception as e:
        logger.warning(f"Noise reduction failed: {e}, using original audio")
        # Copy original file if denoising fails
        shutil.copy2(input_path, output_path)


def compute_confidence(segments: list) -> Optional[float]:
    """
    Compute aggregated confidence score from Whisper segments.
    Returns confidence in [0,1] or None if unavailable.
    """
    if not segments:
        return None
    
    logprobs = []
    for seg in segments:
        avg_logprob = seg.get("avg_logprob")
        if avg_logprob is not None:
            logprobs.append(avg_logprob)
    
    if not logprobs:
        return None
    
    mean_logprob = float(np.mean(logprobs))
    
    # Convert log probability to confidence using sigmoid
    # Adjust scaling factor for better range mapping
    confidence = 1.0 / (1.0 + np.exp(-mean_logprob * 2))
    
    return max(0.0, min(1.0, confidence))


def format_segments(segments: list) -> list:
    """Format segments for cleaner output."""
    formatted = []
    for seg in segments:
        formatted.append({
            "id": seg.get("id"),
            "start": round(seg.get("start", 0), 2),
            "end": round(seg.get("end", 0), 2),
            "text": seg.get("text", "").strip(),
            "confidence": seg.get("avg_logprob")
        })
    return formatted


# ---------- Endpoints ----------

@app.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "service": "Whisper Transcription API",
        "version": "1.0.0",
        "model": MODEL_SIZE,
        "endpoints": {
            "health": "/health",
            "transcribe": "/transcribe (POST)"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    try:
        gpu_available = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if gpu_available else None
    except Exception:
        gpu_available = False
        gpu_name = None
    
    return {
        "status": "healthy",
        "model": MODEL_SIZE,
        "model_loaded": model is not None,
        "gpu_available": gpu_available,
        "gpu_name": gpu_name,
        "max_file_size_mb": MAX_FILE_SIZE_MB
    }


@app.post("/transcribe")
async def transcribe(
    audio: UploadFile = File(..., description="Audio file to transcribe"),
    language: Optional[str] = Form(None, description="Language code (e.g., 'en', 'fr')"),
    denoise: Optional[bool] = Form(DEFAULT_DENOISE, description="Apply noise reduction"),
    temperature: Optional[float] = Form(DEFAULT_TEMPERATURE, description="Sampling temperature"),
    beam_size: Optional[int] = Form(DEFAULT_BEAM_SIZE, description="Beam search size"),
    condition_on_previous_text: Optional[bool] = Form(
        CONDITION_ON_PREVIOUS_TEXT,
        description="Condition on previous text"
    ),
    return_segments: Optional[bool] = Form(True, description="Include segment timestamps"),
):
    """
    Transcribe an audio file using Whisper.
    
    Returns:
        JSON with transcription text, language, confidence, and optional segments.
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please try again later."
        )
    
    tmp_dir = None
    
    try:
        # Read and validate file
        content = await audio.read()
        validate_audio_file(audio, content)
        
        # Create temp directory
        tmp_dir = tempfile.mkdtemp(prefix="whisper_")
        uploaded_path = os.path.join(tmp_dir, "input")
        converted_path = os.path.join(tmp_dir, "converted.wav")
        cleaned_path = os.path.join(tmp_dir, "cleaned.wav")
        
        # Save uploaded file
        with open(uploaded_path, "wb") as f:
            f.write(content)
        
        logger.info(f"Processing file: {audio.filename} ({len(content)/1024:.1f}KB)")
        
        # Convert to standard format
        convert_to_wav_16k_mono(uploaded_path, converted_path)
        
        # Apply denoising if requested
        final_path = converted_path
        if denoise:
            denoise_audio(converted_path, cleaned_path)
            final_path = cleaned_path
        
        # Prepare transcription parameters
        transcribe_kwargs = {
            "temperature": float(temperature),
            "beam_size": int(beam_size),
            "condition_on_previous_text": bool(condition_on_previous_text),
            "verbose": False,  # Reduce console output
        }
        
        # --- Improved automatic language detection (English/French only) ---
        logger.info("Auto-detecting language (restricted to English/French)")

        try:
            # Load and prepare audio for language detection
            audio_data = whisper.load_audio(final_path)
            audio_data = whisper.pad_or_trim(audio_data)
            mel = whisper.log_mel_spectrogram(audio_data).to(model.device)

            # Run Whisper's language detection
            _, lang_probs = model.detect_language(mel)

            # Keep only English and French probabilities
            filtered_probs = {k: v for k, v in lang_probs.items() if k in ["en", "fr"]}

            if filtered_probs:
                detected_language = max(filtered_probs, key=filtered_probs.get)
            else:
                detected_language = max(lang_probs, key=lang_probs.get)  # fallback to any

            logger.info(f"Detected language (restricted): {detected_language}")
            transcribe_kwargs["language"] = detected_language

        except Exception as e:
            logger.warning(f"Language detection failed ({e}), defaulting to English")
            transcribe_kwargs["language"] = "en"
        
        # Transcribe
        logger.info("Starting transcription...")
        result = model.transcribe(final_path, **transcribe_kwargs)
        logger.info("Transcription complete")
        
        # Extract results
        text = result.get("text", "").strip()
        detected_language = result.get("language")
        segments = result.get("segments", [])
        
        # Compute confidence
        confidence = compute_confidence(segments)
        
        # Prepare response
        response = {
            "success": True,
            "text": text,
            "language": detected_language,
            "confidence": confidence,
            "word_count": len(text.split()) if text else 0,
        }
        
        if return_segments and segments:
            response["segments"] = format_segments(segments)
        
        return JSONResponse(response)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Transcription failed: {str(e)}"
        )
    finally:
        # Cleanup temp directory
        if tmp_dir and os.path.exists(tmp_dir):
            try:
                shutil.rmtree(tmp_dir)
            except Exception as e:
                logger.warning(f"Failed to cleanup temp dir: {e}")


# ---------- Run ----------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "speech_recognizer:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )




    #uvicorn speech.speech_to_text:app --host 0.0.0.0 --port 8000 --reload