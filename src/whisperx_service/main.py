"""Modal web endpoint for audio transcription with WhisperX."""

import os
import logging
import uuid
from datetime import datetime
from typing import Optional

import httpx
import modal
from fastapi import HTTPException, Response
from pydantic import BaseModel, HttpUrl

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ✅ Modal Secret (must exist in Modal: "huggingface")
HF_SECRET = modal.Secret.from_name("huggingface")

# Create Modal image with WhisperX dependencies
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "ffmpeg", "libcudnn8", "libcudnn8-dev")
    .pip_install(
        "torch==2.8.0",
        "torchaudio==2.8.0",
        "torchvision==0.23.0",
        extra_options="--extra-index-url https://download.pytorch.org/whl/cu128",
    )
    .pip_install(
        [
            "whisperx",
            "fastapi[standard]",
            "pydantic",
            "httpx",
            "omegaconf",
        ]

    )
    .env({
        "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "true"
    })
)

# Create Modal app
app = modal.App("whisper-transcription", image=image)


class TranscriptionRequest(BaseModel):
    """Request model for audio transcription."""

    audio_url: HttpUrl
    callback_url: HttpUrl


class TranscriptionResponse(BaseModel):
    """Response model for transcription request."""

    status: str
    message: str
    request_id: str


class CallbackPayload(BaseModel):
    """Payload sent to callback URL when transcription is complete."""

    request_id: str
    status: str
    message: str
    timestamp: str
    result: Optional[dict] = None
    result_align: Optional[dict] = None
    result_diarize: Optional[dict] = None
    error: Optional[str] = None


@app.cls(gpu="A10G", timeout=60 * 10, retries=1, scaledown_window=30, secrets=[HF_SECRET])
class WhisperXModel:
    """WhisperX model for audio transcription."""

    @modal.enter()
    def setup(self):
        """Load WhisperX model on container startup."""
        import os
        import whisperx
        import torch
        import inspect
        from importlib.metadata import version, PackageNotFoundError

        logger.info(f"Torch version: {torch.__version__}")

        # ✅ Hugging Face token from Modal Secret (runtime)
        hf_token = os.environ.get("HUGGINGFACE_ACCESS_TOKEN")
        if not hf_token:
            raise RuntimeError(
                "HUGGINGFACE_ACCESS_TOKEN is missing. "
                "Create Modal secret: modal secret create huggingface HUGGINGFACE_ACCESS_TOKEN=..."
            )

        try:
            wx_version = version("whisperx")  # nombre del distribution en pip
        except PackageNotFoundError:
            wx_version = "unknown (distribution not found)"

        logger.info(f"WhisperX version: {wx_version}")
        logger.info(f"WhisperX module path: {whisperx.__file__}")

        # ✅ Torch>=2.6 safe-unpickling allowlist (pyannote checkpoints)
        try:
            from omegaconf import DictConfig, ListConfig

            if hasattr(torch.serialization, "add_safe_globals"):
                torch.serialization.add_safe_globals([ListConfig, DictConfig])
                logger.info("Added OmegaConf safe globals (ListConfig, DictConfig).")
            else:
                logger.warning("torch.serialization.add_safe_globals not found; skipping.")
        except Exception as e:
            logger.warning(f"Could not add OmegaConf safe globals: {e}")

        self.device = "cuda"
        self.model_name = "large-v2"
        self.batch_size = 16
        self.compute_type = "float32" # change to "int8" if low on GPU mem (may reduce accuracy)

        # ✅ decoding / ASR config goes HERE (not in transcribe())
        self.asr_options = {
            "beam_size": 5,  # <-- set beam size here
            "without_timestamps": False,
            # often improves quality vs WhisperX default :contentReference[oaicite:1]{index=1}
            "condition_on_previous_text": True  # closer behavior to "plain Whisper" continuity
        }

        logger.info(f"Loading WhisperX model: {self.model_name}")
        self.model = whisperx.load_model(
            self.model_name,
            self.device,
            compute_type=self.compute_type,
            asr_options=self.asr_options,
            vad_method="silero",
        )
        logger.info("WhisperX model loaded successfully")
        # ✅ Pega esto aquí
        logger.info(f"transcribe signature: {inspect.signature(self.model.transcribe)}")

    @modal.method()
    def transcribe(self, audio_url: str) -> dict:
        """
        Transcribe audio from URL using WhisperX.

        Args:
            audio_url: URL of the audio file to transcribe

        Returns:
            dict: Transcription result with segments and metadata
        """
        import os
        import tempfile
        from urllib.parse import urlparse

        import whisperx

        try:
            # Download audio file
            logger.info(f"Downloading audio from: {audio_url}")

            with httpx.Client(timeout=60.0) as client:
                response = client.get(audio_url)
                response.raise_for_status()

            # Create temporary file
            parsed_url = urlparse(audio_url)
            filename = os.path.basename(parsed_url.path) or "audio.mp3"

            with tempfile.NamedTemporaryFile(
                suffix=f"_{filename}", delete=False
            ) as tmp_file:
                tmp_file.write(response.content)
                temp_path = tmp_file.name

            logger.info(f"Audio downloaded to: {temp_path}")

            # Load audio
            audio = whisperx.load_audio(temp_path)

            # Transcribe
            logger.info("Starting transcription...")
            result = self.model.transcribe(audio, batch_size=self.batch_size)

            logger.info(f"Transcription result keys: {list(result.keys())}")

            # Get language from result or use default
            detected_language = result.get("language", "pt")
            segments = result.get("segments", [])

            # Skip word-level alignment - only return segment-level timestamps

            # Clean up temp file
            os.unlink(temp_path)

            logger.info("Transcription completed successfully")

            return {
                "language": detected_language,
                "segments": segments,
                "duration": len(audio) / 16000,  # Assuming 16kHz sample rate
            }

        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            # Clean up temp file if it exists
            if "temp_path" in locals():
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass
            raise

    @modal.method()
    def transcribe_with_callback(
        self, request_id: str, audio_url: str, callback_url: str, hf_token: str,
    ) -> None:
        """
        Transcribe audio and send result to callback URL.

        Args:
            request_id: Unique identifier for this transcription request
            audio_url: URL of the audio file to transcribe
            callback_url: URL to send the transcription result to
        """
        import os
        import tempfile
        from urllib.parse import urlparse

        import whisperx
        from whisperx.diarize import DiarizationPipeline

        callback_payload = None

        try:
            logger.info(f"Starting transcription for request: {request_id}")
            logger.info(f"Audio URL: {audio_url}")
            logger.info(f"Callback URL: {callback_url}")

            # Perform transcription directly (can't call self.transcribe from within Modal method)
            # Download audio file
            logger.info(f"Downloading audio from: {audio_url}")

            with httpx.Client(timeout=60.0) as client:
                response = client.get(audio_url)
                response.raise_for_status()

                # Create temporary file
                parsed_url = urlparse(audio_url)
                filename = os.path.basename(parsed_url.path) or "audio.mp3"

                with tempfile.NamedTemporaryFile(
                        suffix=f"_{filename}", delete=False
                ) as tmp_file:
                    tmp_file.write(response.content)
                    temp_path = tmp_file.name

            logger.info(f"Audio downloaded to: {temp_path}")

            # Load audio
            audio = whisperx.load_audio(temp_path)

            # Transcribe
            logger.info("Starting transcription...")
            result = self.model.transcribe(
                audio,
                batch_size=self.batch_size,
                chunk_size=30,
                language="es",  # si ya lo sabes
                task="transcribe",
            )

            logger.info(f"Transcription result keys: {list(result.keys())}")

            # Get language from result or use default
            detected_language = result.get("language", "pt")
            segments = result.get("segments", [])

            # Skip word-level alignment - only return segment-level timestamps

            # Clean up temp file
            os.unlink(temp_path)

            logger.info("Transcription completed successfully")

            transcription_result = {
                "language": detected_language,
                "segments": segments,
                "duration": len(audio) / 16000,  # Assuming 16kHz sample rate
            }

            # 2. Align whisper output
            model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=self.device)
            result_align = whisperx.align(result["segments"], model_a, metadata, audio, self.device, return_char_alignments=False)

            logger.info(f"Resul Align segments: {result_align['segments']}")

            # delete model if low on GPU resources
            # import gc; import torch; gc.collect(); torch.cuda.empty_cache(); del model_a

            # 3. Assign speaker labels
            diarize_model = DiarizationPipeline(use_auth_token=hf_token, device=self.device)

            # add min/max number of speakers if known
            diarize_segments = diarize_model(audio)
            # diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)

            result_diarize = whisperx.assign_word_speakers(diarize_segments, result)
            print(diarize_segments)

            # Prepare success callback payload
            callback_payload = CallbackPayload(
                request_id=request_id,
                status="completed",
                message="Audio transcription completed successfully",
                timestamp=datetime.now().isoformat(),
                result=transcription_result,
                result_align=result_align,
                result_diarize=result_diarize,
            )

        except Exception as e:
            logger.error(f"Transcription failed for request {request_id}: {str(e)}")
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")

            # Clean up temp file if it exists
            if "temp_path" in locals():
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass

            # Prepare error callback payload
            callback_payload = CallbackPayload(
                request_id=request_id,
                status="failed",
                message="Audio transcription failed",
                timestamp=datetime.now().isoformat(),
                error=str(e),
            )

        # Call callback URL synchronously
        if callback_payload:
            try:
                logger.info(f"Sending callback to {callback_url}")
                success = call_callback_url_sync(callback_url, callback_payload)
                if success:
                    logger.info(f"Callback sent successfully for request: {request_id}")
                else:
                    logger.error(
                        f"Callback returned failure status for request: {request_id}"
                    )
            except Exception as e:
                logger.error(
                    f"Failed to send callback for request {request_id}: {str(e)}"
                )
                import traceback

                logger.error(f"Callback error traceback: {traceback.format_exc()}")


@app.function()
@modal.fastapi_endpoint(method="POST", secrets=[HF_SECRET])
async def transcribe_audio(
    request: TranscriptionRequest, response: Response
) -> TranscriptionResponse:
    """
    Web endpoint to receive audio file URL and callback URL.

    Returns immediately with 202 status while processing happens in background.
    """
    try:
        # Generate unique request ID
        request_id = str(uuid.uuid4())

        logger.info(
            f"Received transcription request {request_id} for URL: {request.audio_url}"
        )
        logger.info(f"Callback URL: {request.callback_url}")

        hf_token = os.environ.get("HUGGINGFACE_ACCESS_TOKEN")

        logger.info(f"HUGGINGFACE_ACCESS_TOKEN FASTAPI len: {len(hf_token)}")

        # Start background transcription task (fire and forget)
        # Use spawn() for true fire-and-forget behavior in modal
        WhisperXModel().transcribe_with_callback.spawn(
            request_id, str(request.audio_url), str(request.callback_url), hf_token,
        )

        logger.info(f"Background transcription started for request: {request_id}")

        # Set HTTP status code to 202 Accepted
        response.status_code = 202

        return TranscriptionResponse(
            status="accepted",
            message="Audio transcription request accepted and processing started",
            request_id=request_id,
        )

    except Exception as e:
        logger.error(f"Error processing transcription request: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Internal server error: {str(e)}"
        ) from e


def call_callback_url_sync(callback_url: str, payload: CallbackPayload) -> bool:
    """
    Call the callback URL to notify completion (sync version).

    Args:
        callback_url: The URL to call when processing is complete
        payload: The callback payload to send

    Returns:
        bool: True if callback was successful, False otherwise
    """
    try:
        # Make HTTP request to callback URL
        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                callback_url,
                json=payload.dict(),
                headers={"Content-Type": "application/json"},
            )

        if response.status_code in [200, 201, 202, 204]:
            logger.info(f"Callback successful: {response.status_code}")
            return True
        else:
            logger.warning(f"Callback failed with status: {response.status_code}")
            logger.warning(f"Response body: {response.text}")
            return False

    except httpx.TimeoutException:
        logger.error("Callback request timed out")
        return False
    except httpx.RequestError as e:
        logger.error(f"Callback request failed: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error calling callback: {str(e)}")
        return False


async def call_callback_url_async(callback_url: str, payload: CallbackPayload) -> bool:
    """
    Call the callback URL to notify completion (async version).

    Args:
        callback_url: The URL to call when processing is complete
        payload: The callback payload to send

    Returns:
        bool: True if callback was successful, False otherwise
    """
    try:
        # Make HTTP request to callback URL
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                callback_url,
                json=payload.dict(),
                headers={"Content-Type": "application/json"},
            )

            if response.status_code in [200, 201, 202, 204]:
                logger.info(f"Callback successful: {response.status_code}")
                return True
            else:
                logger.warning(f"Callback failed with status: {response.status_code}")
                logger.warning(f"Response body: {response.text}")
                return False

    except httpx.TimeoutException:
        logger.error("Callback request timed out")
        return False
    except httpx.RequestError as e:
        logger.error(f"Callback request failed: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error calling callback: {str(e)}")
        return False


@app.function()
@modal.fastapi_endpoint(method="GET")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "whisper-transcription"}
