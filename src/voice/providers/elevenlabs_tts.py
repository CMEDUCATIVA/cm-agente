"""ElevenLabs text-to-speech implementation."""

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class ElevenLabsTTS:
    """ElevenLabs TTS provider."""

    MIN_TEXT_LENGTH = 3

    def __init__(
        self,
        api_key: str | None = None,
        voice_id: str | None = None,
        model_id: str = "eleven_multilingual_v2",
        output_format: str = "mp3_44100_128",
        language_code: str | None = None,
        voice_settings: dict[str, Any] | None = None,
        base_url: str = "https://api.elevenlabs.io",
    ) -> None:
        """Initialize ElevenLabs TTS.

        Args:
            api_key: ElevenLabs API key
            voice_id: Voice ID to use
            model_id: Model ID (e.g. eleven_multilingual_v2)
            output_format: Output audio format (e.g. mp3_44100_128)
            language_code: Optional ISO 639-1 language code
            voice_settings: Optional voice settings dict
            base_url: ElevenLabs API base URL

        Raises:
            ValueError: If api_key or voice_id is missing
        """
        if not api_key:
            raise ValueError("ElevenLabs API key is required")
        if not voice_id:
            raise ValueError("ElevenLabs voice_id is required")

        self.api_key = api_key
        self.voice_id = voice_id
        self.model_id = model_id
        self.output_format = output_format
        self.language_code = language_code
        self.voice_settings = voice_settings
        self.base_url = base_url.rstrip("/")
        self._format = _guess_mime_type(output_format)

        logger.info(
            "ElevenLabs TTS initialized: voice_id=%s, model_id=%s, output_format=%s",
            voice_id,
            model_id,
            output_format,
        )

    def _prepare_text(self, text: str) -> str | None:
        text = text.strip()
        if len(text) < self.MIN_TEXT_LENGTH:
            logger.debug("ElevenLabs TTS: skipping short text (%s chars)", len(text))
            return None
        return text

    def generate(self, text: str) -> bytes | None:
        """Generate speech from text."""
        prepared_text = self._prepare_text(text)
        if not prepared_text:
            return None

        payload: dict[str, Any] = {"text": prepared_text, "model_id": self.model_id}
        if self.language_code:
            payload["language_code"] = self.language_code
        if self.voice_settings:
            payload["voice_settings"] = self.voice_settings

        url = f"{self.base_url}/v1/text-to-speech/{self.voice_id}"
        headers = {
            "xi-api-key": self.api_key,
            "accept": self._format,
            "content-type": "application/json",
        }
        params = {"output_format": self.output_format}

        try:
            response = httpx.post(url, headers=headers, params=params, json=payload, timeout=60)
            response.raise_for_status()
            audio_bytes = response.content
            logger.info("ElevenLabs TTS: generated %s bytes", len(audio_bytes))
            return audio_bytes
        except Exception as e:
            logger.error("ElevenLabs TTS failed: %s", e, exc_info=True)
            return None

    def get_format(self) -> str:
        """Get audio format (MIME type)."""
        return self._format


def _guess_mime_type(output_format: str) -> str:
    if output_format.startswith("mp3"):
        return "audio/mpeg"
    if output_format.startswith("pcm"):
        return "audio/wav"
    if output_format.startswith("ulaw") or output_format.startswith("mulaw"):
        return "audio/ulaw"
    return "application/octet-stream"
