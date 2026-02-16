"""Text-to-speech factory.

This module provides a factory class that loads the appropriate TTS provider
based on configuration.
"""

import json
import logging
import os
from typing import Literal, cast

logger = logging.getLogger(__name__)

Provider = Literal["openai", "elevenlabs"]


class TextToSpeech:
    """Text-to-speech factory.

    Loads and delegates to specific TTS provider implementations.

    Example:
        >>> tts = TextToSpeech(provider="openai", voice="nova")
        >>> audio = tts.generate("Hello world")
        >>>
        >>> # Or from environment
        >>> tts = TextToSpeech.from_env()
        >>> if tts:
        ...     audio = tts.generate("Hello world")
    """

    def __init__(self, provider: Provider = "openai", api_key: str | None = None, **config):
        """Initialize TTS with specified provider.

        Args:
            provider: Provider name ("openai", "elevenlabs", etc.)
            api_key: API key (uses env var if not provided)
            **config: Provider-specific configuration
                OpenAI: voice="alloy", model="tts-1"
                ElevenLabs: voice_id="...", model_id="..."

        Raises:
            ValueError: If provider is unknown
        """
        self._provider_name = provider

        # Resolve API key from parameter or environment
        resolved_api_key = self._get_api_key(provider, api_key)

        # Load and configure the provider
        self._provider = self._load_provider(provider, resolved_api_key, config)

        logger.info(f"TextToSpeech created with provider={provider}")

    def _get_api_key(self, provider: Provider, api_key: str | None) -> str | None:
        """Get API key from parameter or environment.

        Args:
            provider: Provider name
            api_key: API key from parameter (takes precedence)

        Returns:
            Resolved API key or None
        """
        # If API key provided explicitly, use it
        if api_key:
            return api_key

        # Otherwise, get from environment based on provider
        match provider:
            case "openai":
                return os.getenv("OPENAI_API_KEY")
            case "elevenlabs":
                return os.getenv("ELEVENLABS_API_KEY")
            case _:
                return None

    def _load_provider(self, provider: Provider, api_key: str | None, config: dict):
        """Load the appropriate TTS provider implementation.

        Args:
            provider: Provider name
            api_key: Resolved API key
            config: Provider-specific configuration

        Returns:
            Provider instance

        Raises:
            ValueError: If provider is unknown
            NotImplementedError: If provider not yet implemented
        """
        match provider:
            case "openai":
                from voice.providers.openai_tts import OpenAITTS

                # Extract OpenAI-specific config with defaults
                voice = config.get("voice", "alloy")
                model = config.get("model", "tts-1")

                return OpenAITTS(api_key=api_key, voice=voice, model=model)

            case "elevenlabs":
                from voice.providers.elevenlabs_tts import ElevenLabsTTS

                voice_id = config.get("voice_id")
                model_id = config.get("model_id", "eleven_multilingual_v2")
                output_format = config.get("output_format", "mp3_44100_128")
                language_code = config.get("language_code")
                voice_settings = config.get("voice_settings")

                return ElevenLabsTTS(
                    api_key=api_key,
                    voice_id=voice_id,
                    model_id=model_id,
                    output_format=output_format,
                    language_code=language_code,
                    voice_settings=voice_settings,
                )

            case _:
                # Catch-all for unknown providers
                raise ValueError(
                    f"Unknown TTS provider: {provider}. Available providers: openai, elevenlabs"
                )

    @property
    def provider(self) -> str:
        """Get the provider name.

        Returns:
            Provider name string
        """
        return self._provider_name

    @classmethod
    def from_env(cls) -> "TextToSpeech | None":
        """Create TTS from environment variables.

        Reads VOICE_TTS_PROVIDER env var to determine which provider to use.
        Returns None if not configured.

        Returns:
            TextToSpeech instance or None

        Example:
            >>> # In .env: VOICE_TTS_PROVIDER=openai
            >>> tts = TextToSpeech.from_env()
            >>> if tts:
            ...     audio = tts.generate("Hello world")
        """
        try:
            from dotenv import find_dotenv, load_dotenv

            load_dotenv(find_dotenv())
        except Exception:
            # If dotenv isn't available or fails, continue with existing env
            pass

        provider = os.getenv("VOICE_TTS_PROVIDER")

        # If provider not set, voice features are disabled
        if not provider:
            logger.debug("VOICE_TTS_PROVIDER not set, TTS disabled")
            return None

        def _config_for(p: str) -> dict:
            if p == "openai":
                return {
                    "voice": os.getenv("OPENAI_TTS_VOICE", "alloy"),
                    "model": os.getenv("OPENAI_TTS_MODEL", "tts-1"),
                }
            if p == "elevenlabs":
                voice_settings = None
                voice_settings_raw = os.getenv("ELEVENLABS_VOICE_SETTINGS")
                if voice_settings_raw:
                    try:
                        voice_settings = json.loads(voice_settings_raw)
                    except json.JSONDecodeError:
                        logger.error(
                            "Invalid ELEVENLABS_VOICE_SETTINGS JSON, ignoring it",
                            exc_info=True,
                        )
                return {
                    "voice_id": os.getenv("ELEVENLABS_VOICE_ID"),
                    "model_id": os.getenv("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2"),
                    "output_format": os.getenv("ELEVENLABS_OUTPUT_FORMAT", "mp3_44100_128"),
                    "language_code": os.getenv("ELEVENLABS_LANGUAGE_CODE"),
                    "voice_settings": voice_settings,
                }
            return {}

        try:
            config = _config_for(provider)
            return cls(provider=cast(Provider, provider), **config)
        except Exception as e:
            logger.error(f"Failed to create TTS provider: {e}", exc_info=True)
            # Fallback: if ElevenLabs fails, try OpenAI TTS
            if provider == "elevenlabs":
                try:
                    logger.warning("Falling back to OpenAI TTS")
                    fallback_config = _config_for("openai")
                    return cls(provider="openai", **fallback_config)
                except Exception as fallback_error:
                    logger.error(
                        f"Failed to create OpenAI TTS fallback: {fallback_error}",
                        exc_info=True,
                    )
            return None

    def generate(self, text: str) -> bytes | None:
        """Generate speech from text.

        Delegates to the underlying provider implementation.

        Args:
            text: Text to convert to speech

        Returns:
            Audio bytes (format depends on provider), or None on failure
        """
        audio = self._provider.generate(text)
        if audio is not None:
            return audio
        if self._provider_name == "elevenlabs":
            fallback = self._get_openai_fallback()
            if fallback:
                return fallback.generate(text)
        return None

    async def stream(self, text: str):
        """Stream speech audio from text when supported.

        Falls back to a single chunk if provider doesn't support streaming.
        """
        stream_fn = getattr(self._provider, "stream", None)
        if callable(stream_fn):
            yielded = False
            async for chunk in stream_fn(text):
                if chunk:
                    yielded = True
                    yield chunk
            if yielded or self._provider_name != "elevenlabs":
                return
            fallback = self._get_openai_fallback()
            if fallback:
                stream_fn = getattr(fallback, "stream", None)
                if callable(stream_fn):
                    async for chunk in stream_fn(text):
                        if chunk:
                            yield chunk
                else:
                    audio = fallback.generate(text)
                    if audio:
                        yield audio
            return
        audio = self._provider.generate(text)
        if audio:
            yield audio
            return
        if self._provider_name == "elevenlabs":
            fallback = self._get_openai_fallback()
            if fallback:
                audio = fallback.generate(text)
                if audio:
                    yield audio

    def get_format(self) -> str:
        """Get audio format (MIME type) for this provider.

        Returns:
            MIME type string (e.g., "audio/mp3")
        """
        return self._provider.get_format()

    def _get_openai_fallback(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("OpenAI API key not set; cannot fallback to OpenAI TTS")
            return None
        try:
            from voice.providers.openai_tts import OpenAITTS

            voice = os.getenv("OPENAI_TTS_VOICE", "alloy")
            model = os.getenv("OPENAI_TTS_MODEL", "tts-1")
            logger.warning("Falling back to OpenAI TTS at runtime")
            return OpenAITTS(api_key=api_key, voice=voice, model=model)
        except Exception as e:
            logger.error(f"Failed to create OpenAI TTS fallback: {e}", exc_info=True)
            return None
