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
        provider = os.getenv("VOICE_TTS_PROVIDER")

        # If provider not set, voice features are disabled
        if not provider:
            logger.debug("VOICE_TTS_PROVIDER not set, TTS disabled")
            return None

        try:
            config: dict = {}
            if provider == "openai":
                # Optional overrides for OpenAI TTS
                config = {
                    "voice": os.getenv("OPENAI_TTS_VOICE", "alloy"),
                    "model": os.getenv("OPENAI_TTS_MODEL", "tts-1"),
                }
            elif provider == "elevenlabs":
                # ElevenLabs-specific settings
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
                config = {
                    "voice_id": os.getenv("ELEVENLABS_VOICE_ID"),
                    "model_id": os.getenv("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2"),
                    "output_format": os.getenv("ELEVENLABS_OUTPUT_FORMAT", "mp3_44100_128"),
                    "language_code": os.getenv("ELEVENLABS_LANGUAGE_CODE"),
                    "voice_settings": voice_settings,
                }

            # Create instance with provider from environment
            # Validates provider and raises ValueError if invalid
            return cls(provider=cast(Provider, provider), **config)
        except Exception as e:
            # Log error but don't crash - allow app to continue without voice
            logger.error(f"Failed to create TTS provider: {e}", exc_info=True)
            return None

    def generate(self, text: str) -> bytes | None:
        """Generate speech from text.

        Delegates to the underlying provider implementation.

        Args:
            text: Text to convert to speech

        Returns:
            Audio bytes (format depends on provider), or None on failure
        """
        return self._provider.generate(text)

    def get_format(self) -> str:
        """Get audio format (MIME type) for this provider.

        Returns:
            MIME type string (e.g., "audio/mp3")
        """
        return self._provider.get_format()

