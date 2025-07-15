"""Token counting module using tiktoken."""

import logging
from typing import Dict, Optional

import tiktoken

logger = logging.getLogger(__name__)


# Model to encoding mapping for custom/newer models
MODEL_ENCODING_MAP: Dict[str, str] = {
    # OpenAI GPT-4.1 family
    "gpt-4.1": "cl100k_base",
    "gpt-4.1-mini": "cl100k_base",
    "gpt-4.1-nano": "cl100k_base",
    "gpt-4.1-turbo": "cl100k_base",
    
    # OpenAI O3 family
    "o3": "cl100k_base",
    "o3-mini": "cl100k_base",
    
    # Common aliases
    "gpt4": "cl100k_base",
    "gpt-4-turbo": "cl100k_base",
    "gpt-4-32k": "cl100k_base",
    
    # Claude models (use cl100k_base as approximation)
    "claude-3-opus": "cl100k_base",
    "claude-3-sonnet": "cl100k_base",
    "claude-3-haiku": "cl100k_base",
}

# Default encoding to use when model is unknown
DEFAULT_ENCODING = "cl100k_base"


class Tokenizer:
    """Handles token counting for different models."""
    
    def __init__(self, model: str = DEFAULT_ENCODING) -> None:
        """
        Initialize tokenizer for a specific model or encoding.
        
        Args:
            model: Either a model name (e.g. "gpt-4") or encoding name (e.g. "cl100k_base")
        """
        self.model = model
        self.encoding = self._get_encoding(model)
        
    def _get_encoding(self, model: str) -> tiktoken.Encoding:
        """Get the appropriate encoding for a model.
        
        Args:
            model: Model name or encoding name
            
        Returns:
            Tiktoken encoding object
        """
        # First check our custom mapping
        if model in MODEL_ENCODING_MAP:
            encoding_name = MODEL_ENCODING_MAP[model]
            logger.debug(f"Using custom mapping: {model} -> {encoding_name}")
            return tiktoken.get_encoding(encoding_name)
        
        # Try as a known model name
        try:
            return tiktoken.encoding_for_model(model)
        except KeyError:
            pass
        
        # Try as an encoding name directly
        try:
            return tiktoken.get_encoding(model)
        except ValueError:
            pass
        
        # Default to cl100k_base for unknown models
        logger.warning(
            f"Unknown model '{model}', defaulting to {DEFAULT_ENCODING} encoding"
        )
        return tiktoken.get_encoding(DEFAULT_ENCODING)
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in a text string.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        if not text:
            return 0
        
        try:
            return len(self.encoding.encode(text))
        except Exception as e:
            logger.error(f"Error counting tokens: {e}")
            # Fallback: rough estimate based on characters
            # Average ~4 characters per token for English text
            return len(text) // 4
    
    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token limit.
        
        Args:
            text: Text to truncate
            max_tokens: Maximum number of tokens
            
        Returns:
            Truncated text
        """
        if not text or max_tokens <= 0:
            return ""
        
        tokens = self.encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text
        
        # Truncate and decode
        truncated_tokens = tokens[:max_tokens]
        return self.encoding.decode(truncated_tokens)
    
    @staticmethod
    def get_available_encodings() -> list[str]:
        """Get list of available tiktoken encodings.
        
        Returns:
            List of encoding names
        """
        return list(tiktoken.list_encoding_names())