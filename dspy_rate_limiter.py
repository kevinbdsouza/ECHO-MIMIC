"""
DSPy-compatible rate limiting wrapper for handling API rate limits.
This module provides a custom DSPy LM class that wraps the standard LM with rate limiting.
"""

import time
import logging
from typing import Any, Dict, List, Optional, Union
import dspy
from rate_limiter import RateLimiter, RateLimitError
from config import Config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RateLimitedLM(dspy.LM):
    """
    A DSPy LM wrapper that adds rate limiting to any underlying LM.
    This wraps the standard DSPy LM and applies rate limiting to all API calls.
    """
    
    def __init__(self, model: str, rate_limiter: Optional[RateLimiter] = None, **kwargs):
        """
        Initialize the rate-limited LM.
        
        Args:
            model: The model name (e.g., "gemini/gemini-2.0-flash")
            rate_limiter: Optional custom rate limiter instance
            **kwargs: Additional arguments passed to the underlying LM
        """
        # Initialize the underlying LM
        super().__init__(model=model, **kwargs)
        
        # Set up rate limiter
        if rate_limiter is None:
            cfg = Config()
            self.rate_limiter = RateLimiter(**cfg.rate_limit)
        else:
            self.rate_limiter = rate_limiter
        
        logger.info(f"Initialized RateLimitedLM for model: {model}")
    
    def __call__(self, prompt: Union[str, List[Dict[str, Any]]] = None, messages: Union[str, List[Dict[str, Any]]] = None, **kwargs) -> Any:
        """
        Make a rate-limited call to the underlying LM.
        
        Args:
            prompt: The prompt to send to the model (for direct calls)
            messages: The messages to send to the model (for DSPy adapter calls)
            **kwargs: Additional arguments for the LM call
            
        Returns:
            The response from the LM
        """
        # Handle both calling conventions: direct prompt and DSPy messages
        if messages is not None:
            # DSPy adapter calling convention
            def _make_call():
                return super(RateLimitedLM, self).__call__(messages=messages, **kwargs)
        elif prompt is not None:
            # Direct calling convention
            def _make_call():
                return super(RateLimitedLM, self).__call__(prompt, **kwargs)
        else:
            raise ValueError("Either 'prompt' or 'messages' must be provided")
        
        try:
            return self.rate_limiter.execute_with_retry(_make_call)
        except RateLimitError as e:
            logger.error(f"Rate limit exceeded for DSPy LM call: {e}")
            raise e
    
    def generate(self, prompt: Union[str, List[Dict[str, Any]]] = None, messages: Union[str, List[Dict[str, Any]]] = None, **kwargs) -> Any:
        """
        Make a rate-limited generate call to the underlying LM.
        
        Args:
            prompt: The prompt to send to the model (for direct calls)
            messages: The messages to send to the model (for DSPy adapter calls)
            **kwargs: Additional arguments for the generate call
            
        Returns:
            The response from the LM
        """
        # Handle both calling conventions: direct prompt and DSPy messages
        if messages is not None:
            # DSPy adapter calling convention
            def _make_generate():
                return super(RateLimitedLM, self).generate(messages=messages, **kwargs)
        elif prompt is not None:
            # Direct calling convention
            def _make_generate():
                return super(RateLimitedLM, self).generate(prompt, **kwargs)
        else:
            raise ValueError("Either 'prompt' or 'messages' must be provided")
        
        try:
            return self.rate_limiter.execute_with_retry(_make_generate)
        except RateLimitError as e:
            logger.error(f"Rate limit exceeded for DSPy LM generate: {e}")
            raise e


def configure_dspy_with_rate_limiting(model: str = None, rate_limiter: Optional[RateLimiter] = None, **kwargs):
    """
    Configure DSPy with rate limiting enabled.
    
    Args:
        model: The model name to use (if None, uses config default)
        rate_limiter: Optional custom rate limiter instance
        **kwargs: Additional arguments for DSPy configuration
    """
    cfg = Config()
    
    if model is None:
        model = cfg.lm
    
    # Create rate-limited LM
    lm = RateLimitedLM(model=model, rate_limiter=rate_limiter)
    
    # Configure DSPy with the rate-limited LM
    dspy.configure(lm=lm, **kwargs)
    
    logger.info(f"DSPy configured with rate limiting for model: {model}")
    return lm


class DSPyRateLimitManager:
    """
    Context manager for DSPy rate limiting that can be used to temporarily
    enable rate limiting for specific operations.
    """
    
    def __init__(self, model: str = None, rate_limiter: Optional[RateLimiter] = None, **dspy_kwargs):
        self.model = model
        self.rate_limiter = rate_limiter
        self.dspy_kwargs = dspy_kwargs
        self.original_lm = None
    
    def __enter__(self):
        # Store the original LM configuration
        self.original_lm = dspy.settings.lm if hasattr(dspy.settings, 'lm') else None
        
        # Configure with rate limiting
        lm = configure_dspy_with_rate_limiting(
            model=self.model, 
            rate_limiter=self.rate_limiter, 
            **self.dspy_kwargs
        )
        return lm
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original LM if it existed
        if self.original_lm is not None:
            dspy.configure(lm=self.original_lm)


# Convenience functions
def create_rate_limited_lm(model: str = None, **kwargs) -> RateLimitedLM:
    """
    Create a rate-limited LM instance.
    
    Args:
        model: The model name (if None, uses config default)
        **kwargs: Additional arguments for the LM
        
    Returns:
        A RateLimitedLM instance
    """
    cfg = Config()
    if model is None:
        model = cfg.lm
    
    return RateLimitedLM(model=model, **kwargs)


def patch_dspy_for_rate_limiting():
    """
    Monkey patch DSPy to use rate limiting by default.
    This replaces the default LM class with our rate-limited version.
    
    Warning: This is a global change that affects all DSPy usage in the process.
    """
    original_lm_class = dspy.LM
    
    def rate_limited_lm_factory(*args, **kwargs):
        return RateLimitedLM(*args, **kwargs)
    
    # Replace the LM class
    dspy.LM = rate_limited_lm_factory
    
    logger.info("DSPy has been patched to use rate limiting by default")
    
    return original_lm_class  # Return original in case user wants to restore


# Example usage patterns
def example_usage():
    """
    Example usage patterns for DSPy rate limiting.
    """
    
    # Method 1: Direct configuration
    configure_dspy_with_rate_limiting(model="gemini/gemini-2.0-flash", seed=42)
    
    # Method 2: Context manager
    with DSPyRateLimitManager(model="gemini/gemini-2.0-flash", seed=42) as lm:
        # Use DSPy normally within this context
        pass
    
    # Method 3: Create custom LM instance
    lm = create_rate_limited_lm(model="gemini/gemini-2.0-flash")
    dspy.configure(lm=lm, seed=42)
    
    # Method 4: Global patching (use with caution)
    # original_lm = patch_dspy_for_rate_limiting()
    # # All DSPy LM instances will now use rate limiting
    # # To restore: dspy.LM = original_lm
