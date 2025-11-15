"""
Rate limiting utility for handling API rate limits with retry logic and exponential backoff.
Specifically designed to handle Gemini API rate limits.
"""

import time
import random
import logging
from functools import wraps
from typing import Callable, Any, Optional
import google.generativeai as genai

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RateLimitError(Exception):
    """Custom exception for rate limit errors"""
    pass


class RateLimiter:
    """
    A rate limiter that handles API calls with retry logic and exponential backoff.
    Designed specifically for Gemini API rate limits.
    """
    
    def __init__(self, 
                 max_retries: int = 5,
                 base_delay: float = 1.0,
                 max_delay: float = 60.0,
                 backoff_factor: float = 2.0,
                 jitter: bool = True,
                 requests_per_minute: int = 14):  # Conservative limit for free tier
        """
        Initialize the rate limiter.
        
        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Initial delay in seconds
            max_delay: Maximum delay in seconds
            backoff_factor: Exponential backoff multiplier
            jitter: Whether to add random jitter to delays
            requests_per_minute: Rate limit for requests per minute
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.jitter = jitter
        self.requests_per_minute = requests_per_minute
        
        # Track request timestamps for rate limiting
        self.request_times = []
        
    def _is_rate_limit_error(self, error: Exception) -> bool:
        """Check if the error is a rate limit error"""
        error_str = str(error).lower()
        return any(indicator in error_str for indicator in [
            'rate limit', 'quota', 'resource_exhausted', '429', 
            'too many requests', 'exceeded your current quota'
        ])
    
    def _get_retry_delay_from_error(self, error: Exception) -> Optional[float]:
        """Extract retry delay from error message if available"""
        error_str = str(error)
        # Look for retry delay in the error message
        if 'retryDelay' in error_str:
            try:
                # Extract the delay value (e.g., "7s" -> 7.0)
                import re
                match = re.search(r'"retryDelay":\s*"(\d+)s"', error_str)
                if match:
                    return float(match.group(1))
            except:
                pass
        return None
    
    def _wait_for_rate_limit(self):
        """Wait if we're approaching the rate limit"""
        current_time = time.time()
        # Remove requests older than 1 minute
        self.request_times = [t for t in self.request_times if current_time - t < 60]
        
        if len(self.request_times) >= self.requests_per_minute:
            # Wait until the oldest request is more than 1 minute old
            wait_time = 60 - (current_time - self.request_times[0]) + 1
            if wait_time > 0:
                logger.info(f"Rate limit approaching, waiting {wait_time:.1f} seconds...")
                time.sleep(wait_time)
                # Clean up old requests again
                current_time = time.time()
                self.request_times = [t for t in self.request_times if current_time - t < 60]
    
    def _calculate_delay(self, attempt: int, suggested_delay: Optional[float] = None) -> float:
        """Calculate the delay for the next retry attempt"""
        if suggested_delay is not None:
            # Use the delay suggested by the API
            delay = suggested_delay
        else:
            # Use exponential backoff
            delay = self.base_delay * (self.backoff_factor ** attempt)
        
        # Cap the delay at max_delay
        delay = min(delay, self.max_delay)
        
        # Add jitter to avoid thundering herd
        if self.jitter:
            delay += random.uniform(0, delay * 0.1)
        
        return delay
    
    def execute_with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute a function with retry logic and rate limiting.
        
        Args:
            func: The function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            The result of the function call
            
        Raises:
            RateLimitError: If all retries are exhausted
        """
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                # Wait for rate limit before making the request
                self._wait_for_rate_limit()
                
                # Record the request time
                self.request_times.append(time.time())
                
                # Execute the function
                result = func(*args, **kwargs)
                
                # If successful, return the result
                return result
                
            except Exception as error:
                last_error = error
                
                # Check if this is a rate limit error
                if not self._is_rate_limit_error(error):
                    # If it's not a rate limit error, re-raise immediately
                    raise error
                
                # If this is the last attempt, raise the error
                if attempt == self.max_retries:
                    logger.error(f"All {self.max_retries} retry attempts exhausted")
                    raise RateLimitError(f"Rate limit exceeded after {self.max_retries} retries: {error}")
                
                # Calculate delay for next attempt
                suggested_delay = self._get_retry_delay_from_error(error)
                delay = self._calculate_delay(attempt, suggested_delay)
                
                logger.warning(f"Rate limit hit (attempt {attempt + 1}/{self.max_retries + 1}), "
                             f"retrying in {delay:.1f} seconds... Error: {error}")
                
                time.sleep(delay)
        
        # This should never be reached, but just in case
        raise RateLimitError(f"Unexpected error after retries: {last_error}")


# Global rate limiter instance
_default_rate_limiter = RateLimiter()


def with_rate_limit(rate_limiter: Optional[RateLimiter] = None):
    """
    Decorator to add rate limiting to a function.
    
    Args:
        rate_limiter: Optional custom rate limiter instance
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            limiter = rate_limiter or _default_rate_limiter
            return limiter.execute_with_retry(func, *args, **kwargs)
        return wrapper
    return decorator


def send_message_with_retry(session, message: str, rate_limiter: Optional[RateLimiter] = None) -> Any:
    """
    Send a message to a Gemini chat session with rate limiting and retry logic.
    
    Args:
        session: The Gemini chat session
        message: The message to send
        rate_limiter: Optional custom rate limiter instance
        
    Returns:
        The response from the API
    """
    limiter = rate_limiter or _default_rate_limiter
    
    def _send_message():
        return session.send_message(message)
    
    return limiter.execute_with_retry(_send_message)


def generate_content_with_retry(model, prompt: str, rate_limiter: Optional[RateLimiter] = None) -> Any:
    """
    Generate content with a Gemini model with rate limiting and retry logic.
    
    Args:
        model: The Gemini model
        prompt: The prompt to send
        rate_limiter: Optional custom rate limiter instance
        
    Returns:
        The response from the API
    """
    limiter = rate_limiter or _default_rate_limiter
    
    def _generate_content():
        return model.generate_content(prompt)
    
    return limiter.execute_with_retry(_generate_content)


# Convenience functions for common patterns
def safe_send_message(session, message: str, max_retries: int = 5) -> Any:
    """
    Safely send a message with default retry settings.
    """
    custom_limiter = RateLimiter(max_retries=max_retries)
    return send_message_with_retry(session, message, custom_limiter)


def safe_generate_content(model, prompt: str, max_retries: int = 5) -> Any:
    """
    Safely generate content with default retry settings.
    """
    custom_limiter = RateLimiter(max_retries=max_retries)
    return generate_content_with_retry(model, prompt, custom_limiter)
