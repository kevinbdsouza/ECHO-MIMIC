"""Common building blocks shared across the Gemini experiment scripts."""

from .command import CommandOutputCapture
from .code_runner import copy_input_template, make_code_validator, pushd, validate_python_code, write_code
from .llm import GeminiModels, build_model, configure_genai, ensure_rate_limiter, init_models, is_openai_model
from .metrics import compute_radon_metrics, compute_radon_metrics_from_population
from .models import build_model_client
from .rate_limiter import (
    RateLimitError,
    RateLimiter,
    generate_content_with_retry,
    send_message_with_retry,
    with_rate_limit,
)
from .dspy_rate_limiter import (
    DSPyRateLimitManager,
    RateLimitedLM,
    configure_dspy_with_rate_limiting,
    create_rate_limited_lm,
    patch_dspy_for_rate_limiting,
)
from .text_utils import extract_message, extract_python_code, strip_code_fences

from .fix import fix_with_model

__all__ = [
    "CommandOutputCapture",
    "GeminiModels",
    "build_model",
    "configure_genai",
    "build_model_client",
    "copy_input_template",
    "ensure_rate_limiter",
    "fix_with_model",
    "extract_message",
    "extract_python_code",
    "generate_content_with_retry",
    "init_models",
    "is_openai_model",
    "make_code_validator",
    "pushd",
    "RateLimitError",
    "RateLimiter",
    "DSPyRateLimitManager",
    "RateLimitedLM",
    "configure_dspy_with_rate_limiting",
    "create_rate_limited_lm",
    "patch_dspy_for_rate_limiting",
    "compute_radon_metrics",
    "compute_radon_metrics_from_population",
    "send_message_with_retry",
    "strip_code_fences",
    "validate_python_code",
    "write_code",
    "with_rate_limit",
]
