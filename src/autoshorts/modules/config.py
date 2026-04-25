"""
Configuration module for AutoShorts

Centralizes all API keys, URLs, model settings, and constants.
Loads configuration from .env file with proper fallbacks.
"""

import os
import tempfile
from pathlib import Path

try:
    from dotenv import load_dotenv

    # Try to load .env from project root (4 levels up from this file)
    project_root = Path(__file__).parent.parent.parent.parent
    env_file = project_root / ".env"
    if env_file.exists():
        load_dotenv(env_file)
    else:
        # Fallback to current working directory
        load_dotenv()
except ImportError:
    # python-dotenv not available, continue with environment variables only
    pass


def _get_env_bool(key: str, default: bool = False) -> bool:
    """Get boolean value from environment variable."""
    value = os.getenv(key, str(default)).lower()
    return value in ("true", "1", "yes", "on")


def _get_env_int(key: str, default: int) -> int:
    """Get integer value from environment variable."""
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default


def _get_env_float(key: str, default: float) -> float:
    """Get float value from environment variable."""
    try:
        return float(os.getenv(key, str(default)))
    except ValueError:
        return default


def _get_env_path(key: str, default: str) -> Path:
    """Get Path object from environment variable."""
    return Path(os.getenv(key, default))


# API Configuration
API_URL = os.getenv("API_URL", "https://gen.pollinations.ai/v1/chat/completions")
IMG_URL = os.getenv("IMG_URL", "https://gen.pollinations.ai/image/")
API_KEY = os.getenv("API_KEY", "")

# Model Configuration
MODEL_TEXT = os.getenv("MODEL_TEXT", "nova-micro")
MODEL_IMAGE = os.getenv("MODEL_IMAGE", "flux")

# TTS Configuration
VOICE = os.getenv("VOICE", "de-DE-FlorianMultilingualNeural")
TTS_RATE = os.getenv("TTS_RATE", "+20%")

# Directory Configuration
TEMP_DIR_PREFIX = os.getenv("TEMP_DIR_PREFIX", "autoshorts_")
OUTPUT_DIR = _get_env_path("OUTPUT_DIR", "output")
IMAGE_CACHE_DIR = Path(tempfile.gettempdir()) / f"{TEMP_DIR_PREFIX}image_cache"
TEMP_DIR_PREFIX = os.getenv("TEMP_DIR_PREFIX", "autoshorts_")

# YouTube Configuration
YOUTUBE_MAX_HEIGHT = _get_env_int("YOUTUBE_MAX_HEIGHT", 360)
YOUTUBE_FORMAT = os.getenv(
    "YOUTUBE_FORMAT",
    "bestvideo[height<={max_height}][ext=mp4]/bestvideo[height<={max_height}]/bestvideo[ext=mp4]/bestvideo",
)

# Video Configuration (FPS doesn't affect quality - use resolution for quality)
VIDEO_WIDTH = _get_env_int("VIDEO_WIDTH", 1080)
VIDEO_HEIGHT = _get_env_int("VIDEO_HEIGHT", 1920)
VIDEO_FPS = _get_env_int("VIDEO_FPS", 24)
VIDEO_CODEC = os.getenv("VIDEO_CODEC", "libx264")
AUDIO_CODEC = os.getenv("AUDIO_CODEC", "aac")

# Font Configuration
DEFAULT_FONT = os.getenv("DEFAULT_FONT", "arialbd.ttf")
FALLBACK_FONTS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    "DejaVu-Sans-Bold",
]

# Video Timing Constants
CROSSFADE_TIME = _get_env_float("CROSSFADE_TIME", 0.5)
START_FADE = _get_env_float("START_FADE", 1.0)

# Fixed Video Timing (Flux Images)
VIDEO_DURATION_SECONDS = _get_env_float("VIDEO_DURATION_SECONDS", 30.0)
SCENE_DURATION_SECONDS = _get_env_float("SCENE_DURATION_SECONDS", 3.0)

# Subtitle Configuration
FONT_SIZE = _get_env_int("SUBTITLE_FONT_SIZE", 70)
STROKE_WIDTH = _get_env_int("STROKE_WIDTH", 5)
SUBTITLE_START_Y_RATIO = _get_env_float("SUBTITLE_START_Y_RATIO", 0.65)
BOTTOM_MARGIN = _get_env_int("BOTTOM_MARGIN", 150)
MAX_CHARS_PER_LINE = _get_env_int("MAX_CHARS_PER_LINE", 20)
LINE_SPACING = _get_env_int("LINE_SPACING", 35)

# Colors
COLOR_TEXT = os.getenv("COLOR_TEXT", "#FFFFFF")
COLOR_HIGHLIGHT = os.getenv("COLOR_HIGHLIGHT", "#FFFF00")
COLOR_STROKE = os.getenv("COLOR_STROKE", "#000000")

# Encoding Settings (use more threads for faster rendering)
ENCODING_PRESET = os.getenv("ENCODING_PRESET", "fast")
ENCODING_CRF = _get_env_int("ENCODING_CRF", 23)
ENCODING_THREADS = _get_env_int("ENCODING_THREADS", 8)

# Experimental Mode Constants
IMAGE_OVERLAY_DURATION = _get_env_float("IMAGE_OVERLAY_DURATION", 1.0)
IMAGE_BOUNCE_INTERVAL = _get_env_float("IMAGE_BOUNCE_INTERVAL", 5.0)
IMAGE_FADE_IN_TIME = _get_env_float("IMAGE_FADE_IN_TIME", 0.2)
IMAGE_FADE_OUT_TIME = _get_env_float("IMAGE_FADE_OUT_TIME", 0.2)
MAX_ZOOM_FACTOR = _get_env_float("MAX_ZOOM_FACTOR", 1.05)

# Subtitle Mode (simple = faster, no word highlighting)
SUBTITLE_MODE = os.getenv("SUBTITLE_MODE", "simple")

# API Timeouts (seconds)
API_TIMEOUT_TEXT = _get_env_int("API_TIMEOUT_TEXT", 30)
API_TIMEOUT_IMAGE = _get_env_int("API_TIMEOUT_IMAGE", 60)
API_TIMEOUT_SEARCH = _get_env_int("API_TIMEOUT_SEARCH", 15)
FFPROBE_TIMEOUT = _get_env_int("FFPROBE_TIMEOUT", 30)

# Video Processing
MIN_VIDEO_DURATION = _get_env_int("MIN_VIDEO_DURATION", 60)
MAX_VIDEO_DURATION = _get_env_int("MAX_VIDEO_DURATION", 3600)
MAX_VIDEO_CUT_DURATION = _get_env_int("MAX_VIDEO_CUT_DURATION", 300)
BLUR_RADIUS = _get_env_int("BLUR_RADIUS", 10)

# Image Zoom Settings
MAX_ZOOM_VALUE = _get_env_float("MAX_ZOOM_VALUE", 1.15)
