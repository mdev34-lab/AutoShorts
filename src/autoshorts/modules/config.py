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

__all__ = [
    "API_URL",
    "IMG_URL",
    "API_KEY",
    "MODEL_TEXT",
    "MODEL_IMAGE",
    "VOICE",
    "TTS_RATE",
    "BG_MODE",
    "JUMPCUT_SEG_DUR",
    "BG_SKIP_INTRO",
    "TEMP_DIR_PREFIX",
    "OUTPUT_DIR",
    "IMAGE_CACHE_DIR",
    "IMAGE_PROVIDERS",
    "IMAGE_SEARCH_SAFE",
    "IMAGE_SEARCH_PER_QUERY",
    "IMAGE_MIN_WIDTH",
    "IMAGE_MIN_HEIGHT",
    "TARGET_IMAGE_WIDTH",
    "TARGET_IMAGE_HEIGHT",
    "YOUTUBE_MAX_HEIGHT",
    "YOUTUBE_FORMAT",
    "VIDEO_WIDTH",
    "VIDEO_HEIGHT",
    "VIDEO_FPS",
    "VIDEO_CODEC",
    "AUDIO_CODEC",
    "PROJECT_FONTS_DIR",
    "DEFAULT_FONT",
    "FALLBACK_FONTS",
    "CROSSFADE_TIME",
    "START_FADE",
    "VIDEO_DURATION_SECONDS",
    "SCENE_DURATION_SECONDS",
    "FONT_SIZE",
    "STROKE_WIDTH",
    "SUBTITLE_START_Y_RATIO",
    "BOTTOM_MARGIN",
    "MAX_CHARS_PER_LINE",
    "LINE_SPACING",
    "COLOR_TEXT",
    "COLOR_HIGHLIGHT",
    "COLOR_STROKE",
    "ENCODING_PRESET",
    "ENCODING_CRF",
    "ENCODING_THREADS",
    "IMAGE_OVERLAY_DURATION",
    "IMAGE_BOUNCE_INTERVAL",
    "IMAGE_FADE_IN_TIME",
    "IMAGE_FADE_OUT_TIME",
    "START_WITH_IMAGE",
    "MAX_ZOOM_FACTOR",
    "SUBTITLE_MODE",
    "API_TIMEOUT_TEXT",
    "API_TIMEOUT_IMAGE",
    "API_TIMEOUT_SEARCH",
    "FFPROBE_TIMEOUT",
    "MIN_VIDEO_DURATION",
    "MAX_VIDEO_DURATION",
    "MAX_VIDEO_CUT_DURATION",
    "BLUR_RADIUS",
    "MAX_ZOOM_VALUE",
    "OPENING_ANIMATION_ENABLED",
    "OPENING_ANIMATION_DURATION",
    "OPENING_ZOOM_START",
    "OPENING_EASING_POWER",
    "OPENING_BLUR_SAMPLES",
    "OPENING_SHUTTER_FRACTION",
]


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
VOICE = os.getenv("VOICE", "en-US-BrianMultilingualNeural")
TTS_RATE = os.getenv("TTS_RATE", "+20%")

# TTS engine: "edge" (cloud, default) or "piper" (local neural, offline)
TTS_ENGINE = os.getenv("TTS_ENGINE", "edge").strip().lower()
PIPER_VOICE_ID = os.getenv("PIPER_VOICE_ID", "pt_BR-faber-medium")
PIPER_VOICE_URL = os.getenv("PIPER_VOICE_URL", "")  # optional full .onnx URL override
PIPER_CACHE_DIR = Path(
    os.getenv("PIPER_CACHE_DIR", str(Path.home() / ".cache" / "autoshorts" / "piper"))
)
PIPER_VOICE_SHA256 = os.getenv("PIPER_VOICE_SHA256", "")  # optional integrity pin


# Background Processing Mode
BG_MODE = os.getenv("BG_MODE", "jumpcut")  # jumpcut or speed
JUMPCUT_SEG_DUR = _get_env_float("JUMPCUT_SEG_DUR", 4.0)
BG_SKIP_INTRO = _get_env_float(
    "BG_SKIP_INTRO", 25.0
)  # Skip first N seconds (intros/logos)

# Directory Configuration
TEMP_DIR_PREFIX = os.getenv("TEMP_DIR_PREFIX", "autoshorts_")
OUTPUT_DIR = _get_env_path("OUTPUT_DIR", "output")
IMAGE_CACHE_DIR = Path(tempfile.gettempdir()) / f"{TEMP_DIR_PREFIX}image_cache"

# Image Search Configuration
IMAGE_SEARCH_SAFE = os.getenv("IMAGE_SEARCH_SAFE", "on")
# Image providers in preference order. "commons" returns license-verified
# Wikimedia Commons images; "ddg" is DuckDuckGo (no license metadata -
# stock-agency/watermarked domains are blocked and contextual fit is required).
IMAGE_PROVIDERS = [
    p.strip().lower()
    for p in os.getenv("IMAGE_PROVIDERS", "commons,ddg").split(",")
    if p.strip()
]
IMAGE_SEARCH_PER_QUERY = _get_env_int("IMAGE_SEARCH_PER_QUERY", 5)
IMAGE_MIN_WIDTH = _get_env_int("IMAGE_MIN_WIDTH", 720)
IMAGE_MIN_HEIGHT = _get_env_int("IMAGE_MIN_HEIGHT", 720)
TARGET_IMAGE_WIDTH = _get_env_int("TARGET_IMAGE_WIDTH", 1080)
TARGET_IMAGE_HEIGHT = _get_env_int("TARGET_IMAGE_HEIGHT", 1920)

# Image text removal (Tesseract detection + OpenCV inpainting)
IMAGE_TEXT_REMOVAL = _get_env_bool("IMAGE_TEXT_REMOVAL", True)
TEXT_REMOVAL_MIN_CONFIDENCE = _get_env_int("TEXT_REMOVAL_MIN_CONFIDENCE", 60)
TEXT_REMOVAL_MAX_COVERAGE = _get_env_float("TEXT_REMOVAL_MAX_COVERAGE", 0.35)
TEXT_REMOVAL_INPAINT_RADIUS = _get_env_int("TEXT_REMOVAL_INPAINT_RADIUS", 4)
TEXT_REMOVAL_LANG = os.getenv("TEXT_REMOVAL_LANG", "eng")
# Subject-aware framing for downloaded images
SUBJECT_FRAMING = _get_env_bool("SUBJECT_FRAMING", True)
SUBJECT_FILL_ZOOM_THRESHOLD = _get_env_float("SUBJECT_FILL_ZOOM_THRESHOLD", 1.5)
SUBJECT_FILL_BLUR_RADIUS = _get_env_int("SUBJECT_FILL_BLUR_RADIUS", 25)

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
PROJECT_FONTS_DIR = Path(__file__).parent.parent.parent / "fonts"
DEFAULT_FONT = os.getenv(
    "DEFAULT_FONT", str(PROJECT_FONTS_DIR / "BebasNeue-Regular.ttf")
)
FALLBACK_FONTS = [
    str(PROJECT_FONTS_DIR / "BebasNeue-Regular.ttf"),
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

# Encoding Settings (quality-first defaults)
ENCODING_PRESET = os.getenv("ENCODING_PRESET", "medium")
ENCODING_CRF = _get_env_int("ENCODING_CRF", 18)
ENCODING_THREADS = _get_env_int("ENCODING_THREADS", 8)

# Experimental Mode Constants
IMAGE_OVERLAY_DURATION = _get_env_float("IMAGE_OVERLAY_DURATION", 1.0)
IMAGE_BOUNCE_INTERVAL = _get_env_float("IMAGE_BOUNCE_INTERVAL", 5.0)
IMAGE_FADE_IN_TIME = _get_env_float("IMAGE_FADE_IN_TIME", 0.2)
IMAGE_FADE_OUT_TIME = _get_env_float("IMAGE_FADE_OUT_TIME", 0.2)
START_WITH_IMAGE = _get_env_bool("START_WITH_IMAGE", True)
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

# Opening Animation (cold-open S-curve zoom-out with motion blur)
# Applied once at the start of every video to grab attention.
OPENING_ANIMATION_ENABLED = _get_env_bool("OPENING_ANIMATION_ENABLED", True)
OPENING_ANIMATION_DURATION = _get_env_float("OPENING_ANIMATION_DURATION", 0.9)
OPENING_ZOOM_START = _get_env_float("OPENING_ZOOM_START", 1.25)
OPENING_EASING_POWER = _get_env_float("OPENING_EASING_POWER", 3.0)
OPENING_BLUR_SAMPLES = _get_env_int("OPENING_BLUR_SAMPLES", 8)
OPENING_SHUTTER_FRACTION = _get_env_float("OPENING_SHUTTER_FRACTION", 0.5)
