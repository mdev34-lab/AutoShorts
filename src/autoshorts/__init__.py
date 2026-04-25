"""
AutoShorts - Automated Viral Video Generator

A Python package for generating automated short-form video content
with AI-powered scripts, text-to-speech, and subtitle generation.
"""

__version__ = "1.0.0"
__author__ = "AutoShorts Team"

from .modules.logging_system import Colors, log
from .modules.script_generator import ScriptGenerator
from .modules.subtitle_system import SubtitleGenerator, SubtitleRenderer, SubtitleSystem
from .modules.tts_system import TTSSystem
from .modules.utils import (
    clean_temp_files,
    create_temp_dir,
    ensure_dir_exists,
    get_file_size_mb,
    get_system_font,
    get_video_duration,
    safe_filename,
    setup_directories,
)

__all__ = [
    "SubtitleSystem",
    "SubtitleGenerator",
    "SubtitleRenderer",
    "log",
    "Colors",
    "ScriptGenerator",
    "TTSSystem",
    "setup_directories",
    "create_temp_dir",
    "clean_temp_files",
    "get_system_font",
    "get_video_duration",
    "safe_filename",
    "ensure_dir_exists",
    "get_file_size_mb",
]
