"""
Utility functions module for AutoShorts

Common utility functions for directory management, cleanup, and system operations.
"""

import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from .config import (
    DEFAULT_FONT,
    FALLBACK_FONTS,
    FFPROBE_TIMEOUT,
    OUTPUT_DIR,
    TEMP_DIR_PREFIX,
)
from .logging_system import log


def setup_directories():
    """Create necessary directories."""
    OUTPUT_DIR.mkdir(exist_ok=True)


def create_temp_dir() -> Path:
    """Create a temporary directory with the project prefix."""
    return Path(tempfile.mkdtemp(prefix=TEMP_DIR_PREFIX))


def clean_temp_files(temp_dir: Path = None):
    """Clean up temporary files and directories."""
    target_dir = temp_dir or Path(tempfile.gettempdir()) / TEMP_DIR_PREFIX

    if target_dir.exists():
        try:
            shutil.rmtree(target_dir)
            log(f"Cleaned temporary directory: {target_dir}", "SUCCESS")
        except (OSError, PermissionError):
            pass


def get_system_font():
    """Return a safe bold font path based on OS."""
    system = platform.system()
    if system == "Windows":
        return DEFAULT_FONT
    elif system == "Darwin":  # MacOS
        return "Arial-Bold"
    else:  # Linux
        for font in FALLBACK_FONTS:
            if os.path.exists(font) or not font.startswith("/"):
                return font
        raise FileNotFoundError("No suitable font found on system")


def get_video_duration(video_path: str) -> float:
    """Get video duration using FFprobe."""
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            video_path,
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=FFPROBE_TIMEOUT
        )

        duration_str = result.stdout.strip()
        if not duration_str:
            # Try alternative method if empty
            cmd_alt = [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "csv=p=0",
                video_path,
            ]
            result_alt = subprocess.run(
                cmd_alt, capture_output=True, text=True, timeout=FFPROBE_TIMEOUT
            )
            duration_str = result_alt.stdout.strip()

        if not duration_str:
            raise ValueError("FFprobe returned empty output")

        return float(duration_str)
    except subprocess.TimeoutExpired:
        log("FFprobe timeout", "ERROR")
        raise
    except Exception as e:
        log(f"FFprobe error: {e}", "ERROR")
        raise


def safe_filename(filename: str) -> str:
    """Create a safe filename by removing invalid characters."""
    import re

    # Remove invalid characters
    safe_name = re.sub(r'[<>:"/\\|?*]', "", filename)
    # Replace spaces with underscores
    safe_name = safe_name.replace(" ", "_")
    # Limit length
    return safe_name[:50] if len(safe_name) > 50 else safe_name


def ensure_dir_exists(path: Path):
    """Ensure directory exists, create if necessary."""
    path.mkdir(parents=True, exist_ok=True)


def get_file_size_mb(file_path: str) -> float:
    """Get file size in MB."""
    try:
        size_bytes = os.path.getsize(file_path)
        return size_bytes / (1024 * 1024)
    except OSError:
        return 0.0


def shutdown_computer():
    """Shutdown the computer after processing is complete."""
    import os
    import platform

    try:
        system = platform.system()
        if system == "Windows":
            os.system("shutdown /s /t 30")
            log("Computer will shutdown in 30 seconds...")
        elif system == "Linux" or system == "Darwin":
            os.system("shutdown -h +1")
            log("Computer will shutdown in 1 minute...")
        else:
            log("Unsupported OS for auto-shutdown", "WARNING")
    except Exception as e:
        log(f"Failed to shutdown computer: {e}", "ERROR")
