"""
Logging system module for AutoShorts

Provides colored logging functionality for better console output.
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import time

from .config import *


class Colors:
    """ANSI color codes for terminal output."""

    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"


def log(message: str, level: str = "INFO"):
    """Print colored log message."""
    timestamp = time.strftime("%H:%M:%S")
    level_colors = {
        "INFO": Colors.CYAN,
        "SUCCESS": Colors.GREEN,
        "WARNING": Colors.WARNING,
        "ERROR": Colors.FAIL,
    }
    color = level_colors.get(level, Colors.ENDC)
    print(f"{color}[{timestamp}] [{level}] {message}{Colors.ENDC}")
