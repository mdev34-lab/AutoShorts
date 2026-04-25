"""
Test coverage for __init__.py module exports
"""

from autoshorts import (
    Colors,
    ScriptGenerator,
    SubtitleGenerator,
    SubtitleRenderer,
    SubtitleSystem,
    TTSSystem,
    clean_temp_files,
    create_temp_dir,
    ensure_dir_exists,
    get_file_size_mb,
    get_system_font,
    get_video_duration,
    log,
    safe_filename,
    setup_directories,
)


class TestInitExports:
    """Test that all exports from __init__.py are accessible"""

    def test_subtitle_exports(self):
        """Test subtitle-related exports"""
        assert SubtitleSystem is not None
        assert SubtitleGenerator is not None
        assert SubtitleRenderer is not None

    def test_module_exports(self):
        """Test core module exports"""
        assert ScriptGenerator is not None
        assert TTSSystem is not None

    def test_logging_exports(self):
        """Test logging system exports"""
        assert log is not None
        assert Colors is not None

    def test_utility_exports(self):
        """Test utility function exports"""
        assert setup_directories is not None
        assert create_temp_dir is not None
        assert clean_temp_files is not None
        assert get_system_font is not None
        assert get_video_duration is not None
        assert safe_filename is not None
        assert ensure_dir_exists is not None
        assert get_file_size_mb is not None
