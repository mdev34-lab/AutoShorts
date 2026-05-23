"""
Test utility functions
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from autoshorts.modules.utils import (
    clean_temp_files,
    create_temp_dir,
    ensure_dir_exists,
    get_file_size_mb,
    get_system_font,
    get_video_duration,
    safe_filename,
    setup_directories,
)


class TestDirectoryFunctions:
    """Test cases for directory management functions"""

    def setup_method(self):
        """Setup test fixtures"""
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Cleanup test fixtures"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_setup_directories(self):
        """Test directory creation"""
        with patch("autoshorts.modules.utils.OUTPUT_DIR", self.temp_dir / "output"):
            setup_directories()
            assert (self.temp_dir / "output").exists()

    def test_create_temp_dir(self):
        """Test temporary directory creation"""
        temp_path = create_temp_dir()

        assert temp_path.exists()
        assert temp_path.is_dir()
        assert temp_path.name.startswith("autoshorts_")

        # Cleanup
        shutil.rmtree(temp_path)

    def test_clean_temp_files_with_path(self):
        """Test temporary file cleanup with specific path"""
        # Create a test temp directory
        test_temp = self.temp_dir / "test_cleanup"
        test_temp.mkdir()
        test_file = test_temp / "test.txt"
        test_file.write_text("test")

        clean_temp_files(test_temp)

        assert not test_temp.exists()

    def test_clean_temp_files_nonexistent(self):
        """Test cleanup of non-existent directory"""
        # Should not raise an exception
        clean_temp_files(Path("/nonexistent/path"))

    def test_clean_temp_files_error(self):
        """Test cleanup handles errors gracefully"""
        with patch("shutil.rmtree", side_effect=PermissionError("Access denied")):
            # Should not raise exception
            clean_temp_files(self.temp_dir)


class TestFileFunctions:
    """Test cases for file utility functions"""

    def setup_method(self):
        """Setup test fixtures"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.test_file = self.temp_dir / "test_file.txt"
        self.test_file.write_text("test content")

    def teardown_method(self):
        """Cleanup test fixtures"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_safe_filename(self):
        """Test filename sanitization"""
        # Test with special characters
        unsafe_name = "file/name?with*special:chars"
        safe_name = safe_filename(unsafe_name)

        assert "/" not in safe_name
        assert "?" not in safe_name
        assert "*" not in safe_name
        assert ":" not in safe_name

    def test_safe_filename_with_backslash(self):
        """Test filename sanitization with backslash"""
        unsafe_name = "file\\name\\with\\backslashes"
        safe_name = safe_filename(unsafe_name)

        assert "\\" not in safe_name

    def test_safe_filename_with_quotes(self):
        """Test filename sanitization with quotes"""
        unsafe_name = 'file"name"with<quotes>'
        safe_name = safe_filename(unsafe_name)

        assert '"' not in safe_name
        assert "<" not in safe_name
        assert ">" not in safe_name

    def test_safe_filename_with_pipe(self):
        """Test filename sanitization with pipe character"""
        unsafe_name = "file|name"
        safe_name = safe_filename(unsafe_name)

        assert "|" not in safe_name

    def test_safe_filename_spaces(self):
        """Test filename sanitization replaces spaces"""
        name_with_spaces = "file name with spaces"
        safe_name = safe_filename(name_with_spaces)

        assert " " not in safe_name
        assert "_" in safe_name

    def test_safe_filename_long_name(self):
        """Test filename sanitization truncates long names"""
        long_name = "a" * 100
        safe_name = safe_filename(long_name)

        assert len(safe_name) <= 50

    def test_safe_filename_normal(self):
        """Test filename sanitization with normal name"""
        normal_name = "normal_filename"
        safe_name = safe_filename(normal_name)

        assert safe_name == normal_name

    def test_ensure_dir_exists(self):
        """Test directory existence assurance"""
        new_dir = self.temp_dir / "new_directory"

        ensure_dir_exists(new_dir)
        assert new_dir.exists()
        assert new_dir.is_dir()

        # Should not raise error if dir already exists
        ensure_dir_exists(new_dir)

    def test_ensure_dir_exists_nested(self):
        """Test directory creation with nested paths"""
        nested_dir = self.temp_dir / "level1" / "level2" / "level3"

        ensure_dir_exists(nested_dir)
        assert nested_dir.exists()

    def test_get_file_size_mb(self):
        """Test file size calculation"""
        # Create a file with known content
        test_content = "x" * 1024 * 1024  # 1MB
        large_file = self.temp_dir / "large_file.txt"
        large_file.write_text(test_content)

        size_mb = get_file_size_mb(large_file)

        # Should be approximately 1MB (allowing for some rounding)
        assert 0.9 <= size_mb <= 1.1

    def test_get_file_size_mb_nonexistent(self):
        """Test file size calculation for non-existent file"""
        nonexistent_file = self.temp_dir / "nonexistent.txt"

        size_mb = get_file_size_mb(nonexistent_file)
        assert size_mb == 0.0

    def test_get_file_size_mb_small_file(self):
        """Test file size calculation for small file"""
        small_file = self.temp_dir / "small.txt"
        small_file.write_text("test")

        size_mb = get_file_size_mb(small_file)

        assert size_mb > 0
        assert size_mb < 0.001  # Very small


class TestSystemFunctions:
    """Test cases for system utility functions"""

    @patch("platform.system", return_value="Windows")
    def test_get_system_font_windows(self, mock_platform):
        """Test font detection on Windows"""
        font = get_system_font()
        # Windows returns DEFAULT_FONT from config which is "Arial"
        assert font is not None

    @patch("platform.system", return_value="Darwin")
    def test_get_system_font_macos(self, mock_platform):
        """Test font detection on macOS"""
        font = get_system_font()
        assert font == "Arial-Bold"

    @patch("platform.system", return_value="Linux")
    def test_get_system_font_linux(self, mock_platform):
        """Test font detection on Linux — returns first non-/-starting fallback"""
        font = get_system_font()
        assert isinstance(font, str)
        assert len(font) > 0

    @patch("platform.system", return_value="Linux")
    def test_get_system_font_linux_fallback(self, mock_platform):
        """Test font fallback on Linux — handles empty font list"""
        with patch("autoshorts.modules.utils.FALLBACK_FONTS", []):
            with pytest.raises(FileNotFoundError):
                get_system_font()

    @patch("platform.system", return_value="Linux")
    @patch(
        "autoshorts.modules.utils.FALLBACK_FONTS",
        [
            "/nonexistent/font.ttf",
            "/usr/share/fonts/liberation/LiberationSans-Bold.ttf",
        ],
    )
    @patch(
        "os.path.exists",
        side_effect=lambda p: (
            p == "/usr/share/fonts/liberation/LiberationSans-Bold.ttf"
        ),
    )
    def test_get_system_font_linux_second_fallback(self, mock_exists, mock_platform):
        """Test font fallback on Linux with second font"""
        font = get_system_font()
        assert "LiberationSans-Bold" in font

    @patch("subprocess.run")
    def test_get_video_duration(self, mock_run):
        """Test video duration extraction"""
        # Mock ffprobe output
        mock_result = Mock()
        mock_result.stdout = "123.45\n"
        mock_result.stderr = ""
        mock_result.returncode = 0

        mock_run.return_value = mock_result

        result = get_video_duration("test.mp4")

        assert isinstance(result, float)
        assert result == 123.45

    @patch("subprocess.run")
    def test_get_video_duration_with_alternative_method(self, mock_run):
        """Test video duration with alternative ffprobe format"""
        # First call returns empty, second returns value
        first_result = Mock()
        first_result.stdout = ""
        first_result.stderr = ""

        second_result = Mock()
        second_result.stdout = "456.78\n"
        second_result.stderr = ""

        mock_run.side_effect = [first_result, second_result]

        result = get_video_duration("test.mp4")

        assert result == 456.78

    @patch("subprocess.run")
    def test_get_video_duration_empty_output(self, mock_run):
        """Test video duration with empty output raises error"""
        mock_result = Mock()
        mock_result.stdout = ""
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        with pytest.raises(ValueError):
            get_video_duration("test.mp4")

    @patch(
        "subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd="ffprobe", timeout=30),
    )
    def test_get_video_duration_timeout(self, mock_run):
        """Test video duration with timeout"""
        with pytest.raises(subprocess.TimeoutExpired):
            get_video_duration("test.mp4")

    @patch("subprocess.run", side_effect=Exception("FFprobe error"))
    def test_get_video_duration_error(self, mock_run):
        """Test video duration with general error"""
        with pytest.raises(Exception):
            get_video_duration("test.mp4")


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_safe_filename_empty(self):
        """Test safe filename with empty string"""
        result = safe_filename("")
        assert result == ""

    def test_safe_filename_exactly_50_chars(self):
        """Test safe filename exactly at limit"""
        name = "a" * 50
        result = safe_filename(name)
        assert len(result) == 50

    def test_safe_filename_over_50_chars(self):
        """Test safe filename over limit gets truncated"""
        name = "a" * 60
        result = safe_filename(name)
        assert len(result) == 50

    def test_get_file_size_mb_error(self):
        """Test file size calculation handles OSError"""
        with patch("os.path.getsize", side_effect=OSError("File error")):
            result = get_file_size_mb("nonexistent.txt")
            assert result == 0.0


if __name__ == "__main__":
    pytest.main([__file__])
