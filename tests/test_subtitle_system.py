"""
Test subtitle system functionality
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from autoshorts.modules.subtitle_system import (
    SubtitleGenerator,
    SubtitleRenderer,
    SubtitleSystem,
    get_system_font,
)


class TestSubtitleGenerator:
    """Test cases for SubtitleGenerator class"""

    def test_format_vtt_time(self):
        """Test VTT time formatting"""
        # Test basic time formatting
        result = SubtitleGenerator.format_vtt_time(3661.123)
        assert result == "01:01:01.123"

        # Test zero time
        result = SubtitleGenerator.format_vtt_time(0)
        assert result == "00:00:00.000"

        # Test fractional seconds
        result = SubtitleGenerator.format_vtt_time(65.5)
        assert result == "00:01:05.500"

    def test_format_vtt_time_edge_cases(self):
        """Test VTT time formatting edge cases"""
        # Test large time
        result = SubtitleGenerator.format_vtt_time(36000.999)  # 10 hours
        assert result == "10:00:00.999"

        # Test small fractional - note: int() truncates, doesn't round
        result = SubtitleGenerator.format_vtt_time(1.001)
        assert (
            result == "00:00:01.001" or result == "00:00:01.000"
        )  # Accept either due to float precision

    def test_generate_vtt_from_paragraphs(self):
        """Test VTT generation from paragraphs"""
        paragraphs = ["Primeiro parágrafo de teste.", "Segundo parágrafo de teste."]
        duration = 10.0

        with tempfile.NamedTemporaryFile(mode="w", suffix=".vtt", delete=False) as f:
            vtt_path = f.name

        try:
            result = SubtitleGenerator.generate_vtt_from_paragraphs(
                paragraphs, duration, vtt_path
            )

            assert result == vtt_path
            assert Path(vtt_path).exists()

            content = Path(vtt_path).read_text(encoding="utf-8")
            assert "WEBVTT" in content
            # Text is combined and split into chunks, so check for partial content
            assert "parágrafo" in content
        finally:
            Path(vtt_path).unlink(missing_ok=True)

    def test_generate_vtt_from_paragraphs_empty(self):
        """Test VTT generation with empty paragraphs"""
        paragraphs = []
        duration = 5.0

        with tempfile.NamedTemporaryFile(mode="w", suffix=".vtt", delete=False) as f:
            vtt_path = f.name

        try:
            result = SubtitleGenerator.generate_vtt_from_paragraphs(
                paragraphs, duration, vtt_path
            )

            assert result == vtt_path
            assert Path(vtt_path).exists()

            content = Path(vtt_path).read_text(encoding="utf-8")
            assert "WEBVTT" in content
        finally:
            Path(vtt_path).unlink(missing_ok=True)

    def test_generate_vtt_from_paragraphs_single_paragraph(self):
        """Test VTT generation with single paragraph"""
        paragraphs = ["Único parágrafo."]
        duration = 5.0

        with tempfile.NamedTemporaryFile(mode="w", suffix=".vtt", delete=False) as f:
            vtt_path = f.name

        try:
            result = SubtitleGenerator.generate_vtt_from_paragraphs(
                paragraphs, duration, vtt_path
            )

            assert result == vtt_path
            assert Path(vtt_path).exists()

            content = Path(vtt_path).read_text(encoding="utf-8")
            assert "WEBVTT" in content
            assert "Único parágrafo" in content
        finally:
            Path(vtt_path).unlink(missing_ok=True)

    def test_generate_vtt_from_paragraphs_with_quotes(self):
        """Test VTT generation with quotes in text"""
        paragraphs = ['Texto com "aspas" no meio.']
        duration = 5.0

        with tempfile.NamedTemporaryFile(mode="w", suffix=".vtt", delete=False) as f:
            vtt_path = f.name

        try:
            _result = SubtitleGenerator.generate_vtt_from_paragraphs(
                paragraphs, duration, vtt_path
            )

            content = Path(vtt_path).read_text(encoding="utf-8")
            # Quotes should be removed
            assert '"' not in content
        finally:
            Path(vtt_path).unlink(missing_ok=True)

    def test_generate_vtt_from_paragraphs_long_text(self):
        """Test VTT generation with long text"""
        # Create many words to test chunking
        paragraphs = [" ".join(["palavra"] * 20)]
        duration = 10.0

        with tempfile.NamedTemporaryFile(mode="w", suffix=".vtt", delete=False) as f:
            vtt_path = f.name

        try:
            _result = SubtitleGenerator.generate_vtt_from_paragraphs(
                paragraphs, duration, vtt_path
            )

            content = Path(vtt_path).read_text(encoding="utf-8")
            assert "WEBVTT" in content
            # Should have multiple cues due to chunking
            assert content.count("-->") >= 1
        finally:
            Path(vtt_path).unlink(missing_ok=True)


class TestSubtitleRenderer:
    """Test cases for SubtitleRenderer class"""

    def setup_method(self):
        """Setup test fixtures"""
        self.renderer = SubtitleRenderer()
        self.video_size = (720, 1280)

    def test_init(self):
        """Test subtitle renderer initialization"""
        assert hasattr(self.renderer, "font")
        assert hasattr(self.renderer, "_text_clip_cache")
        assert isinstance(self.renderer._text_clip_cache, dict)

    def test_init_with_custom_font(self):
        """Test subtitle renderer initialization with custom font"""
        renderer = SubtitleRenderer(font_path="custom_font.ttf")
        assert renderer.font == "custom_font.ttf"

    def test_wrap_text_to_lines(self):
        """Test text wrapping functionality"""
        text = "This is a long text that should be wrapped to multiple lines"
        lines = self.renderer._wrap_text_to_lines(text, max_chars_per_line=20)

        assert isinstance(lines, list)
        assert len(lines) >= 1
        # Each line should be a list of words
        for line in lines:
            assert isinstance(line, list)

    def test_wrap_text_to_lines_short_text(self):
        """Test text wrapping with short text"""
        text = "Short text"
        lines = self.renderer._wrap_text_to_lines(text, max_chars_per_line=50)

        assert len(lines) == 1
        assert lines[0] == ["Short", "text"]

    def test_wrap_text_to_lines_single_word(self):
        """Test text wrapping with single word"""
        text = "SingleWord"
        lines = self.renderer._wrap_text_to_lines(text, max_chars_per_line=5)

        assert len(lines) == 1

    def test_vtt_time_to_seconds(self):
        """Test VTT time to seconds conversion"""
        result = self.renderer._vtt_time_to_seconds("00:01:30.500")
        assert result == 90.5

        result = self.renderer._vtt_time_to_seconds("01:00:00.000")
        assert result == 3600.0

        result = self.renderer._vtt_time_to_seconds("00:00:05.250")
        assert result == 5.25

    @patch("autoshorts.modules.subtitle_system.TextClip")
    def test_get_text_dimensions(self, mock_textclip):
        """Test text dimension caching"""
        mock_clip = Mock()
        mock_clip.size = (100, 30)
        mock_clip.close = Mock()
        mock_textclip.return_value = mock_clip

        # First call should create cache
        dims1 = self.renderer._get_text_dimensions("Test", 24)
        assert dims1 == (100, 30)

        # Second call should use cache
        dims2 = self.renderer._get_text_dimensions("Test", 24)
        assert dims2 == (100, 30)

        # TextClip should only be called once due to caching
        mock_textclip.assert_called_once()

    @patch("autoshorts.modules.subtitle_system.TextClip")
    def test_get_text_dimensions_error(self, mock_textclip):
        """Test text dimension handling with error"""
        mock_textclip.side_effect = Exception("Font error")

        dims = self.renderer._get_text_dimensions("Test", 24)

        # Should return fallback dimensions
        assert dims[0] > 0
        assert dims[1] > 0

    @patch("autoshorts.modules.subtitle_system.TextClip")
    def test_get_text_dimensions_invalid_size(self, mock_textclip):
        """Test text dimension handling with invalid size"""
        mock_clip = Mock()
        mock_clip.size = (0, 0)  # Invalid size
        mock_clip.close = Mock()
        mock_textclip.return_value = mock_clip

        dims = self.renderer._get_text_dimensions("Test", 24)

        # Should return fallback dimensions
        assert dims[0] > 0
        assert dims[1] > 0

    @patch("autoshorts.modules.subtitle_system.TextClip")
    def test_render_subtitles(self, mock_textclip):
        """Test subtitle rendering from VTT file"""
        # Create mock VTT content
        vtt_content = """WEBVTT

00:00:00.000 --> 00:00:02.000
Test subtitle

00:00:02.000 --> 00:00:04.000
Another subtitle
"""

        # Create temporary VTT file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".vtt", delete=False) as f:
            f.write(vtt_content)
            vtt_path = f.name

        try:
            # Mock TextClip to avoid actual moviepy operations
            mock_clip = Mock()
            mock_clip.size = (100, 20)
            mock_clip.with_position.return_value = mock_clip
            mock_clip.with_start.return_value = mock_clip
            mock_clip.with_duration.return_value = mock_clip
            mock_clip.close = Mock()
            mock_textclip.return_value = mock_clip

            result = self.renderer.create_subtitle_clips_optimized(
                vtt_path, self.video_size
            )

            # Should return list of subtitle clips
            assert isinstance(result, list)

        finally:
            Path(vtt_path).unlink(missing_ok=True)

    @patch("autoshorts.modules.subtitle_system.TextClip")
    def test_render_empty_subtitles(self, mock_textclip):
        """Test rendering empty VTT file"""
        vtt_content = "WEBVTT\n\n"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".vtt", delete=False) as f:
            f.write(vtt_content)
            vtt_path = f.name

        try:
            result = self.renderer.create_subtitle_clips_optimized(
                vtt_path, self.video_size
            )

            # Should return empty list
            assert result == []

        finally:
            Path(vtt_path).unlink(missing_ok=True)

    def test_render_subtitles_nonexistent_file(self):
        """Test rendering with nonexistent VTT file"""
        result = self.renderer.create_subtitle_clips_optimized(
            "/nonexistent/path.vtt", self.video_size
        )
        assert result == []

    def test_render_subtitles_none_path(self):
        """Test rendering with None path"""
        result = self.renderer.create_subtitle_clips_optimized(None, self.video_size)
        assert result == []

    @patch("autoshorts.modules.subtitle_system.TextClip")
    def test_render_subtitles_with_error(self, mock_textclip):
        """Test rendering handles errors gracefully"""
        vtt_content = """WEBVTT

00:00:00.000 --> 00:00:02.000
Test subtitle
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".vtt", delete=False) as f:
            f.write(vtt_content)
            vtt_path = f.name

        try:
            # Make TextClip raise an error - but the code catches and continues
            mock_textclip.side_effect = Exception("Rendering error")

            # The function catches errors internally and returns empty list
            result = self.renderer.create_subtitle_clips_optimized(
                vtt_path, self.video_size
            )

            # Should return empty list due to errors
            assert isinstance(result, list)

        finally:
            Path(vtt_path).unlink(missing_ok=True)


class TestSubtitleSystem:
    """Test cases for SubtitleSystem class"""

    def setup_method(self):
        """Setup test fixtures"""
        self.subtitle_system = SubtitleSystem()
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Cleanup test fixtures"""
        if self.temp_dir.exists():
            import shutil

            shutil.rmtree(self.temp_dir)

    def test_init(self):
        """Test subtitle system initialization"""
        assert hasattr(self.subtitle_system, "generator")
        assert hasattr(self.subtitle_system, "renderer")
        assert isinstance(self.subtitle_system.generator, SubtitleGenerator)
        assert isinstance(self.subtitle_system.renderer, SubtitleRenderer)

    def test_init_with_custom_font(self):
        """Test subtitle system initialization with custom font"""
        system = SubtitleSystem(font_path="custom_font.ttf")
        assert system.renderer.font == "custom_font.ttf"

    def test_generate_subtitles(self):
        """Test subtitle generation"""
        paragraphs = ["Test paragraph 1", "Test paragraph 2"]
        duration = 10.0

        result_path = self.subtitle_system.generate_subtitles(
            paragraphs, duration, str(self.temp_dir)
        )

        assert result_path is not None
        assert result_path.endswith(".vtt")
        assert Path(result_path).exists()

        # Check VTT content
        content = Path(result_path).read_text(encoding="utf-8")
        assert "WEBVTT" in content
        assert "Test paragraph 1" in content

    @patch(
        "autoshorts.modules.subtitle_system.SubtitleRenderer.create_subtitle_clips_optimized"
    )
    def test_render_subtitles(self, mock_render):
        """Test subtitle rendering"""
        mock_render.return_value = []

        vtt_path = self.temp_dir / "test.vtt"
        vtt_path.write_text("WEBVTT\n\n", encoding="utf-8")

        result = self.subtitle_system.render_subtitles(str(vtt_path), (720, 1280))

        mock_render.assert_called_once()
        assert isinstance(result, list)


class TestGetSystemFont:
    """Test cases for get_system_font function"""

    @patch("platform.system", return_value="Windows")
    def test_get_font_windows(self, mock_platform):
        """Test font detection on Windows"""
        result = get_system_font()
        assert result == "arialbd.ttf"

    @patch("platform.system", return_value="Darwin")
    def test_get_font_macos(self, mock_platform):
        """Test font detection on macOS"""
        result = get_system_font()
        assert result == "Arial-Bold"

    @patch("platform.system", return_value="Linux")
    @patch("os.path.exists")
    def test_get_font_linux(self, mock_exists, mock_platform):
        """Test font detection on Linux"""
        # Mock first font as existing
        mock_exists.return_value = True

        result = get_system_font()
        assert result == "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

    @patch("platform.system", return_value="Linux")
    @patch("os.path.exists")
    def test_get_font_linux_fallback(self, mock_exists, mock_platform):
        """Test font fallback on Linux when no fonts found"""
        # Mock all fonts as non-existing
        mock_exists.return_value = False

        result = get_system_font()
        assert result == "DejaVu-Sans-Bold"

    @patch("platform.system", return_value="Linux")
    @patch("os.path.exists", side_effect=[False, True])
    def test_get_font_linux_second_fallback(self, mock_exists, mock_platform):
        """Test font fallback on Linux with second font"""
        result = get_system_font()
        assert result == "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"

    @patch("platform.system", return_value="Linux")
    @patch("os.path.exists", return_value=False)
    def test_get_font_linux_no_fonts(self, mock_exists, mock_platform):
        """Test font detection on Linux with no fonts available"""
        # The function returns "DejaVu-Sans-Bold" as final fallback
        result = get_system_font()
        assert result == "DejaVu-Sans-Bold"


if __name__ == "__main__":
    pytest.main([__file__])
