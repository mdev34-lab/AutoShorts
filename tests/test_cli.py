"""
Tests for the CLI layer (Typer app, commands, help resolution).
"""

from unittest.mock import Mock, patch

import typer
from typer.testing import CliRunner

from autoshorts.cli import app
from autoshorts.cli.commands.help_cmd import make_help_command
from autoshorts.main import main


class TestAppStructure:
    """Test the Typer app structure."""

    def test_app_is_typer(self):
        assert isinstance(app, typer.Typer)

    def test_app_has_new_command(self):
        assert any(g.name == "new" for g in app.registered_groups)

    def test_app_has_help_command(self):
        commands = {c.name for c in app.registered_commands}
        assert "help" in commands

    def test_new_app_has_explainer(self):
        new_group = app.registered_groups[0]
        assert new_group.name == "new"
        typer_new = new_group.typer_instance
        explainer_names = {c.name for c in typer_new.registered_commands}
        assert "explainer" in explainer_names


class TestHelpCommand:
    """Test the custom help command."""

    def setup_method(self):
        self.runner = CliRunner()

    def test_make_help_command_returns_function(self):
        cmd = make_help_command(app)
        assert callable(cmd)

    def test_custom_help_root(self):
        result = self.runner.invoke(app, ["help"])
        assert result.exit_code == 0

    def test_custom_help_new(self):
        result = self.runner.invoke(app, ["help", "new"])
        assert result.exit_code == 0
        assert "explainer" in result.output

    def test_custom_help_explainer(self):
        result = self.runner.invoke(app, ["help", "new", "explainer"])
        assert result.exit_code == 0
        assert "subject" in result.output

    def test_custom_help_invalid(self):
        result = self.runner.invoke(app, ["help", "nonexistent"])
        assert result.exit_code != 0


class TestCliRunner:
    """Integration-style tests using CliRunner."""

    def setup_method(self):
        self.runner = CliRunner()

    def test_help_flag(self):
        result = self.runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "autoshorts" in result.output

    def test_new_help(self):
        result = self.runner.invoke(app, ["new", "--help"])
        assert result.exit_code == 0
        assert "explainer" in result.output

    def test_explainer_help(self):
        result = self.runner.invoke(app, ["new", "explainer", "--help"])
        assert result.exit_code == 0
        assert "subject" in result.output

    def test_custom_help_command(self):
        result = self.runner.invoke(app, ["help"])
        assert result.exit_code == 0

    def test_custom_help_new(self):
        result = self.runner.invoke(app, ["help", "new"])
        assert result.exit_code == 0
        assert "explainer" in result.output

    def test_custom_help_invalid(self):
        result = self.runner.invoke(app, ["help", "nonexistent"])
        assert result.exit_code != 0


class TestExplainerCommand:
    """Test the explainer command argument validation."""

    def setup_method(self):
        self.runner = CliRunner()

    def test_no_args_fails(self):
        result = self.runner.invoke(app, ["new", "explainer"])
        assert result.exit_code != 0

    def test_no_images_and_images_only_mutually_exclusive(self):
        result = self.runner.invoke(
            app, ["new", "explainer", "test", "--no-images", "--images-only"]
        )
        assert result.exit_code != 0

    @patch("autoshorts.cli.commands.explainer.VIDEO_TYPES")
    @patch("autoshorts.cli.commands.explainer.asyncio.run")
    def test_youtube_url_with_images_only_fails(self, mock_asyncio_run, mock_video_types):
        gen_mock = Mock()
        gen_mock.generate = Mock(return_value=True)
        gen_mock.cleanup = Mock()
        mock_video_types.__getitem__.return_value = lambda **kw: gen_mock

        result = self.runner.invoke(
            app, ["new", "explainer", "--youtube-url", "https://youtube.com/watch?v=123", "--images-only"]
        )
        assert result.exit_code != 0

    @patch("autoshorts.cli.commands.explainer.VIDEO_TYPES")
    @patch("autoshorts.cli.commands.explainer.asyncio.run")
    def test_youtube_url_mode(self, mock_asyncio_run, mock_video_types):
        gen_mock = Mock()
        gen_mock.generate = Mock(return_value=True)
        gen_mock.cleanup = Mock()
        mock_video_types.__getitem__.return_value = lambda **kw: gen_mock
        mock_asyncio_run.return_value = True

        result = self.runner.invoke(
            app,
            ["new", "explainer", "--youtube-url", "https://youtube.com/watch?v=123"],
        )
        assert result.exit_code == 0

    @patch("autoshorts.cli.commands.explainer.VIDEO_TYPES")
    @patch("autoshorts.cli.commands.explainer.asyncio.run")
    def test_subject_mode(self, mock_asyncio_run, mock_video_types):
        gen_mock = Mock()
        gen_mock.generate = Mock(return_value=True)
        gen_mock.cleanup = Mock()
        mock_video_types.__getitem__.return_value = lambda **kw: gen_mock
        mock_asyncio_run.return_value = True

        result = self.runner.invoke(app, ["new", "explainer", "test subject"])
        assert result.exit_code == 0

    @patch("autoshorts.cli.commands.explainer.VIDEO_TYPES")
    @patch("autoshorts.cli.commands.explainer.asyncio.run")
    def test_images_only_mode(self, mock_asyncio_run, mock_video_types):
        gen_mock = Mock()
        gen_mock.generate = Mock(return_value=True)
        gen_mock.cleanup = Mock()
        mock_video_types.__getitem__.return_value = lambda **kw: gen_mock
        mock_asyncio_run.return_value = True

        result = self.runner.invoke(app, ["new", "explainer", "test", "--images-only"])
        assert result.exit_code == 0

    @patch("autoshorts.cli.commands.explainer.VIDEO_TYPES")
    @patch("autoshorts.cli.commands.explainer.asyncio.run")
    def test_batch_mode(self, mock_asyncio_run, mock_video_types):
        gen_mock = Mock()
        gen_mock.generate = Mock(return_value=True)
        gen_mock.cleanup = Mock()
        mock_video_types.__getitem__.return_value = lambda **kw: gen_mock
        mock_asyncio_run.return_value = True

        result = self.runner.invoke(
            app, ["new", "explainer", "--batch", "a", "--batch", "b"]
        )
        assert result.exit_code == 0

    @patch("autoshorts.cli.commands.explainer.VIDEO_TYPES")
    @patch("autoshorts.cli.commands.explainer.asyncio.run")
    def test_goodnight_flag(self, mock_asyncio_run, mock_video_types):
        gen_mock = Mock()
        gen_mock.generate = Mock(return_value=True)
        gen_mock.cleanup = Mock()
        mock_video_types.__getitem__.return_value = lambda **kw: gen_mock
        mock_asyncio_run.return_value = True

        with patch("autoshorts.cli.commands.explainer.shutdown_computer") as mock_shutdown:
            result = self.runner.invoke(
                app, ["new", "explainer", "test", "--goodnight"]
            )
            assert result.exit_code == 0
            mock_shutdown.assert_called_once()

    @patch("autoshorts.cli.commands.explainer.VIDEO_TYPES")
    @patch("autoshorts.cli.commands.explainer.asyncio.run")
    def test_no_images_flag(self, mock_asyncio_run, mock_video_types):
        gen_mock = Mock()
        gen_mock.generate = Mock(return_value=True)
        gen_mock.cleanup = Mock()
        mock_video_types.__getitem__.return_value = lambda **kw: gen_mock
        mock_asyncio_run.return_value = True

        result = self.runner.invoke(app, ["new", "explainer", "test", "--no-images"])
        assert result.exit_code == 0

    @patch("autoshorts.cli.commands.explainer.VIDEO_TYPES")
    @patch("autoshorts.cli.commands.explainer.asyncio.run")
    def test_no_web_search_flag(self, mock_asyncio_run, mock_video_types):
        gen_mock = Mock()
        gen_mock.generate = Mock(return_value=True)
        gen_mock.cleanup = Mock()
        mock_video_types.__getitem__.return_value = lambda **kw: gen_mock
        mock_asyncio_run.return_value = True

        result = self.runner.invoke(app, ["new", "explainer", "test", "--no-web-search"])
        assert result.exit_code == 0

    @patch("autoshorts.cli.commands.explainer.VIDEO_TYPES")
    @patch("autoshorts.cli.commands.explainer.asyncio.run")
    def test_images_flag_web(self, mock_asyncio_run, mock_video_types):
        gen_mock = Mock()
        gen_mock.generate = Mock(return_value=True)
        gen_mock.cleanup = Mock()
        mock_video_types.__getitem__.return_value = lambda **kw: gen_mock
        mock_asyncio_run.return_value = True

        result = self.runner.invoke(app, ["new", "explainer", "test", "--images", "web"])
        assert result.exit_code == 0

    @patch("autoshorts.cli.commands.explainer.VIDEO_TYPES")
    @patch("autoshorts.cli.commands.explainer.asyncio.run")
    def test_images_flag_ai(self, mock_asyncio_run, mock_video_types):
        gen_mock = Mock()
        gen_mock.generate = Mock(return_value=True)
        gen_mock.cleanup = Mock()
        mock_video_types.__getitem__.return_value = lambda **kw: gen_mock
        mock_asyncio_run.return_value = True

        result = self.runner.invoke(app, ["new", "explainer", "test", "--images", "ai"])
        assert result.exit_code == 0

    def test_images_flag_invalid(self):
        result = self.runner.invoke(app, ["new", "explainer", "test", "--images", "invalid"])
        assert result.exit_code != 0

    def test_images_flag_shows_in_help(self):
        result = self.runner.invoke(app, ["new", "explainer", "--help"])
        assert result.exit_code == 0
        assert "--images" in result.output

    @patch("autoshorts.cli.commands.explainer.VIDEO_TYPES")
    @patch("autoshorts.cli.commands.explainer.asyncio.run")
    def test_generator_failure_handled(self, mock_asyncio_run, mock_video_types):
        gen_mock = Mock()
        gen_mock.generate = Mock(return_value=False)
        gen_mock.cleanup = Mock()
        mock_video_types.__getitem__.return_value = lambda **kw: gen_mock
        mock_asyncio_run.return_value = False

        result = self.runner.invoke(app, ["new", "explainer", "test"])
        assert result.exit_code == 0  # CLI handles failures gracefully

    @patch("autoshorts.cli.commands.explainer.VIDEO_TYPES")
    @patch("autoshorts.cli.commands.explainer.asyncio.run")
    def test_generator_exception_handled(self, mock_asyncio_run, mock_video_types):
        gen_mock = Mock()
        gen_mock.generate = Mock(side_effect=Exception("test error"))
        gen_mock.cleanup = Mock()
        mock_video_types.__getitem__.return_value = lambda **kw: gen_mock
        mock_asyncio_run.side_effect = Exception("test error")

        result = self.runner.invoke(app, ["new", "explainer", "test"])
        assert result.exit_code == 0  # CLI handles exceptions gracefully


class TestMainEntry:
    """Test the main entry point."""

    def test_main_calls_app(self):
        with patch("autoshorts.cli.app") as mock_app:
            main()
            mock_app.assert_called_once()
