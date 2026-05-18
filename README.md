# AutoShorts — Automated Short-Form Video Generator

AI-powered tool for generating YouTube Shorts / TikTok videos with script generation, text-to-speech, subtitle rendering, and video composition.

## Features

- **AI Script Generation** — viral-optimized scripts in Brazilian Portuguese via Pollinations AI
- **AI Image Generation** — background images via Pollinations AI (images-only mode)
- **Text-to-Speech** — natural audio via Edge TTS
- **Subtitle System** — VTT generation + word-level highlight rendering
- **Video Composition** — blurred YouTube background or AI images with smooth overlay animation
- **YouTube Integration** — download any video as background footage
- **Two pipelines**: normal (YouTube bg + optional AI image overlays) and images-only (AI images + overlay animation, no YouTube bg)
- **Typer CLI** — nested subcommands, auto-generated `--help`, shell completion

## Installation

### Prerequisites
- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Steps
```bash
git clone https://github.com/yourusername/AutoShorts.git
cd AutoShorts

uv venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

uv pip install -e ".[dev]"
# or: pip install -e ".[dev]"
```

## Quick Start

```bash
# See available commands
autoshorts --help

# Generate an explainer video from a topic
autoshorts new explainer "artificial intelligence"

# AI images only (no YouTube background)
autoshorts new explainer "space exploration" --images-only

# Use a YouTube video as background footage
autoshorts new explainer --youtube-url "https://youtube.com/watch?v=VIDEO_ID"

# Skip AI image overlays (blurred bg only)
autoshorts new explainer "climate change" --no-images

# Batch mode
autoshorts new explainer --batch "robotics" "quantum computing" "neural networks"

# Web search for richer script content
autoshorts new explainer "oceanography" --web-search

# Auto-shutdown after completion
autoshorts new explainer "future technology" --goodnight

# Get help for any command
autoshorts help
autoshorts help new
autoshorts help new explainer
```

## Configuration

Copy `.env.example` to `.env` and edit:

```env
API_KEY=your_pollinations_api_key_here
API_URL=https://gen.pollinations.ai/v1/chat/completions
IMG_URL=https://gen.pollinations.ai/image/
MODEL_TEXT=nova-fast
MODEL_IMAGE=zimage
VOICE=pt-BR-AntonioNeural
TTS_RATE=+20%
OUTPUT_DIR=output
DEFAULT_FONT=fonts/BebasNeue-Regular.ttf
VIDEO_WIDTH=1080
VIDEO_HEIGHT=1920
VIDEO_FPS=24
```

## Project Structure

```
AutoShorts/
├── src/
│   └── autoshorts/
│       ├── __init__.py
│       ├── main.py                 # Entry point → app()
│       ├── cli/                    # Typer CLI
│       │   ├── __init__.py         # Root app + `help` command
│       │   ├── new.py              # `new` command group
│       │   └── commands/
│       │       ├── explainer.py    # `new explainer` subcommand
│       │       └── help_cmd.py     # `help` subcommand factory
│       ├── generators/             # Plugin-style generator registry
│       │   ├── __init__.py         # VIDEO_TYPES = {"explainer": ...}
│       │   └── explainer.py        # ExplainerGenerator (both pipelines)
│       └── modules/                # Core modules
│           ├── config.py
│           ├── logging_system.py
│           ├── script_generator.py
│           ├── subtitle_system.py
│           ├── tts_system.py
│           ├── utils.py
│           ├── video_background.py
│           └── video_compositor.py
├── tests/
│   ├── test_cli.py                 # CLI layer (28 tests)
│   ├── test_fluximages.py          # Explainer generator tests
│   ├── test_video_background.py    # Video background (24 tests)
│   ├── test_video_compositor.py    # Video compositor (11 tests)
│   ├── test_script_generator.py
│   ├── test_subtitle_system.py
│   ├── test_utils.py
│   ├── test_edge_cases.py
│   ├── test_tts_system.py
│   └── conftest.py
├── fonts/                          # Bundled Bebas Neue font
├── .env.example
├── pyproject.toml
├── uv.lock
└── README.md
```

## Dependencies

Core:
- `moviepy` — video editing and composition
- `edge-tts` — text-to-speech
- `requests` — HTTP client
- `yt-dlp` — YouTube downloading
- `webvtt-py` — subtitle processing
- `python-dotenv` — environment loading
- `typer` — CLI framework

Dev:
- `pytest` + `pytest-asyncio` + `pytest-cov`
- `black` + `ruff` + `mypy`

## Development

```bash
# Run tests (omit --cov on Windows to avoid numpy multiprocessing conflict)
pytest tests/

# Format & lint
black src/ tests/
ruff check src/ tests/
mypy src/
```

## License

MIT — see LICENSE.
