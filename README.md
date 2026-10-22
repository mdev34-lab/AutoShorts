# AutoShorts — Automated Short-Form Video Generator

AI-powered tool for generating YouTube Shorts / TikTok videos with script generation, text-to-speech, subtitle rendering, and video composition.

## Features

- **AI Script Generation** — curiosity-driven scripts in Brazilian Portuguese via Pollinations AI with fact verification and hallucination guards
- **Web-Grounded Scripts** — automatic web search generates independent queries, grounds the script in real sources, then cross-checks every claim
- **Title Validation** — auto-validates hashtag count (3+), length (≤ 100 chars), and lowercases tags
- **AI Image Generation** — background images via Pollinations AI or **real web images** via DuckDuckGo search (default), with NSFW domain/keyword filter
- **Overlay Text Removal** — captions/watermarks on downloaded images are detected (morphology + OCR confirmation) and erased with OpenCV inpainting; text-heavy graphics pass through untouched
- **Text-to-Speech** — natural audio via Edge TTS
- **Subtitle System** — VTT generation + word-level highlight rendering
- **Video Composition** — blurred YouTube background or AI/web images with smooth overlay animation
- **YouTube Integration** — download any video as background footage
- **Two pipelines**: normal (YouTube bg + optional image overlays) and images-only (AI/web images + overlay animation, no YouTube bg)
- **Typer CLI** — nested subcommands, auto-generated `--help`, shell completion
- **Batch Processing** — semicolon-separated subjects for multi-video runs

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

# Generate an explainer video from a topic (web search + web images by default)
autoshorts new explainer "artificial intelligence"

# AI images only (no YouTube background)
autoshorts new explainer "space exploration" --images-only

# Use a YouTube video as background footage
autoshorts new explainer --youtube-url "https://youtube.com/watch?v=VIDEO_ID"

# Skip image overlays (blurred bg only)
autoshorts new explainer "climate change" --no-images

# Batch mode with semicolon-separated subjects
autoshorts new explainer --batch "robotics; quantum computing; neural networks"

# Image source: 'ai' uses Pollinations (default is 'web' via DDGS)
autoshorts new explainer "oceanography" --images ai

# Script tone: 'corporate' (neutral, factual) or 'opinionated' (curiosity-driven, narrative)
autoshorts new explainer "bitcoin" --tone corporate

# Disable web search (uses model knowledge only)
autoshorts new explainer "neural networks" --no-web-search

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
│           ├── image_searcher.py   # Web/AI image search + NSFW filter
│           ├── logging_system.py
│           ├── script_generator.py # Script gen, fact verification, title validation
│           ├── subtitle_system.py
│           ├── tts_system.py
│           ├── utils.py
│           ├── video_background.py # YouTube search & download
│           ├── video_compositor.py
│           └── web_search.py       # DuckDuckGo web search
├── tests/
│   ├── test_cli.py                 # CLI layer (32 tests)
│   ├── test_config.py
│   ├── test_edge_cases.py
│   ├── test_fluximages.py          # Explainer generator tests
│   ├── test_init.py
│   ├── test_integration.py
│   ├── test_script_generator.py
│   ├── test_subtitle_system.py
│   ├── test_tts_system.py
│   ├── test_utils.py
│   ├── test_video_background.py    # Video background (24 tests)
│   ├── test_video_compositor.py    # Video compositor (11 tests)
│   ├── test_web_search.py
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
- `duckduckgo-search` — web search and image search (DDGS)
- `Pillow` — image processing and resizing
- `webvtt-py` — subtitle processing
- `python-dotenv` — environment loading
- `typer` — CLI framework

Dev:
- `pytest` + `pytest-asyncio` + `pytest-cov`
- `ruff` + `mypy`

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
