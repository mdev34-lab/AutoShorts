# AutoShorts - Automated Viral Short-Form Video Generator

A Python tool for generating automated short-form video content (YouTube Shorts/TikTok) with AI-powered scripts, text-to-speech, subtitle generation, and video composition.

## Features

- **AI Script Generation**: Generate engaging, viral-optimized scripts in Brazilian Portuguese (PT-BR) using Pollinations AI
- **AI Image Generation**: Create custom background images via Pollinations AI image API
- **Text-to-Speech**: Convert scripts to natural-sounding audio using Edge TTS
- **Subtitle System**: Automatic subtitle generation, rendering, and word-level highlighting
- **Video Composition**: Create professional videos with AI-generated images or YouTube video backgrounds
- **YouTube Integration**: Download and summarize YouTube videos for content repurposing
- **Multiple Modes**: Standard flux image mode, YouTube summarizer mode, and experimental background processing

## Installation

### Prerequisites
- Python 3.8+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Steps
```bash
# Clone the repository
git clone https://github.com/yourusername/AutoShorts.git
cd AutoShorts

# Create virtual environment (uv recommended)
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install package in development mode with dev dependencies
uv pip install -e ".[dev]"
# OR with pip:
pip install -e ".[dev]"
```

## Quick Start

### 1. Configure Environment
Copy the example environment file and add your API key:
```bash
cp .env.example .env
```
Edit `.env` and add your Pollinations AI API key (get one at [gen.pollinations.ai](https://gen.pollinations.ai)).

### 2. Generate a Video
#### Flux Image Mode (AI-generated backgrounds)
```bash
# Generate a video about a topic
python -m autoshorts.fluximages "artificial intelligence"

# Batch process multiple topics
python -m autoshorts.fluximages --batch "space exploration" "machine learning" "robotics"

# Use web search for enhanced content
python -m autoshorts.fluximages "climate change" --web-search

# Auto-shutdown after processing
python -m autoshorts.fluximages "future technology" --goodnight
```

#### YouTube Summarizer Mode
```bash
# Summarize a YouTube video into a short
python -m autoshorts.yt_summarizer "https://www.youtube.com/watch?v=VIDEO_ID"
```

#### Experimental Mode (YouTube background videos)
```bash
autoshorts "space exploration"
```

## Configuration

All configuration is managed via the `.env` file. See `.env.example` for all available options:

```env
# API Configuration (Pollinations AI)
API_KEY=your_pollinations_api_key_here
API_URL=https://gen.pollinations.ai/v1/chat/completions
IMG_URL=https://gen.pollinations.ai/image/

# Model Configuration
MODEL_TEXT=nova-fast  # Text generation model
MODEL_IMAGE=zimage   # Image generation model

# TTS Configuration
VOICE=pt-BR-AntonioNeural  # Edge TTS voice (PT-BR)
TTS_RATE=+20%              # Speech rate adjustment

# Output Settings
OUTPUT_DIR=output          # Generated video directory
DEFAULT_FONT=fonts/BebasNeue-Regular.ttf

# Video Settings
VIDEO_WIDTH=1080
VIDEO_HEIGHT=1920
VIDEO_FPS=24
```

## Project Structure

```
AutoShorts/
├── src/
│   └── autoshorts/
│       ├── __init__.py          # Package initialization
│       ├── fluximages.py        # Flux image video generator
│       ├── experimental.py      # Experimental YouTube+AI image mode
│       ├── yt_summarizer.py     # YouTube video summarizer
│       └── modules/             # Core modules
│           ├── config.py        # Configuration loader
│           ├── logging_system.py # Logging utilities
│           ├── script_generator.py # AI script generation
│           ├── subtitle_system.py # Subtitle generation/rendering
│           ├── tts_system.py    # Text-to-speech system
│           ├── video_background.py # Background video processing
│           ├── video_compositor.py # Video editing/composition
│           └── utils.py         # Utility functions
├── tests/                       # Unit and integration tests
├── fonts/                       # Custom fonts (Bebas Neue)
├── .env.example                 # Example environment configuration
├── .gitignore                   # Git ignore rules
├── pyproject.toml               # Project metadata and dependencies
├── uv.lock                      # Dependency lock file (uv)
└── README.md                    # This file
```

## Dependencies

Core dependencies (installed automatically):
- `moviepy` - Video editing and composition
- `edge-tts` - Text-to-speech synthesis
- `requests` - HTTP client for API calls
- `yt-dlp` - YouTube video downloading
- `webvtt-py` - Subtitle file processing
- `python-dotenv` - Environment variable loading

Dev dependencies (optional, for development):
- `pytest` - Testing framework
- `pytest-cov` - Test coverage
- `black` - Code formatter
- `ruff` - Linter
- `mypy` - Type checker

## Development

### Running Tests
```bash
pytest tests/ --cov=src --cov-report=term-missing
```

### Code Quality
```bash
# Format code
black src/ tests/

# Lint code
ruff check src/ tests/

# Type check
mypy src/
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a pull request

## Support

For issues and questions, please open an issue on the GitHub repository.
