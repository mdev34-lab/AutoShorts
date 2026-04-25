# AutoShorts - Automated Viral Video Generator

A Python package for generating automated short-form video content with AI-powered scripts, text-to-speech, and subtitle generation.

## Features

- **AI Script Generation**: Generate engaging video scripts using advanced language models
- **Text-to-Speech**: Convert scripts to natural-sounding audio using Edge TTS
- **Subtitle System**: Automatic subtitle generation and rendering
- **Video Composition**: Create professional videos with AI-generated images and effects
- **YouTube Integration**: Download and summarize YouTube videos for content repurposing

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd AutoShorts

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install package in development mode
pip install -e .
```

## Usage

### Flux Image Video Generation

```bash
# Generate a video from a topic
python -m autoshorts.fluximages "artificial intelligence"

# Batch processing multiple topics
python -m autoshorts.fluximages --batch "space exploration" "machine learning" "robotics"

# Use web search for enhanced content
python -m autoshorts.fluximages "climate change" --web-search

# Auto-shutdown after processing
python -m autoshorts.fluximages "future technology" --goodnight
```

### YouTube Video Summarization

```bash
# Summarize a YouTube video
python -m autoshorts.yt_summarizer "https://www.youtube.com/watch?v=VIDEO_ID"
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
│       ├── modules/             # Core modules
│       │   ├── config.py        # Configuration constants
│       │   ├── logging_system.py # Logging utilities
│       │   ├── script_generator.py # AI script generation
│       │   ├── subtitle_system.py # Subtitle generation and rendering
│       │   ├── tts_system.py    # Text-to-speech system
│       │   └── utils.py         # Utility functions
│       └── ...
├── tests/                       # Unit tests
├── output/                      # Generated videos
├── .env                         # Environment variables
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

## Configuration

Create a `.env` file in the project root:

```env
# API Configuration
API_KEY=your_api_key_here
API_URL=https://api.example.com/v1
IMG_URL=https://api.example.com/v1/images

# Output Settings
OUTPUT_DIR=output
DEFAULT_FONT=Arial
```

## Dependencies

- **moviepy**: Video editing and composition
- **edge-tts**: Text-to-speech synthesis
- **requests**: HTTP client for API calls
- **yt-dlp**: YouTube video downloading
- **pathlib**: Path manipulation
- **asyncio**: Asynchronous processing

## Development

### Running Tests

```bash
pytest tests/
```

### Code Style

This project follows PEP 8 style guidelines. Use linting tools to maintain code quality.

## License

This project is licensed under the MIT License.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## Support

For issues and questions, please open an issue on the GitHub repository.
