# AutoShorts CLI Reference

## Basic usage

```
autoshorts new explainer [SUBJECT] [OPTIONS]
```

## Options

| Option | Description |
|---|---|
| `-o, --output DIR` | Output directory (default: `output`) |
| `-y, --youtube-url URL` | Direct YouTube URL (skips web search for background) |
| `--images-only` | AI-generated images only, no YouTube background |
| `--no-images` | Skip image overlays entirely |
| `--images {web,ai}` | Image source: `web` or `ai` (Pollinations). Default: `web`. Web photos are used only for broad generic requests that pass the license allowlist, an OCR pixel gate and a local CLIP class check (optional `subject` extra and an explicit model download; the model mirror declares no license, the upstream OpenAI CLIP repo is MIT). Requests naming people, teams, products, dates or events use AI images. `ALLOW_UNVERIFIED_IMAGES=1` disables all of these gates; entries it creates are marked `unverified_opt_in` in their credits record. |
| `--no-web-search` | Disable web search (model knowledge only) |
| `--batch "topic1; topic2"` | Semicolon-separated subjects for batch mode |
| `-c, --custom-instructions TEXT` | Extra instructions for the AI scriptwriter |
| `--goodnight` | Shutdown computer after processing |

## Examples

```bash
# Basic explainer
autoshorts new explainer "Como a Lua se formou"

# Images-only (no YouTube bg)
autoshorts new explainer "Segredos do Titanic" --images-only

# Custom instructions
autoshorts new explainer "Fla-Flu" -c "Foco no recorde de público de 1963"

# From a specific YouTube video
autoshorts new explainer -y "https://youtube.com/watch?v=..."

# Batch mode
autoshorts new explainer --batch "Tópico 1; Tópico 2; Tópico 3"

# With AI-generated images
autoshorts new explainer "Fenômenos da natureza" --images ai
```

## Image text removal

Downloaded web images are cleaned before compositing: overlay text (captions,
prices, watermarks) is detected and inpainted away. Requires the `tesseract`
binary; disable with `IMAGE_TEXT_REMOVAL=off`. Text-heavy graphics (tables,
posters) are detected by coverage and kept untouched. See `.env.example`.
## Image framing

Downloaded web images are reframed to the video aspect ratio around their subject: faces are located with OpenCV's bundled Haar cascade, otherwise a spectral-residual saliency centroid picks the interesting region. When the aspect mismatch is small the image is cover-cropped around that point; when it is large (e.g. a wide landscape photo in a 9:16 short) the frame is filled with a blurred cover-scaled copy and the sharp image is placed on top, so nothing of the subject is cut out. Disable with `SUBJECT_FRAMING=false`; tune with `SUBJECT_FILL_ZOOM_THRESHOLD` and `SUBJECT_FILL_BLUR_RADIUS`.
## TTS engines

Edge TTS (cloud) is the default. For a fully offline local neural voice, use Piper: install the optional dependency with `pip install "autoshorts[piper]"` and set `TTS_ENGINE=piper`. An unrecognized `TTS_ENGINE` value (for example a typo) raises an error instead of silently falling back to the cloud engine. The configured voice (default `PIPER_VOICE_ID=pt_BR-faber-medium`) downloads once from Hugging Face into `PIPER_CACHE_DIR` (`~/.cache/autoshorts/piper`); cached voice files pass integrity checks before reuse (a corrupt or truncated cache is re-downloaded), and the model can be pinned with `PIPER_VOICE_SHA256`. Narration is synthesized to WAV and converted to mp3 with ffmpeg, so the rest of the pipeline works as with Edge TTS, and subtitles follow the real per-sentence audio timings. `TTS_RATE` (e.g. `+20%`) is mapped to Piper's speaking rate, so narration length stays close to Edge TTS, though not identical.
## Product videos

`autoshorts new product <shopee-link>` generates a showcase video from a Shopee product link. The link is parsed locally (the slug is the only product data - Shopee pages and CDN are never fetched; shp.ee short links are resolved through a guarded redirect resolver that stops before Shopee). Photos come from public image search, and every photo is only used when its source page proves the exact model code from the slug in its title/H1 AND the image appears in that page's gallery with model-labelled alt text - any doubt aborts before download. The narration is composed deterministically from facts verified in the product name (specs, resolution, features) - no free text generation, so no fabricated specification can be narrated; a fail-closed guard re-checks the composed script before use. Narration goes through TTS-only pronunciation respellings (brand names, model codes like HY300, decimals); subtitles keep the original spelling. Extra respellings: `--pronunciation-file respellings.json` or `AUTOSHORTS_PRONUNCIATION_FILE`.

## Requirements

- Python 3.10+
- `.env` file with API keys (see `.env.example`)
- ffmpeg installed and in PATH
- `.venv` with dependencies: `pip install -e ".[dev]"`

## Output

Videos are saved as `.mp4` files (720x1280, 15fps, H.264) in the output directory with auto-generated names including the sanitized subject and timestamp.
