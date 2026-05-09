#!/usr/bin/env python3
"""
Video Background Module

Provides unified YouTube video search and download functionality for AutoShorts.
Handles search query generation, video filtering, and downloading.
"""

import requests  # type: ignore[import-untyped]
import yt_dlp

from .config import (
    API_KEY,
    API_TIMEOUT_SEARCH,
    API_URL,
    MAX_VIDEO_DURATION,
    MIN_VIDEO_DURATION,
    MODEL_TEXT,
    YOUTUBE_FORMAT,
    YOUTUBE_MAX_HEIGHT,
)
from .utils import create_temp_dir, log


class VideoBackgroundManager:
    """Handles YouTube video search and download for video backgrounds."""

    def __init__(self):
        self.temp_dir = create_temp_dir()
        self.ydl_opts = {
            "format": YOUTUBE_FORMAT.format(max_height=YOUTUBE_MAX_HEIGHT),
            "outtmpl": str(self.temp_dir / "source_video.%(ext)s"),
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "extract_flat": False,
            "check_formats": False,
            "http_headers": {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
            },
            "no_check_certificate": True,
            "ignoreerrors": False,
            "fragment_retries": 3,
            "retry_sleep_functions": [lambda x: 1, lambda x: 2],
        }

    def generate_search_query(self, subject: str) -> str:
        """Generate an AI-optimized YouTube search query."""
        log("Generating AI-optimized search query...")

        system_prompt = """You are an expert at crafting YouTube search queries to find high-quality, family-friendly content.
CRITICAL RULES:
1. Output ONLY a single search query string in ENGLISH
2. Translate the subject to English first, then create the search query
3. Use NATURAL English with spaces, NOT dashes
4. Include terms like "explained", "story", "documentary", "educational"
5. Add "-shorts" to exclude YouTube Shorts
6. Make it specific and searchable
7. NO quotes, NO special formatting, NO excessive dashes
8. DO NOT just append the original text to template words
9. Example: "mysterious flash drive found on street explained documentary"
"""

        user_prompt = f"Subject: {subject}\n\nCreate an ENGLISH YouTube search query that will find family-friendly, educational videos about this subject. Focus on documentary-style content, news reports, or educational explanations. Avoid anything that might be age-restricted."

        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": MODEL_TEXT,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": 50,
            "temperature": 0.3,
        }

        try:
            response = requests.post(
                API_URL, headers=headers, json=payload, timeout=API_TIMEOUT_SEARCH
            )
            response.raise_for_status()
            data = response.json()

            raw_content = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )
            log(f"Raw AI response: '{raw_content}'", "INFO")

            query = raw_content.replace('"', "").replace("'", "").strip()

            if len(query) < 10 or query.lower().endswith(subject.lower()[:20]):
                raise ValueError(f"Generated query is too generic: {query}")

            log(f"Generated search query: {query}", "SUCCESS")
            return query

        except Exception as e:
            log(f"Failed to generate AI query: {e}", "ERROR")
            raise

    def _is_suitable_video(
        self,
        video_info: dict,
        min_duration: int = MIN_VIDEO_DURATION,
        max_duration: int = MAX_VIDEO_DURATION,
    ) -> bool:
        """Filter videos based on duration and availability."""
        duration = video_info.get("duration", 0)
        title = video_info.get("title", "").lower()

        if (
            video_info.get("availability") == "private"
            or video_info.get("availability") == "unavailable"
        ):
            log(f"FILTERED: '{title[:30]}...' - Video not available", "WARNING")
            return False

        if not video_info.get("id") or not video_info.get("webpage_url"):
            log(f"FILTERED: '{title[:30]}...' - Missing video ID or URL", "WARNING")
            return False

        if duration < min_duration:
            log(f"FILTERED: '{title[:30]}...' - Too short: {duration}s", "WARNING")
            return False
        elif duration > max_duration:
            log(f"FILTERED: '{title[:30]}...' - Too long: {duration}s", "WARNING")
            return False
        return True

    def _extract_error_message(self, exc: Exception) -> str:
        """Extract a readable message from yt-dlp exceptions."""
        if hasattr(exc, "msg") and exc.msg:
            return str(exc.msg)
        if hasattr(exc, "excn_msg") and exc.excn_msg:
            return str(exc.excn_msg)
        if isinstance(exc, dict):
            msg = exc.get("msg") or exc.get("error") or ""
            return str(msg) if msg else str(exc)
        if isinstance(exc, list) and exc:
            first_item = exc[0]
            if isinstance(first_item, dict):
                return first_item.get("msg") or first_item.get("error") or str(first_item)
            return str(first_item)
        return str(exc)

    def _pytubefix_video_to_dict(self, video) -> dict:
        """Convert a pytubefix video object to a dict compatible with _is_suitable_video."""
        return {
            "duration": getattr(video, "length", 0) or 0,
            "title": getattr(video, "title", "") or "",
            "id": getattr(video, "video_id", "") or "",
            "webpage_url": getattr(video, "watch_url", "") or "",
            "availability": None,
        }

    def _download_with_pytubefix(self, video_url: str, output_dir, title: str) -> str | None:
        """Try to download a video using pytubefix. Returns path or None."""
        try:
            from pytubefix import YouTube

            log(f"Attempting pytubefix download: {title[:40]}...", "INFO")
            yt = YouTube(video_url, "WEB")
            ys = yt.streams.get_highest_resolution()
            if not ys:
                log("No streams available via pytubefix", "WARNING")
                return None
            downloaded = ys.download(output_path=str(output_dir))
            if downloaded:
                log(f"Successfully downloaded via pytubefix: {title}", "SUCCESS")
                return downloaded
            return None
        except Exception as e:
            log(f"pytubefix failed: {e}", "WARNING")
            return None

    def search_and_download(self, subject: str) -> str:
        """Search and download video using pytubefix first, fall back to yt-dlp."""
        search_query = self.generate_search_query(subject)

        # Step 1: Try pytubefix search + download
        video_path = self._try_pytubefix(search_query)
        if video_path:
            return video_path

        # Step 2: Fall back to yt-dlp
        log("pytubefix failed, falling back to yt-dlp...", "WARNING")
        return self._try_ytdlp(search_query)

    def _try_pytubefix(self, search_query: str) -> str | None:
        """Search and download using pytubefix. Returns path or None."""
        try:
            from pytubefix import Search

            log("Searching with pytubefix...", "INFO")
            results = Search(search_query)
            if not results.videos:
                log("No pytubefix results", "WARNING")
                return None

            suitable = []
            for v in results.videos:
                info = self._pytubefix_video_to_dict(v)
                if self._is_suitable_video(info):
                    suitable.append((v.watch_url, info["title"]))

            if not suitable:
                log("No suitable videos found via pytubefix", "WARNING")
                return None

            download_dir = create_temp_dir()
            for i, (url, title) in enumerate(suitable[:10]):
                log(f"Attempting video {i + 1}: {title[:40]}...", "INFO")
                path = self._download_with_pytubefix(url, download_dir, title)
                if path:
                    return path

            return None
        except Exception as e:
            log(f"pytubefix search failed: {e}", "WARNING")
            return None

    def _try_ytdlp(self, search_query: str) -> str:
        """Fallback search and download using yt-dlp."""
        search_query = f"ytsearch20:{search_query}"

        with yt_dlp.YoutubeDL(
            {"quiet": True, "no_warnings": True, "ignoreerrors": True}
        ) as ydl:
            try:
                info = ydl.extract_info(search_query, download=False)

                if "entries" not in info or not info["entries"]:
                    raise ValueError("No video found")

                all_videos = [v for v in info["entries"] if v is not None]

                if not all_videos:
                    raise ValueError("No videos found in search results")

                suitable_videos = [v for v in all_videos if self._is_suitable_video(v)]

                if not suitable_videos:
                    for v in info["entries"]:
                        if (
                            v
                            and v.get("id")
                            and not v.get("title", "").lower().startswith("[deleted]")
                        ):
                            suitable_videos = [v]
                            break

                if not suitable_videos:
                    raise ValueError("No suitable videos found in search results")

                download_temp_dir = create_temp_dir()
                ydl_opts_with_dir = self.ydl_opts.copy()
                ydl_opts_with_dir["outtmpl"] = str(
                    download_temp_dir / "source_video.%(ext)s"
                )

                for attempt, video in enumerate(suitable_videos[:10]):
                    video_url = video.get("webpage_url", "")
                    if not video_url:
                        continue

                    log(
                        f"Attempting video {attempt + 1}: {video.get('title', 'Unknown')[:40]}...",
                        "INFO",
                    )

                    try:
                        with yt_dlp.YoutubeDL(ydl_opts_with_dir) as ydl:
                            ydl.download([video_url])
                            log(
                                f"Successfully downloaded: {video.get('title', 'Unknown')}",
                                "SUCCESS",
                            )
                            video_files = list(download_temp_dir.glob("source_video.*"))
                            if video_files:
                                return str(video_files[0])
                            return str(download_temp_dir / "source_video.mp4")
                    except Exception as e:
                        error_msg = self._extract_error_message(e)
                        error_lower = error_msg.lower()
                        if (
                            "not available" in error_lower
                            or "unavailable" in error_lower
                            or "private" in error_lower
                        ):
                            log(
                                "Video unavailable, trying next candidate...", "WARNING"
                            )
                            continue
                        else:
                            log(f"Download failed: {error_msg}", "ERROR")
                            if attempt == len(suitable_videos) - 1:
                                raise

                raise ValueError("No available videos could be downloaded")
            except Exception as e:
                log(f"Search failed: {e}", "ERROR")
                raise

    def download_from_url(self, url: str) -> tuple:
        """Download directly from URL and return video path and metadata."""
        log(f"Downloading from URL: {url}")

        with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True}) as ydl:
            try:
                info = ydl.extract_info(url, download=False)
                title = info.get("title", "")
                description = info.get("description", "")
                log(f"Video title: {title}")
                log(f"Description length: {len(description)} chars")
            except Exception as e:
                log(f"Failed to extract metadata: {e}", "WARNING")
                title = ""
                description = ""

        with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
            ydl.download([url])

        video_files = list(self.temp_dir.glob("source_video.*"))
        if not video_files:
            raise FileNotFoundError("Video download failed")

        return str(video_files[0]), title, description
