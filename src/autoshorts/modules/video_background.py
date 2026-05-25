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
                "Accept-Language": "pt-BR,pt;q=0.9,en;q=0.5",
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

        system_prompt = """You are an expert at crafting YouTube search queries. Output ONLY the query, nothing else.
CRITICAL RULES:
1. Use the SAME language as the subject (do NOT translate)
2. Use NATURAL language with spaces, NOT dashes
3. Add "-shorts" at the end to exclude YouTube Shorts
4. Be specific to the subject — use concrete names, events, places
5. NO generic template words like "explicado", "document\u00e1rio", "reportagem", "hist\u00f3ria"
6. NO quotes, NO special formatting
7. DO NOT just repeat the subject — add specific qualifiers
"""

        user_prompt = f"Subject: {subject}\n\nCreate a YouTube search query that returns videos directly about this subject. Be specific."

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
        subject: str | None = None,
    ) -> bool:
        """Filter videos based on duration, availability, and title relevance."""
        duration = video_info.get("duration", 0)
        title = (video_info.get("title") or "").lower()

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

        if subject:
            subject_lower = subject.lower()
            subject_words = [w for w in subject_lower.split() if len(w) > 3]
            if subject_words and not any(w in title for w in subject_words):
                log(
                    f"FILTERED: '{title[:40]}...' - No subject keywords in title",
                    "WARNING",
                )
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
                return (
                    first_item.get("msg") or first_item.get("error") or str(first_item)
                )
            return str(first_item)
        return str(exc)

    def search_and_download(self, subject: str) -> str:
        """Search and download video using DDG first, then yt-dlp search as fallback."""
        search_query = self.generate_search_query(subject)

        video_path = self._search_with_ddg(search_query, subject)
        if video_path:
            return video_path

        log("DDG search failed, falling back to yt-dlp search...", "WARNING")
        return self._search_with_ytdlp(search_query, subject)

    def _search_with_ddg(self, search_query: str, subject: str | None = None) -> str | None:
        """Search YouTube via DuckDuckGo and download with yt-dlp."""
        try:
            from ddgs import DDGS

            log("Searching YouTube via DuckDuckGo...", "INFO")
            results = list(
                DDGS().text(f"site:youtube.com {search_query}", max_results=10)
            )
            urls = [
                r.get("href") for r in results if "youtube.com/watch" in r.get("href", "")
            ]
            if not urls:
                log("No YouTube URLs found via DDG", "WARNING")
                return None
            log(f"Found {len(urls)} YouTube videos, extracting metadata...", "INFO")
            return self._download_first_suitable(urls, subject)
        except Exception as e:
            log(f"DDG search failed: {e}", "WARNING")
            return None

    def _search_with_ytdlp(self, search_query: str, subject: str | None = None) -> str:
        """Fallback search using yt-dlp built-in search."""
        yt_query = f"ytsearch20:{search_query}"

        with yt_dlp.YoutubeDL(
            {"quiet": True, "no_warnings": True, "ignoreerrors": True}
        ) as ydl:
            try:
                info = ydl.extract_info(yt_query, download=False)

                if "entries" not in info or not info["entries"]:
                    raise ValueError("No video found")

                all_videos = [v for v in info["entries"] if v is not None]

                if not all_videos:
                    raise ValueError("No videos found in search results")

                suitable_videos = [v for v in all_videos if self._is_suitable_video(v, subject=subject)]

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

                urls = [
                    v.get("webpage_url", "")
                    for v in suitable_videos[:10]
                    if v.get("webpage_url")
                ]
                path = self._download_first_suitable(urls, subject)
                if path:
                    return path
                raise ValueError("No available videos could be downloaded")
            except Exception as e:
                log(f"yt-dlp search failed: {e}", "ERROR")
                raise

    def _download_first_suitable(self, urls: list[str], subject: str | None = None) -> str | None:
        """Try URLs one by one, return path of first successful download."""
        download_temp_dir = create_temp_dir()
        ydl_opts_with_dir = self.ydl_opts.copy()
        ydl_opts_with_dir["outtmpl"] = str(download_temp_dir / "source_video.%(ext)s")

        for attempt, video_url in enumerate(urls):
            if not video_url:
                continue

            # Get metadata first to filter unsuitable videos
            title = "Unknown"
            try:
                with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True}) as ydl:
                    info = ydl.extract_info(video_url, download=False)
                    if not self._is_suitable_video(info, subject=subject):
                        log(f"Skipping unsuitable video {attempt + 1}", "WARNING")
                        continue
                    title = info.get("title", "Unknown")
            except Exception as e:
                error_msg = self._extract_error_message(e)
                if (
                    "not available" in error_msg.lower()
                    or "private" in error_msg.lower()
                ):
                    log(f"Video {attempt + 1} unavailable, trying next...", "WARNING")
                    continue
                log(f"Metadata extraction failed: {error_msg}", "WARNING")

            log(f"Attempting video {attempt + 1}: {title[:40]}...", "INFO")

            try:
                with yt_dlp.YoutubeDL(ydl_opts_with_dir) as ydl:
                    ydl.download([video_url])
                    log(f"Successfully downloaded: {title}", "SUCCESS")
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
                    log("Video unavailable, trying next candidate...", "WARNING")
                    continue
                else:
                    log(f"Download failed: {error_msg}", "ERROR")
                    if attempt == len(urls) - 1:
                        raise

        return None

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
