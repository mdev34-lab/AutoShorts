"""
Script generation module for AutoShorts

Combines script generation functionality from both generators.
"""

import json

import requests  # type: ignore[import-untyped]

from .config import (
    API_KEY,
    API_TIMEOUT_TEXT,
    API_URL,
    MODEL_TEXT,
)
from .logging_system import log


class ScriptGenerator:
    """Unified script generation using Pollinations API."""

    # Concrete opening hooks optimized for retention (first 1-3 seconds)
    CONCRETE_HOOKS = [
        "em [ano] algo mudou para sempre",
        "tudo começou quando",
        "o marco histórico que",
        "em uma data que",
        "o momento exato em que",
        "a origem de tudo",
        "foi em [ano] que",
    ]

    def __init__(self, web_search: bool = False):
        self.api_url = API_URL
        self.api_key = API_KEY
        self.model = "gemini-fast" if web_search else MODEL_TEXT
        self.web_search = web_search
        self.tools = (
            [{"type": "function", "function": {"name": "google_search"}}]
            if web_search
            else None
        )

    def generate_script(self, subject: str) -> list:
        """Generate script from subject."""
        log("Generating script...")

        system_prompt = """You are a master storyteller for viral YouTube Shorts.
CRITICAL RETENTION RULES:
1.Write in Brazilian Portuguese (PT-BR).
2.First paragraph MUST start with a SPECIFIC FACT (date, number, name, place) — not a generic teaser.
3.NEVER start with "ninguém sabia", "o segredo", "a verdade escondida" or "você não vai acreditar" — these are vague and weak.
4.Always lead with concrete, specific details: "Em 1914...", "Tudo começou quando...", "O placar foi 8 a 0..."
5.Include origin stories — explain HOW something started or WHY it matters, not just THAT it exists.
6.Each paragraph 2-3 sentences for pacing (each ~3 seconds of audio).
7.Write exactly 4-5 paragraphs.
8.NO markdown formatting, NO JSON, just plain text paragraphs.
9.Every paragraph must advance the story with a new specific fact — no filler."""

        user_prompt = f"""Crie uma história envolvente em 4-5 parágrafos sobre "{subject}".

REGRAS CRÍTICAS:
- Primeiro parágrafo DEVE começar com um FATO CONCRETO (data, número, nome, lugar) — NÃO use ganchos genéricos
- NUNCA comece com "ninguém sabia", "o segredo", "a verdade escondida" — isso é vago e fraco
- Inclua nomes, datas, lugares e números específicos sempre que possível
- Conte a ORIGEM: como tudo começou, por que existe
- Cada parágrafo deve avançar a história com um novo fato concreto
- NADA de frases de enchimento

Escreva cada parágrafo em uma linha separada."""

        return self._make_text_api_call(system_prompt, user_prompt)

    def generate_script_from_metadata(self, title: str, description: str) -> list:
        """Generate script from YouTube video title and description."""
        log("Generating script from video metadata...")

        # Create a combined prompt from title and description
        desc = description[:1000] if description else ""
        combined_content = f"Title: {title}\n\nDescription: {desc}"

        system_prompt = """You are a master storyteller for viral YouTube Shorts.
CRITICAL RETENTION RULES:
1.Write in Brazilian Portuguese (PT-BR).
2.First paragraph MUST start with a SPECIFIC FACT from the video (date, number, name, place).
3.NEVER start with "ninguém sabia", "o segredo", or "a verdade escondida" — these are vague and weak.
4.Extract concrete details from the video metadata: dates, names, places, statistics, historical context.
5.Include origin stories — explain HOW something started, not just THAT it happened.
6.Each paragraph 2-3 sentences for pacing (each ~3 seconds of audio).
7.Write exactly 4-5 paragraphs.
8.NO markdown formatting, NO JSON, just plain text paragraphs.
9.Base your story entirely on the video's content, adding only well-known historical context."""

        user_prompt = f"""Crie uma história envolvente em 4-5 parágrafos baseada neste vídeo do YouTube.

REGRAS CRÍTICAS:
- Primeiro parágrafo DEVE começar com um FATO CONCRETO extraído do vídeo (data, nome, lugar, número)
- NUNCA comece com "ninguém sabia", "o segredo" ou "a verdade escondida"
- Extraia detalhes específicos do título e descrição: datas, nomes, locais, estatísticas
- Conte a ORIGEM: como tudo começou, por que é importante
- Cada parágrafo deve avançar a história com um novo fato
- NADA de ganchos genéricos ou frases de enchimento

Vídeo:
{combined_content}

Escreva cada parágrafo em uma linha separada."""

        return self._make_text_api_call(system_prompt, user_prompt)

    def generate_script_with_prompts(self, subject: str) -> tuple:
        """Generate script paragraphs only.Image prompts will be generated later based on TTS duration."""
        log(f"Generating script paragraphs for: {subject}...")

        system_prompt = """You are a master storyteller for viral YouTube Shorts.
Output ONLY a JSON object with:
1.'paragraphs': Array of 7 strings (PT-BR).
CRITICAL: First paragraph MUST start with a SPECIFIC FACT (date, name, number, place).
NEVER use "ninguém sabia", "o segredo", or "a verdade" — these are vague.
Always lead with concrete details: dates, names, places, statistics.
Include origin stories: explain how it started and why it matters.
Keep each paragraph 2-3 sentences (~3 seconds audio each)."""

        user_prompt = f"Tell a story about: {subject}. Start with a specific concrete fact (date, name, number). Include origin and specific details."

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
        }

        # Add tools if web search is enabled
        if self.web_search and self.tools:
            payload["tools"] = self.tools

        try:
            response = requests.post(
                self.api_url, headers=headers, json=payload, timeout=API_TIMEOUT_TEXT
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            data = json.loads(content)
            return data["paragraphs"], []  # Return empty list for image prompts

        except Exception as e:
            log(f"Script generation failed: {e}", "ERROR")
            raise

    def generate_image_prompts_from_script(
        self, paragraphs: list, num_images: int
    ) -> list:
        """Generate image prompts based on script paragraphs and desired image count."""
        log(f"Generating {num_images} image prompts based on script...")

        script_text = " ".join(paragraphs)

        system_prompt = f"""
        Output ONLY a JSON object with:
        1.'image_prompts': Array of {num_images} DETAILED English prompts for an image generator.
           Prompts should be cinematic, realistic, 4k, and describe specific scenes
           matching the story.Each prompt should represent a scene that can be
           displayed for approximately 3 seconds.
        """

        user_prompt = (
            f"Create {num_images} detailed image prompts for this story: {script_text}"
        )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
        }

        try:
            response = requests.post(
                self.api_url, headers=headers, json=payload, timeout=API_TIMEOUT_TEXT
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            data = json.loads(content)
            return data["image_prompts"]

        except Exception as e:
            log(f"Image prompt generation failed: {e}", "ERROR")
            raise

    def _make_text_api_call(self, system_prompt: str, user_prompt: str) -> list:
        """Make API call and parse plain text response into paragraphs."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.7,
        }

        # Add tools if web search is enabled
        if self.web_search and self.tools:
            payload["tools"] = self.tools

        try:
            response = requests.post(
                self.api_url, headers=headers, json=payload, timeout=API_TIMEOUT_TEXT
            )
            response.raise_for_status()
            response_data = response.json()

            # Handle case where API returns a list instead of dict
            if isinstance(response_data, list):
                log("API returned a list, attempting to extract content...", "WARNING")
                # Try to extract content from list format
                if len(response_data) > 0 and isinstance(response_data[0], dict):
                    content = response_data[0].get("content", "") or response_data[
                        0
                    ].get("text", "")
                else:
                    content = str(response_data[0]) if response_data else ""
            else:
                # Normal dict response
                content = response_data["choices"][0]["message"]["content"]

            # Clean up the response
            content = (
                content.replace("```", "").replace("**", "").replace("*", "").strip()
            )

            # Split by lines and filter out empty lines
            lines = [line.strip() for line in content.split("\n") if line.strip()]

            # If no line breaks found, try to split by double newlines
            if len(lines) <= 1:
                lines = [para.strip() for para in content.split("\n\n") if para.strip()]

            # Still no good split? Try to split by periods
            if len(lines) <= 1:
                sentences = [s.strip() + "." for s in content.split(".") if s.strip()]
                # Group sentences into paragraphs
                lines = []
                current_para = ""
                for sentence in sentences[:15]:  # Limit to prevent too many paragraphs
                    current_para += sentence
                    if len(current_para) > 100:  # Rough paragraph length
                        lines.append(current_para)
                        current_para = ""
                if current_para:
                    lines.append(current_para)

            # Ensure we have 4-5 paragraphs
            if len(lines) < 4:
                # Pad with fallback content
                fallback = [
                    "Esta história começa com um fato que marcou época.",
                    "Os detalhes revelam como tudo aconteceu ao longo do tempo.",
                    "Cada etapa trouxe consequências que mudaram o rumo dos acontecimentos.",
                    "O desfecho mostra por que este tema continua relevante até hoje.",
                    "No final, fica uma lição que vale a pena conhecer.",
                ]
                lines.extend(fallback[len(lines) :])

            # Return exactly 5 paragraphs
            return lines[:5]

        except Exception as e:
            log(f"Script generation failed: {e}", "ERROR")
            # Return fallback script
            return [
                "Esta história começa com um fato que marcou época.",
                "Os detalhes revelam como tudo aconteceu ao longo do tempo.",
                "Cada etapa trouxe consequências que mudaram o rumo dos acontecimentos.",
                "O desfecho mostra por que este tema continua relevante até hoje.",
                "No final, fica uma lição que vale a pena conhecer.",
            ]
