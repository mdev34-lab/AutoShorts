import json

import requests  # type: ignore[import-untyped]

from .config import (
    API_KEY,
    API_TIMEOUT_TEXT,
    API_URL,
    MODEL_TEXT,
)
from .logging_system import log
from .web_search import WebSearcher

FALLBACK_PARAGRAPHS = [
    "Esta hist\u00f3ria come\u00e7a com um fato que marcou \u00e9poca.",
    "Os detalhes revelam como tudo aconteceu ao longo do tempo.",
    "Cada etapa trouxe consequ\u00eancias que mudaram o rumo dos acontecimentos.",
    "O desfecho mostra por que este tema continua relevante at\u00e9 hoje.",
    "No final, fica uma li\u00e7\u00e3o que vale a pena conhecer.",
]


class ScriptGenerator:
    """Unified script generation using Pollinations API.

    When web_search is enabled, uses a two-pass approach:
      Pass 1 — generate a draft script + verification search queries + title
      Pass 2 — search the web, then verify and fix factual errors in the draft
    """

    def __init__(self, web_search: bool = True):
        self.api_url = API_URL
        self.api_key = API_KEY
        self.model = MODEL_TEXT
        self.web_search = web_search
        self.searcher = WebSearcher() if web_search else None
        self.generated_title: str | None = None

    # ── Public API ──────────────────────────────────────────────────────

    def generate_script(self, subject: str) -> list:
        """Generate script from subject. Two-pass when web_search is enabled."""
        log("Generating script...")

        if not self.web_search or not self.searcher or not subject:
            return self._make_text_api_call(
                _SYSTEM_PROMPT_SINGLE,
                _user_prompt_single(subject, ""),
            )

        # Pass 1: draft + verification queries + title
        log("Pass 1: generating draft and verification queries...")
        draft_data = self._generate_draft(subject, num_paragraphs=5)
        self.generated_title = draft_data.get("title") or None
        draft = draft_data.get("draft") or []
        queries = draft_data.get("queries") or []

        if not draft:
            log("Draft generation returned empty, using fallback", "WARNING")
            return list(FALLBACK_PARAGRAPHS)

        # Search using LLM-generated queries
        results = None
        if queries:
            log(f"Searching {len(queries)} LLM-generated queries...")
            results = self.searcher.search_with_queries(queries)
        else:
            log("No queries generated, falling back to subject-based search")
            results = self.searcher.search(subject)

        # Pass 2: verify draft against search results
        if results:
            log(f"Pass 2: verifying draft against {len(results)} sources...")
            context = self.searcher.format_context(results)
            verified = self._verify_script(draft, context)
            if verified:
                log("Script verified and corrected", "SUCCESS")
                return self._ensure_paragraph_count(verified, 5)
            log("Verification failed, returning draft as-is", "WARNING")
        else:
            log("No search results, returning unverified draft", "WARNING")

        return self._ensure_paragraph_count(draft, 5)

    def generate_script_from_metadata(self, title: str, description: str) -> list:
        """Generate script from YouTube video title and description."""
        log("Generating script from video metadata...")
        desc = description[:1000] if description else ""
        combined_content = f"Title: {title}\n\nDescription: {desc}"
        return self._make_text_api_call(
            _SYSTEM_PROMPT_METADATA,
            _user_prompt_metadata(combined_content),
        )

    def generate_script_with_prompts(self, subject: str) -> tuple:
        """Generate script paragraphs. Two-pass when web_search is enabled."""
        log(f"Generating script paragraphs for: {subject}...")

        if not self.web_search or not self.searcher or not subject:
            return self._generate_script_with_prompts_single(subject)

        # Pass 1: draft + verification queries + title
        log("Pass 1: generating draft and verification queries...")
        draft_data = self._generate_draft(subject, num_paragraphs=7)
        self.generated_title = draft_data.get("title") or None
        draft = draft_data.get("draft") or []
        queries = draft_data.get("queries") or []

        if not draft:
            log("Draft generation returned empty, using fallback", "WARNING")
            return list(FALLBACK_PARAGRAPHS), []

        # Search using LLM-generated queries
        results = None
        if queries:
            log(f"Searching {len(queries)} LLM-generated queries...")
            results = self.searcher.search_with_queries(queries)
        else:
            results = self.searcher.search(subject)

        # Pass 2: verify against search results
        if results:
            log(f"Pass 2: verifying draft against {len(results)} sources...")
            context = self.searcher.format_context(results)
            verified = self._verify_script(draft, context)
            if verified:
                log("Script verified and corrected", "SUCCESS")
                return self._ensure_paragraph_count(verified, 7), []
            log("Verification failed, returning draft as-is", "WARNING")
        else:
            log("No search results, returning unverified draft", "WARNING")

        return self._ensure_paragraph_count(draft, 7), []

    def generate_image_prompts_from_script(
        self, paragraphs: list, num_images: int
    ) -> list:
        """Generate image prompts based on script paragraphs."""
        log(f"Generating {num_images} image prompts based on script...")
        script_text = " ".join(paragraphs)
        system_prompt = f"""
        Output ONLY a JSON object with:
        1.'image_prompts': Array of {num_images} DETAILED English prompts for an image generator.
           Prompts should be cinematic, realistic, 4k, and describe specific scenes
           matching the story.Each prompt should represent a scene that can be
           displayed for approximately 3 seconds.
        """
        user_prompt = f"Create {num_images} detailed image prompts for this story: {script_text}"
        data = self._make_json_api_call(system_prompt, user_prompt)
        return data.get("image_prompts", [])

    # ── Two-pass helpers ─────────────────────────────────────────────────

    def _generate_draft(self, subject: str, num_paragraphs: int = 5) -> dict:
        """Pass 1: generate draft script + verification queries + title."""
        system_prompt = (
            f"You are a YouTube Shorts scriptwriter. Output ONLY valid JSON with these exact keys:\n"
            f'1. "draft": Array of {num_paragraphs} strings (PT-BR), each 2-3 sentences '
            f'\u2014 a first-draft script about "{subject}".\n'
            f"   - First paragraph MUST start with a specific concrete fact (date, name, number, place).\n"
            f"   - Include specific names, dates, statistics, locations.\n"
            f"   - Tell an origin story: how it started, why it matters.\n"
            f"   - This is a DRAFT \u2014 it may contain errors. Do NOT fact-check yourself.\n"
             f'2. "queries": Array of 5-7 Portuguese web search queries to VERIFY '
             f"the factual claims in your draft.\n"
             f"   - At least 2 queries must be BROADER independent searches about the subject "
             f"(e.g. 'Botafogo principais rivais' instead of 'hist\u00f3ria do cl\u00e1ssico X').\n"
             f"   - Each specific query should target one or more claims from the draft.\n"
             f"   - Be specific: include names, dates, unique terms.\n"
             f"   - Cover all major factual claims (dates, names, places, statistics, origins).\n"
            f'3. "title": string, max 60 chars (PT-BR) \u2014 catchy YouTube Shorts title about {subject}.'
        )
        user_prompt = (
            f"Write a first-draft script about {subject} in {num_paragraphs} paragraphs "
            f"(PT-BR), generate 4-6 Portuguese search queries to verify its facts, "
            f"and suggest a catchy title."
        )
        try:
            data = self._make_json_api_call(system_prompt, user_prompt)
            if "draft" not in data:
                log("Draft response missing 'draft' key", "ERROR")
                return {"draft": [], "queries": [subject], "title": ""}
            num_q = len(data.get("queries") or [])
            log(f"Draft generated: {len(data['draft'])} paragraphs, {num_q} queries", "INFO")
            return data
        except Exception as e:
            log(f"Draft generation failed: {e}", "ERROR")
            return {"draft": [], "queries": [subject], "title": ""}

    def _verify_script(self, draft: list, search_context: str) -> list | None:
        """Pass 2: verify and fix draft using web search results."""
        system_prompt = (
            "You are a fact-checking editor for YouTube Shorts.\n"
            "Fix factual errors in a draft script using the provided web sources.\n\n"
            "CRITICAL \u2014 Be skeptical, do not just confirm terms exist:\n"
            '   - Verify RELATIONSHIPS, not just names. E.g., if the draft says '
            '"the classic is X", check whether the SUBJECT is part of X.\n'
            "   - A web source mentioning a term does NOT mean the draft's claim about it is correct.\n"
            "   - Broader context queries in the sources may override specific claim queries.\n\n"
            "Output ONLY a JSON object with:\n"
            '1. "paragraphs": Array of corrected strings (PT-BR).\n'
            "   - Fix ANY factual errors based on the web sources.\n"
            "   - Keep the engaging storytelling tone and the same number of paragraphs.\n"
            "   - First paragraph MUST start with a specific VERIFIED fact.\n"
            "   - Each paragraph 2-3 sentences.\n"
            "   - If a fact in the draft contradicts a web source, ALWAYS trust the web source.\n"
            "   - Remove or correct any claim that cannot be verified by the sources."
        )
        user_prompt = (
            f"Fix this draft using the web sources below.\n\n"
            f"DRAFT:\n{json.dumps(draft, ensure_ascii=False, indent=2)}\n\n"
            f"WEB SOURCES:\n{search_context}\n\n"
            f"Output ONLY a JSON object with 'paragraphs'."
        )
        try:
            data = self._make_json_api_call(system_prompt, user_prompt)
            paragraphs = data.get("paragraphs")
            if paragraphs and len(paragraphs) >= 3:
                log(f"Verification complete: {len(paragraphs)} paragraphs", "SUCCESS")
                return paragraphs
            log(f"Verification returned only {len(paragraphs) if paragraphs else 0} paragraphs", "WARNING")
            return None
        except Exception as e:
            log(f"Verification failed: {e}", "WARNING")
            return None

    # ── API helpers ──────────────────────────────────────────────────────

    def _make_json_api_call(self, system_prompt: str, user_prompt: str) -> dict:
        """Make API call expecting JSON response."""
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
            "temperature": 0.7,
        }
        response = requests.post(
            self.api_url, headers=headers, json=payload, timeout=API_TIMEOUT_TEXT
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        return json.loads(content)

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

        try:
            response = requests.post(
                self.api_url, headers=headers, json=payload, timeout=API_TIMEOUT_TEXT
            )
            response.raise_for_status()
            response_data = response.json()

            if isinstance(response_data, list):
                log("API returned a list, attempting to extract content...", "WARNING")
                if len(response_data) > 0 and isinstance(response_data[0], dict):
                    content = response_data[0].get("content", "") or response_data[
                        0
                    ].get("text", "")
                else:
                    content = str(response_data[0]) if response_data else ""
            else:
                content = response_data["choices"][0]["message"]["content"]

            content = (
                content.replace("```", "").replace("**", "").replace("*", "").strip()
            )

            lines = [line.strip() for line in content.split("\n") if line.strip()]

            if len(lines) <= 1:
                lines = [para.strip() for para in content.split("\n\n") if para.strip()]

            if len(lines) <= 1:
                sentences = [s.strip() + "." for s in content.split(".") if s.strip()]
                lines = []
                current_para = ""
                for sentence in sentences[:15]:
                    current_para += sentence
                    if len(current_para) > 100:
                        lines.append(current_para)
                        current_para = ""
                if current_para:
                    lines.append(current_para)

            if len(lines) < 4:
                lines.extend(FALLBACK_PARAGRAPHS[len(lines):])

            return lines[:5]

        except Exception as e:
            log(f"Script generation failed: {e}", "ERROR")
            return list(FALLBACK_PARAGRAPHS)

    @staticmethod
    def _ensure_paragraph_count(paragraphs: list, target: int) -> list:
        """Pad or trim paragraphs to target count."""
        if len(paragraphs) >= target:
            return paragraphs[:target]
        return paragraphs + FALLBACK_PARAGRAPHS[len(paragraphs):target]

    # ── Single-pass path for images-only without web search ──────────────

    def _generate_script_with_prompts_single(self, subject: str) -> tuple:
        """Original single-pass JSON generation (no web search)."""
        system_prompt = (
            "You are a master storyteller for viral YouTube Shorts.\n"
            "Output ONLY a JSON object with:\n"
            "1.'paragraphs': Array of 7 strings (PT-BR).\n"
            "CRITICAL: First paragraph MUST start with a SPECIFIC FACT (date, name, number, place).\n"
            "NEVER use \"ningu\u00e9m sabia\", \"o segredo\", or \"a verdade\" \u2014 these are vague.\n"
            "Always lead with concrete details: dates, names, places, statistics.\n"
            "Include origin stories: explain how it started and why it matters.\n"
            "Keep each paragraph 2-3 sentences (~3 seconds audio each)."
        )
        user_prompt = (
            f"Tell a story about: {subject}. "
            f"Start with a specific concrete fact (date, name, number). "
            f"Include origin and specific details."
        )
        try:
            data = self._make_json_api_call(system_prompt, user_prompt)
            return data.get("paragraphs", []), data.get("image_prompts", [])
        except Exception as e:
            log(f"Script generation failed: {e}", "ERROR")
            raise


# ── Prompt templates for single-pass text path ──────────────────────────

_SYSTEM_PROMPT_SINGLE = (
    "You are a master storyteller for viral YouTube Shorts.\n"
    "CRITICAL RETENTION RULES:\n"
    "1.Write in Brazilian Portuguese (PT-BR).\n"
    "2.First paragraph MUST start with a SPECIFIC FACT (date, number, name, place) "
    "\u2014 not a generic teaser.\n"
    '3.NEVER start with "ningu\u00e9m sabia", "o segredo", "a verdade escondida" '
    'or "voc\u00ea n\u00e3o vai acreditar" \u2014 these are vague and weak.\n'
    '4.Always lead with concrete, specific details: "Em 1914...", '
    '"Tudo come\u00e7ou quando...", "O placar foi 8 a 0..."\n'
    "5.Include origin stories \u2014 explain HOW something started or WHY it matters, "
    "not just THAT it exists.\n"
    "6.Each paragraph 2-3 sentences for pacing (each ~3 seconds of audio).\n"
    "7.Write exactly 4-5 paragraphs.\n"
    "8.NO markdown formatting, NO JSON, just plain text paragraphs.\n"
    "9.Every paragraph must advance the story with a new specific fact \u2014 no filler.\n"
    "10.USE the provided web sources as your primary source of facts. "
    "Cite specific data from them."
)


def _user_prompt_single(subject: str, search_context: str) -> str:
    return (
        f'Crie uma hist\u00f3ria envolvente em 4-5 par\u00e1grafos sobre "{subject}".\n\n'
        f"{search_context}\n\n"
        f"REGRAS CR\u00cdTICAS:\n"
        f"- Primeiro par\u00e1grafo DEVE come\u00e7ar com um FATO CONCRETO "
        f"(data, n\u00famero, nome, lugar) \u2014 N\u00c3O use ganchos gen\u00e9ricos\n"
        f'- NUNCA comece com "ningu\u00e9m sabia", "o segredo", '
        f'"a verdade escondida" \u2014 isso \u00e9 vago e fraco\n'
        f"- Inclua nomes, datas, lugares e n\u00fameros espec\u00edficos sempre que poss\u00edvel\n"
        f"- Conte a ORIGEM: como tudo come\u00e7ou, por que existe\n"
        f"- Cada par\u00e1grafo deve avan\u00e7ar a hist\u00f3ria com um novo fato concreto\n"
        f"- NADA de frases de enchimento\n"
        f"- Use as FONTES DA WEB fornecidas como base para sua hist\u00f3ria\n\n"
        f"Escreva cada par\u00e1grafo em uma linha separada."
    )


_SYSTEM_PROMPT_METADATA = (
    "You are a master storyteller for viral YouTube Shorts.\n"
    "CRITICAL RETENTION RULES:\n"
    "1.Write in Brazilian Portuguese (PT-BR).\n"
    "2.First paragraph MUST start with a SPECIFIC FACT from the video "
    "(date, number, name, place).\n"
    '3.NEVER start with "ningu\u00e9m sabia", "o segredo", '
    'or "a verdade escondida" \u2014 these are vague and weak.\n'
    "4.Extract concrete details from the video metadata: "
    "dates, names, places, statistics, historical context.\n"
    "5.Include origin stories \u2014 explain HOW something started, "
    "not just THAT it happened.\n"
    "6.Each paragraph 2-3 sentences for pacing (each ~3 seconds of audio).\n"
    "7.Write exactly 4-5 paragraphs.\n"
    "8.NO markdown formatting, NO JSON, just plain text paragraphs.\n"
    "9.Base your story entirely on the video's content, "
    "adding only well-known historical context."
)


def _user_prompt_metadata(combined_content: str) -> str:
    return (
        "Crie uma hist\u00f3ria envolvente em 4-5 par\u00e1grafos baseada "
        "neste v\u00eddeo do YouTube.\n\n"
        "REGRAS CR\u00cdTICAS:\n"
        "- Primeiro par\u00e1grafo DEVE come\u00e7ar com um FATO CONCRETO "
        "extra\u00eddo do v\u00eddeo (data, nome, lugar, n\u00famero)\n"
        '- NUNCA comece com "ningu\u00e9m sabia", "o segredo" '
        'ou "a verdade escondida"\n'
        "- Extraia detalhes espec\u00edficos do t\u00edtulo e descri\u00e7\u00e3o: "
        "datas, nomes, locais, estat\u00edsticas\n"
        "- Conte a ORIGEM: como tudo come\u00e7ou, por que \u00e9 importante\n"
        "- Cada par\u00e1grafo deve avan\u00e7ar a hist\u00f3ria com um novo fato\n"
        "- NADA de ganchos gen\u00e9ricos ou frases de enchimento\n\n"
        f"V\u00eddeo:\n{combined_content}\n\n"
        "Escreva cada par\u00e1grafo em uma linha separada."
    )
