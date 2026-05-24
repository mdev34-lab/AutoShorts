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

FALLBACK_PARAGRAPHS: list[str] = []


class ScriptGenerator:
    """Unified script generation using Pollinations API.

    When web_search is enabled, uses a two-step approach:
      Step 1 — generate verification search queries + title (draft is a byproduct)
      Step 2 — search the web, then generate the final script grounded in results
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
        """Generate script from subject. Two-step when web_search is enabled."""
        log("Generating script...")

        if not self.web_search or not self.searcher or not subject:
            return self._make_text_api_call(
                _SYSTEM_PROMPT_SINGLE,
                _user_prompt_single(subject, ""),
            )

        # Step 1: generate search queries + title
        log("Step 1: generating search queries...")
        draft_data = self._generate_draft(subject, num_paragraphs=5)
        self.generated_title = draft_data.get("title") or None
        queries = draft_data.get("queries") or []

        # Search using LLM-generated queries
        results = None
        if queries:
            log(f"Searching {len(queries)} LLM-generated queries...")
            results = self.searcher.search_with_queries(queries)
        else:
            log("No queries generated, falling back to subject-based search")
            results = self.searcher.search(subject)

        # Step 2: generate final script grounded in search results
        if results:
            context = self.searcher.format_context(results[:15])
            log("Step 2: generating script with search context...")
            script = self._make_text_api_call(
                _SYSTEM_PROMPT_SINGLE,
                _user_prompt_single(subject, context),
            )
            script = self._validate_paragraphs(script)
            script = self._ensure_paragraph_count(script, 5)
            if script:
                log("Script generated with web sources", "SUCCESS")
                return script
            log("Script generation with context failed or produced filler", "WARNING")
        else:
            log("No search results for grounding", "WARNING")

        log("Falling back to draft script (ungrounded)", "WARNING")
        draft = draft_data.get("draft") or []
        draft = self._validate_paragraphs(draft)
        draft = self._ensure_paragraph_count(draft, 5)
        if draft:
            return draft

        log("All script generation paths failed", "ERROR")
        return []

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
        """Generate script paragraphs. Two-step when web_search is enabled."""
        log(f"Generating script paragraphs for: {subject}...")

        if not self.web_search or not self.searcher or not subject:
            return self._generate_script_with_prompts_single(subject)

        # Step 1: generate search queries + title
        log("Step 1: generating search queries...")
        draft_data = self._generate_draft(subject, num_paragraphs=7)
        self.generated_title = draft_data.get("title") or None
        queries = draft_data.get("queries") or []

        # Search using LLM-generated queries
        results = None
        if queries:
            log(f"Searching {len(queries)} LLM-generated queries...")
            results = self.searcher.search_with_queries(queries)
        else:
            results = self.searcher.search(subject)

        # Step 2: generate final script grounded in search results
        if results:
            context = self.searcher.format_context(results[:15])
            log("Step 2: generating script with search context...")
            paragraphs = self._generate_script_with_context(subject, context)
            if paragraphs:
                paragraphs = self._validate_paragraphs(paragraphs)
                paragraphs = self._ensure_paragraph_count(paragraphs, 7)
                if paragraphs:
                    log("Script generated with web sources", "SUCCESS")
                    return paragraphs, []
            log("Script generation with context failed or produced filler", "WARNING")
        else:
            log("No search results for grounding", "WARNING")

        log("Falling back to draft script (ungrounded)", "WARNING")
        draft = draft_data.get("draft") or []
        draft = self._validate_paragraphs(draft)
        draft = self._ensure_paragraph_count(draft, 7)
        if draft:
            return draft, []

        log("All script generation paths failed", "ERROR")
        return [], []

    def generate_image_prompts_from_script(
        self, paragraphs: list, num_images: int
    ) -> list[dict]:
        """Generate paired image prompts (web query + AI prompt) based on script.

        Returns list of dicts: [{"web_query": "...", "ai_prompt": "..."}, ...]
        """
        log(f"Generating {num_images} paired image prompts based on script...")
        script_text = " ".join(paragraphs)
        system_prompt = f"""
        Output ONLY a JSON object with one key:
        'images': Array of {num_images} objects, each with:
          - 'web_query': short (3-6 word) search query for finding REAL photos on the web.
            Use simple keywords like "subject crowd", "subject stadium", "subject close up".
            NO descriptive adjectives, just concrete nouns and the subject.
          - 'ai_prompt': detailed English prompt for an AI image generator.
            Cinematic, dramatic lighting, ultra detailed, 4k photography style.
            Describe a specific scene matching the story.
        """
        user_prompt = f"Create {num_images} image pairs (web query + AI prompt) for this story: {script_text}"
        data = self._make_json_api_call(system_prompt, user_prompt)
        return data.get("images", [])

    # ── Two-pass helpers ─────────────────────────────────────────────────

    def _generate_draft(self, subject: str, num_paragraphs: int = 5) -> dict:
        """Pass 1: generate draft script + verification queries + title."""
        system_prompt = (
            f"You are a YouTube Shorts scriptwriter. Output ONLY valid JSON with these exact keys:\n"
            f'1. "draft": Array of {num_paragraphs} strings (PT-BR), each 2-3 sentences '
            f"\u2014 a first-draft script about \"{subject}\".\n"
            f"   - First paragraph MUST start with a specific concrete fact (date, name, number, place).\n"
            f"   - Every paragraph MUST contain a verifiable fact \u2014 no generalities, no filler.\n"
            f"   - CRITICAL: Do the math yourself. If you mention a date range, calculate the years correctly.\n"
            f"   - Include specific names, dates, statistics, locations \u2014 and VERIFY them in your head.\n"
            f"   - Tell an origin story: how it started, why it matters.\n"
            f"   - End with a factual conclusion, NOT 'fica uma li\u00e7\u00e3o' or similar generic phrases.\n"
            f'2. "queries": Array of 7-9 Portuguese web search queries to VERIFY '
            f"the factual claims in your draft.\n"
            f"   - At least 3 queries must be BROADER independent searches about the subject "
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
            log(
                f"Draft generated: {len(data['draft'])} paragraphs, {num_q} queries",
                "INFO",
            )
            return data
        except Exception as e:
            log(f"Draft generation failed: {e}", "ERROR")
            return {"draft": [], "queries": [subject], "title": ""}

    # ── API helpers ──────────────────────────────────────────────────────

    def _make_json_api_call(self, system_prompt: str, user_prompt: str) -> dict:
        """Make API call expecting JSON response. Retries once on empty content."""
        import time

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
        for attempt in range(2):
            try:
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    timeout=API_TIMEOUT_TEXT,
                )
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
                if content and content.strip():
                    return json.loads(content)
                log(f"API returned empty content (attempt {attempt + 1})", "WARNING")
            except (json.JSONDecodeError, KeyError, requests.RequestException) as e:
                log(
                    f"JSON API call failed (attempt {attempt + 1}): {e}",
                    "WARNING",
                )
            if attempt == 0:
                time.sleep(1)
        raise ValueError("JSON API call failed after 2 attempts")

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
                log(
                    f"Only {len(lines)} paragraphs generated, expected at least 4",
                    "WARNING",
                )

            return lines[:5]

        except Exception as e:
            log(f"Script generation failed: {e}", "ERROR")
            return []

    @staticmethod
    def _is_filler(paragraph: str) -> bool:
        """Check if a paragraph is generic filler that should be rejected."""
        filler_patterns = [
            "fica uma li\u00e7\u00e3o",
            "vale a pena conhecer",
            "li\u00e7\u00e3o que vale",
            "ningu\u00e9m sabia",
            "o segredo",
            "a verdade escondida",
            "voc\u00ea n\u00e3o vai acreditar",
            "poucos conhecem",
            "pouca gente sabe",
            "muita gente n\u00e3o sabe",
            "o que poucos sabem",
        ]
        lower = paragraph.lower()
        return any(p in lower for p in filler_patterns)

    @staticmethod
    def _validate_paragraphs(paragraphs: list) -> list:
        """Remove filler paragraphs and validate quality. Returns cleaned list or empty."""
        if not paragraphs:
            return []
        cleaned = [p for p in paragraphs if not ScriptGenerator._is_filler(p)]
        if len(cleaned) < 3:
            log(
                f"Validation: {len(paragraphs)} input, {len(cleaned)} after removing filler",
                "WARNING",
            )
        return cleaned

    @staticmethod
    def _ensure_paragraph_count(paragraphs: list, target: int) -> list:
        """Trim or validate paragraph count. Never pads with filler."""
        if len(paragraphs) >= target:
            return paragraphs[:target]
        if len(paragraphs) < 3:
            log(
                f"Only {len(paragraphs)} paragraphs, need at least 3 \u2014 returning empty",
                "ERROR",
            )
            return []
        return paragraphs

    # ── Single-pass path for images-only without web search ──────────────

    def _generate_script_with_context(
        self, subject: str, search_context: str
    ) -> list | None:
        """Generate paragraphs grounded in search context (JSON API call)."""
        system_prompt = (
            "You are a master storyteller for viral YouTube Shorts.\n"
            "Output ONLY a JSON object with:\n"
            "1.'paragraphs': Array of 7 strings (PT-BR).\n"
            "CRITICAL: First paragraph MUST start with a SPECIFIC FACT (date, name, number, place).\n"
            'NEVER use "ningu\u00e9m sabia", "o segredo", or "a verdade" \u2014 these are vague.\n'
            "Always lead with concrete details: dates, names, places, statistics.\n"
            "Every paragraph MUST contain a verifiable fact \u2014 no generalities, no filler.\n"
            "Include origin stories: explain how it started and why it matters.\n"
            "Keep each paragraph 2-3 sentences (~3 seconds audio each).\n"
            "End with a factual conclusion, NOT 'fica uma li\u00e7\u00e3o' or similar.\n"
            "Use the provided web sources as your primary source of facts. Verify every number against them."
        )
        user_prompt = (
            f"Tell a story about: {subject}. "
            f"Start with a specific concrete fact (date, name, number). "
            f"Include origin and specific details. "
            f"Double-check every date and number \u2014 calculate ranges correctly.\n\n"
            f"WEB SOURCES:\n{search_context}"
        )
        try:
            data = self._make_json_api_call(system_prompt, user_prompt)
            return data.get("paragraphs")
        except Exception as e:
            log(f"Script generation with context failed: {e}", "WARNING")
            return None

    def _generate_script_with_prompts_single(self, subject: str) -> tuple:
        """Original single-pass JSON generation (no web search)."""
        system_prompt = (
            "You are a master storyteller for viral YouTube Shorts.\n"
            "Output ONLY a JSON object with:\n"
            "1.'paragraphs': Array of 7 strings (PT-BR).\n"
            "CRITICAL: First paragraph MUST start with a SPECIFIC FACT (date, name, number, place).\n"
            'NEVER use "ningu\u00e9m sabia", "o segredo", or "a verdade" \u2014 these are vague.\n'
            "Every paragraph MUST contain a verifiable fact \u2014 no generalities, no filler.\n"
            "Always lead with concrete details: dates, names, places, statistics.\n"
            "Include origin stories: explain how it started and why it matters.\n"
            "Keep each paragraph 2-3 sentences (~3 seconds audio each).\n"
            'End with a factual conclusion, NOT "fica uma li\u00e7\u00e3o" or similar.\n'
            "Do the math yourself. If you mention a date range, calculate the years correctly."
        )
        user_prompt = (
            f"Tell a story about: {subject}. "
            f"Start with a specific concrete fact (date, name, number). "
            f"Include origin and specific details. "
            f"Double-check every date and number \u2014 calculate ranges correctly."
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
    "Cite specific data from them.\n"
    '11.End with a factual conclusion, NOT "fica uma li\u00e7\u00e3o" or similar generic phrases.\n'
    '12.NEVER use "no final fica uma li\u00e7\u00e3o" or "vale a pena conhecer" \u2014 these are filler.\n'
    "13.Do the math yourself. If you mention a date range or time period, calculate the years correctly."
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
        f"- Cada par\u00e1grafo DEVE conter um FATO VERIFIC\u00c1VEL \u2014 nada de generaliza\u00e7\u00f5es\n"
        f"- NADA de frases de enchimento\n"
        f"- TERMINE com uma conclus\u00e3o factual, N\u00c3O com 'fica uma li\u00e7\u00e3o'\n"
        f"- Fa\u00e7a a conta voc\u00ea mesmo: se mencionar um per\u00edodo, calcule os anos corretamente\n"
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
    "adding only well-known historical context.\n"
    '10.End with a factual conclusion, NOT "fica uma li\u00e7\u00e3o" or similar.\n'
    "11.Every paragraph must contain a verifiable fact \u2014 no generalities."
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
        "- Cada par\u00e1grafo DEVE conter um FATO VERIFIC\u00c1VEL \u2014 nada de generaliza\u00e7\u00f5es\n"
        "- NADA de ganchos gen\u00e9ricos ou frases de enchimento\n"
        "- TERMINE com uma conclus\u00e3o factual, N\u00c3O com 'fica uma li\u00e7\u00e3o'\n"
        "- Fa\u00e7a a conta voc\u00ea mesmo: se mencionar um per\u00edodo, calcule os anos corretamente\n\n"
        f"V\u00eddeo:\n{combined_content}\n\n"
        "Escreva cada par\u00e1grafo em uma linha separada."
    )
