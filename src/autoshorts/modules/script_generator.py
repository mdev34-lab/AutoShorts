import json
import re
import time

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

    def _tone_instructions(self) -> str:
        return (
            "TONE: Curiosity-driven, narrative, engaging. "
            "Write like a storyteller uncovering a fascinating truth \u2014 "
            "never like Wikipedia or a corporate press release.\n"
            "FIRST SENTENCE: Drop the viewer right into the action \u2014 "
            "the goal, the controversy, the fact itself. "
            "NO: 'Prepare-se', 'Voc\u00ea sabia', rhetorical questions. "
            "Never waste the first 2 seconds on setup.\n"
            "STRUCTURE: Hook (the fact itself) \u2192 Context \u2192 Revelation \u2192 Strong conclusion\n"
            "FORBIDDEN: Legal names (Ltda, S.A.), addresses, city/state abbreviations. "
            "NO corporate language.\n"
            "FORBIDDEN: Hyperboles, exaggerated claims, 'designed by a god', "
            "'you won't believe', 'shocking truth' \u2014 these sound fake.\n"
        )
        return (
            "TONE: Curiosity-driven, narrative, engaging. "
            "Write like a storyteller uncovering a fascinating truth \u2014 "
            "never like Wikipedia or a corporate press release.\n"
            "FIRST SENTENCE: Drop the viewer right into the action \u2014 "
            "the goal, the controversy, the fact itself. "
            "NO: 'Prepare-se', 'Voc\u00ea sabia', 'Uma pergunta', rhetorical questions. "
            "YES: 'O Corinthians tomou uma virada hist\u00f3rica...'\n"
            "STRUCTURE: Hook (the fact itself) \u2192 Context \u2192 Revelation \u2192 Strong conclusion\n"
            "FORBIDDEN: Legal names (Ltda, S.A.), addresses, city/state abbreviations. "
            "NO corporate language.\n"
            "FORBIDDEN: Hyperboles, exaggerated claims, 'designed by a god', "
            "'you won't believe', 'shocking truth' \u2014 these sound fake.\n"
        )

    # ── Public API ──────────────────────────────────────────────────────

    def generate_script(self, subject: str) -> list:
        """Generate script from subject. Searches web first, then generates grounded script."""
        log("Generating script...")

        tone_block = self._tone_instructions()

        if not self.web_search or not self.searcher or not subject:
            return self._make_text_api_call(
                tone_block + _SYSTEM_PROMPT_SINGLE,
                _user_prompt_single(subject, ""),
            )

        # Step 1: generate independent search queries (NOT from draft — avoids circular hallucination)
        log("Step 1: generating independent search queries...")
        queries = self._generate_search_queries(subject)

        # Step 2: search the web with neutral queries
        results = self.searcher.search_with_queries(queries)

        # Step 3: generate script grounded in search results
        if results:
            context = self.searcher.format_context(results[:15])
            log("Step 2: generating script with search context...")
            script = self._make_text_api_call(
                tone_block + _SYSTEM_PROMPT_SINGLE,
                _user_prompt_single(subject, context),
            )
            cleaned = self._validate_paragraphs(script)
            script = self._ensure_paragraph_count(cleaned, 5)
            if script:
                log("Script generated with web sources", "SUCCESS")
                # Step 4: post-generation fact verification
                script = self._verify_factual_claims(script, subject)
                self.generated_title = self._generate_title_from_script(script, subject)
                log("Script verified", "SUCCESS")
                return script

            # Try to repair instead of full regeneration
            repair = self._repair_paragraphs(cleaned, subject, 5)
            if repair and len(repair) >= 3:
                log("Script repaired after validation", "SUCCESS")
                script = repair
                script = self._verify_factual_claims(script, subject)
                self.generated_title = self._generate_title_from_script(script, subject)
                log("Script verified", "SUCCESS")
                return script

            log("Script generation with context failed or produced filler", "WARNING")
        else:
            log("No search results for grounding", "WARNING")

        # Fallback: generate draft for title + fallback content
        log("Generating draft as fallback...")
        draft_data = self._generate_draft(subject, num_paragraphs=5)
        self.generated_title = draft_data.get("title") or None
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
        tone_block = self._tone_instructions()
        return self._make_text_api_call(
            tone_block + _SYSTEM_PROMPT_METADATA,
            _user_prompt_metadata(combined_content),
        )

    def generate_script_with_prompts(self, subject: str) -> tuple:
        """Generate script paragraphs. Searches web first, then generates grounded script."""
        log(f"Generating script paragraphs for: {subject}...")

        if not self.web_search or not self.searcher or not subject:
            return self._generate_script_with_prompts_single(subject)

        # Step 1: generate independent search queries (NOT from draft)
        log("Step 1: generating independent search queries...")
        queries = self._generate_search_queries(subject)

        # Step 2: search the web with neutral queries
        results = self.searcher.search_with_queries(queries)

        # Step 3: generate script grounded in search results
        if results:
            context = self.searcher.format_context(results[:15])
            log("Step 2: generating script with search context...")
            paragraphs = self._generate_script_with_context(subject, context)
            if paragraphs:
                cleaned = self._validate_paragraphs(paragraphs)
                paragraphs = self._ensure_paragraph_count(cleaned, 7)
                if paragraphs:
                    log("Script generated with web sources", "SUCCESS")
                    paragraphs = self._verify_factual_claims(paragraphs, subject)
                    self.generated_title = self._generate_title_from_script(paragraphs, subject)
                    log("Script verified", "SUCCESS")
                    return paragraphs, []

                # Try to repair instead of full regeneration
                repair = self._repair_paragraphs(cleaned, subject, 7)
                if repair and len(repair) >= 3:
                    log("Script repaired after validation", "SUCCESS")
                    paragraphs = self._verify_factual_claims(repair, subject)
                    self.generated_title = self._generate_title_from_script(paragraphs, subject)
                    return paragraphs, []

            log("Script generation with context failed or produced filler", "WARNING")
        else:
            log("No search results for grounding", "WARNING")

        # Fallback to draft
        log("Generating draft as fallback...")
        draft_data = self._generate_draft(subject, num_paragraphs=7)
        self.generated_title = draft_data.get("title") or None
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
          - 'web_query': short (3-8 word) search query for finding REAL photos on the web.
            CRITICAL: Include context qualifiers like year, league/country, team name, event name.
            NEVER use a generic descriptor alone (e.g. "jogador comemorando") without the team/league context.
            Example: "Corinthians Neo Quimica Arena torcida 2024" instead of "stadium crowd".
            Use concrete nouns and the specific subject from the story.
            NO descriptive adjectives, NO filler words.
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
        tone_block = self._tone_instructions()
        system_prompt = (
            f"You are a YouTube Shorts scriptwriter. Output ONLY valid JSON with these exact keys:\n"
            f'1. "draft": Array of {num_paragraphs} strings (PT-BR), each 1-2 short sentences '
            f"\u2014 a first-draft script about \"{subject}\".\n"
            f"   {tone_block}"
            f"   - Every paragraph MUST contain a verifiable fact \u2014 no generalities, no filler.\n"
            f"   - CRITICAL: Do the math yourself. If you mention a date range, calculate the years correctly.\n"
            f"   - End with a strong conclusion, NOT 'fica uma li\u00e7\u00e3o' or similar generic phrases.\n"
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

    # ── Independent query generation (breaks circular hallucination) ─────

    def _generate_search_queries(self, subject: str) -> list[str]:
        """Generate neutral, independent search queries — NOT derived from draft content."""
        system_prompt = (
            "You are a research assistant. Output ONLY valid JSON with one key:\n"
            "'queries': Array of 6-8 specific web search queries in Portuguese.\n"
            "Your goal: find ACCURATE factual data about a topic.\n"
            "Each query must target a different angle: origins, dates, key events, statistics, people.\n"
            "Be specific: include names, years, locations.\n"
            "These queries will be used to fact-check, so prioritize queries that return concrete data.\n"
            "NEVER include the topic name alone as a query \u2014 always add qualifiers like year, event, or location."
        )
        user_prompt = (
            f"Generate search queries to find accurate factual information about: {subject}"
        )
        try:
            data = self._make_json_api_call(system_prompt, user_prompt)
            queries = data.get("queries") or []
            log(f"Generated {len(queries)} independent search queries", "SUCCESS")
            return queries
        except Exception as e:
            log(f"Search query generation failed: {e}", "WARNING")
            return [subject]

    # ── Post-generation fact verification ────────────────────────────────

    def _verify_factual_claims(self, paragraphs: list, subject: str) -> list:
        """Cross-check dates, scores, tabus, and numbers in script against web search results."""
        script_text = " ".join(paragraphs)
        verification_queries: list[str] = []

        # 1. Extract years
        years = set(re.findall(r"\b(1[4-9]\d{2}|20[0-2]\d)\b", script_text))
        for y in sorted(years):
            verification_queries.append(f"{subject} {y}")

        # 2. Extract score/result patterns: "3 a 2", "3x2", "por 3 a 2", "3-2"
        score_matches = re.findall(
            r"(\d+)\s*(?:[a\u00e0x-]\s*|a\s+|venceu por\s+|por\s+)(\d+)",
            script_text, re.IGNORECASE
        )
        for s1, s2 in score_matches:
            for sep in (" a ", "x"):
                q = f"{subject} {s1}{sep}{s2}"
                if q not in verification_queries:
                    verification_queries.append(q)

        # 3. Detect tabu/streak claims
        if re.search(
            r"(?:n\u00e3o\s+\w+\s+(?:vence|ganha|perde|supera)|tabu|"
            r"sem\s+\w+\s+(?:vence|ganha|perde|supera))",
            script_text, re.IGNORECASE
        ):
            tabu_q = f"{subject} tabu hist\u00f3rico"
            if tabu_q not in verification_queries:
                verification_queries.append(tabu_q)

        # 4. Extract "em [month] de [year]" / "desde [month] de [year]" contexts
        context_years = re.findall(
            r"(?:em|desde|no|na)\s+\w+\s+de\s+(\d{4})",
            script_text, re.IGNORECASE
        )
        for y in context_years:
            q = f"{subject} {y}"
            if q not in verification_queries:
                verification_queries.append(q)

        verification_queries.append(
            f"{subject} hist\u00f3rico funda\u00e7\u00e3o dados"
        )

        if not verification_queries:
            return paragraphs

        log(
            f"Verifying claims with {len(verification_queries)} targeted queries",
            "INFO",
        )

        if not self.searcher:
            log("Fact verification skipped: searcher not available", "WARNING")
            return paragraphs
        results = self.searcher.search_with_queries(list(dict.fromkeys(verification_queries)))
        if not results:
            log("Fact verification: no web sources found", "WARNING")
            return paragraphs

        context = self.searcher.format_context(results[:10])
        system_prompt = (
            "You are a strict fact-checker. Output ONLY valid JSON with exactly these keys:\n"
            '1. "verified": boolean \u2014 true if ALL claims match the web sources\n'
            '2. "corrections": array of objects with "claim" and "correction" strings '
            "\u2014 empty if verified is true\n"
            '3. "paragraphs": array of strings (PT-BR) \u2014 corrected script paragraphs, '
            "or the original if no changes needed\n\n"
            "CRITICAL rules:\n"
            "- Compare EVERY date, number, name, and place against the web sources.\n"
            "- If a source contradicts a script claim, the SOURCE wins. "
            "NEVER leave a hallucination uncorrected.\n"
            "- If NO source confirms a specific claim (score, streak, percentage, event), "
            "consider it UNVERIFIED and remove or rephrase it as uncertain.\n"
            "- Pay attention to chronology: if sources mention an event ended or a record was broken in year X, "
            "do NOT let the script claim it still holds in a later year."
        )
        user_prompt = (
            f"Subject: {subject}\n\n"
            f"SCRIPT:\n{script_text}\n\n"
            f"WEB SOURCES:\n{context}\n\n"
            "Cross-check every date, number, score, name, and factual claim. "
            "Output corrected paragraphs."
        )
        target_count = len(paragraphs)
        prev_count = 0
        for attempt in range(2):
            try:
                prompt = user_prompt
                if attempt == 1 and prev_count != target_count:
                    prompt += (
                        f"\n\nCORREÇÃO: Na tentativa anterior você retornou "
                        f"{prev_count} parágrafos, "
                        f"mas o script original tem {target_count}. "
                        f"Retorne EXATAMENTE {target_count} parágrafos. "
                        f"Não mescle, não remova, não junte parágrafos. "
                        f"Apenas corrija erros factuais mantendo a estrutura original."
                    )
                data = self._make_json_api_call(system_prompt, prompt)
                corrected = data.get("paragraphs") or []
                prev_count = len(corrected)
                corrections = data.get("corrections") or []
                is_verified = data.get("verified", False)
                if corrections:
                    log(
                        f"Fact verification: {len(corrections)} corrections applied",
                        "WARNING",
                    )
                    for c in corrections:
                        log(
                            f"  '{c.get('claim', '?')}' -> '{c.get('correction', '?')}'",
                            "INFO",
                        )
                    corrected = self._validate_paragraphs(corrected)
                    corrected = self._ensure_paragraph_count(corrected, target_count)
                elif is_verified:
                    log("Fact verification: all claims match sources", "SUCCESS")
                if corrected and len(corrected) >= 3:
                    return corrected
                log(
                    f"Verification returned {len(corrected)} paragraphs, need >= 3, retrying...",
                    "WARNING",
                )
            except Exception as e:
                log(f"Fact verification attempt {attempt + 1} failed: {e}", "WARNING")
                corrected = []
        if paragraphs and len(paragraphs) >= 3:
            return paragraphs
        return paragraphs

    # ── Title generation ─────────────────────────────────────────────────

    @staticmethod
    def _validate_title(title: str) -> tuple[bool, list[str]]:
        issues: list[str] = []
        hashtags = re.findall(r"#\w+", title)
        if not hashtags:
            issues.append("n\u00e3o possui hashtags")
        elif len(hashtags) < 3:
            issues.append(f"tem apenas {len(hashtags)} hashtags (m\u00ednimo 3)")
        if len(title) > 100:
            issues.append(f"tem {len(title)} caracteres (m\u00e1ximo 100)")
        return len(issues) == 0, issues

    @staticmethod
    def _fix_hashtags_case(title: str) -> str:
        return re.sub(r"#(\w+)", lambda m: f"#{m.group(1).lower()}", title)

    def _generate_title_from_script(
        self, paragraphs: list, subject: str
    ) -> str | None:
        """Generate a YouTube Shorts title from the final script."""
        script_text = " ".join(paragraphs)[:500]
        system_prompt = (
            "Output ONLY a JSON object with one key: 'title'.\n"
            "Title must be in PT-BR, max 100 characters, YouTube Shorts title.\n"
            "UPPERCASE RULES:\n"
            "- Use UPPERCASE for key words or short sections of the title text itself.\n"
            "- Example: 'A VERDADE sobre o Caso Girabank' (title before tags).\n"
            "- Do NOT put tags in uppercase.\n\n"
            "TAGS RULES:\n"
            "- Append EXACTLY 3-4 lowercase tags at the end, no spaces between words.\n"
            "- Tags MUST be SPECIFIC to the video subject, not generic categories.\n"
            "- Example tags for Mario: '#supermario #nintendo #galaxy'\n"
            "- Example tags for Carlinhos Maia: '#carlinhosmaia #girabank'\n"
            "- NEVER tag unrelated topics like #futebol for a movie.\n\n"
            "FULL EXAMPLE:\n"
            "'A VERDADE sobre o Caso Girabank #carlinhosmaia #girabank'"
        )
        user_prompt = f"Crie um t\u00edtulo PT-BR para este roteiro sobre {subject}, com se\u00e7\u00f5es em UPPERCASE e tags SPECIFICAS em lowercase: {script_text}"
        for attempt in range(2):
            try:
                prompt = user_prompt
                if attempt == 1:
                    prompt += (
                        "\n\nCORRE\u00c7\u00c3O: Na tentativa anterior o t\u00edtulo tinha problemas. "
                        "Siga as regras: max 100 chars, 3-4 hashtags em lowercase, "
                        "nada de hashtags gen\u00e9ricas."
                    )
                data = self._make_json_api_call(system_prompt, prompt)
                title = data.get("title") or ""
                title = self._fix_hashtags_case(title)
                ok, issues = self._validate_title(title)
                if ok:
                    return title
                log(f"Title validation: {', '.join(issues)}, retrying...", "WARNING")
            except Exception:
                return None
        return None

    def _repair_paragraphs(self, good: list, subject: str, target: int) -> list:
        """Extend existing good paragraphs to reach target count instead of regenerating everything."""
        if len(good) >= target or not good:
            return good

        needed = target - len(good)
        good_text = "\n".join(good)

        system_prompt = (
            f"Output ONLY a JSON object with one key:\n"
            f"'paragraphs': Array of {needed} strings (PT-BR), each 1-2 short sentences.\n"
            "Extend an existing script. Match the style, tone, and factual density of the existing paragraphs.\n"
            "Each paragraph MUST contain a verifiable fact. No filler, no generalities, no conclusions.\n"
            "Write paragraphs that would fit naturally BETWEEN the existing ones or after them.\n"
        )
        user_prompt = (
            f"Topic: {subject}\n\n"
            f"EXISTING PARAGRAPHS:\n{good_text}\n\n"
            f"Write {needed} more paragraphs (PT-BR) that extend this story. "
            "Do NOT repeat existing content."
        )
        try:
            data = self._make_json_api_call(system_prompt, user_prompt)
            new_p = data.get("paragraphs") or []
            log(
                f"Repair attempt: API returned {len(new_p)} paragraphs, "
                f"had {len(good)} good",
                "INFO",
            )
            combined = good + new_p
            before = len(combined)
            combined = self._validate_paragraphs(combined)
            after_validation = len(combined)
            if before != after_validation:
                log(
                    f"Repair validation: {before} -> {after_validation} "
                    f"({before - after_validation} removed as filler)",
                    "INFO",
                )
            combined = self._ensure_paragraph_count(combined, target)
            if combined:
                log(f"Repair success: {len(good)} -> {len(combined)} paragraphs", "SUCCESS")
                return combined
            log(
                f"Repair failed: only {after_validation} good paragraphs "
                f"after validation, needed {target}",
                "WARNING",
            )
            return []
        except Exception as e:
            log(f"Repair LLM call failed: {e}", "WARNING")
            return []

    # ── API helpers ──────────────────────────────────────────────────────

    def _make_json_api_call(self, system_prompt: str, user_prompt: str) -> dict:
        """Make API call expecting JSON response. Retries once on empty content."""

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
                    result = json.loads(content)
                    if not isinstance(result, dict):
                        log(
                            f"API returned {type(result).__name__} instead of dict",
                            "WARNING",
                        )
                        return {}
                    return result
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
        """Check if a paragraph is generic filler or corporate language that should be rejected."""
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
            "ltda",
            "s.a.",
            "institui\u00e7\u00e3o de pagamento",
            "pessoa jur\u00eddica",
        ]
        lower = paragraph.lower()
        return any(p in lower for p in filler_patterns)

    @staticmethod
    def _validate_paragraphs(paragraphs: list) -> list:
        """Remove filler paragraphs and validate quality. Returns cleaned list or empty."""
        if not paragraphs:
            return []
        cleaned = [p for p in paragraphs if isinstance(p, str) and not ScriptGenerator._is_filler(p)]
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
        tone_block = self._tone_instructions()
        system_prompt = (
            "You are a master storyteller for viral YouTube Shorts.\n"
            "Output ONLY a JSON object with:\n"
            "1.'paragraphs': Array of 7 strings (PT-BR), each 1-2 short sentences.\n"
            f"{tone_block}"
            'NEVER use "ningu\u00e9m sabia", "o segredo", or "a verdade" \u2014 these are vague.\n'
            "Every paragraph MUST contain a verifiable fact \u2014 no generalities, no filler.\n"
            "Keep each paragraph 1-2 sentences (~2-3 seconds audio each).\n"
            "End with a strong conclusion, NOT 'fica uma li\u00e7\u00e3o' or similar.\n"
            "Use the provided web sources as your primary source of facts. Verify every number against them."
        )
        user_prompt = (
            f"Tell a story about: {subject}. "
            f"Include origin, key facts, and specific details. "
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
        tone_block = self._tone_instructions()
        system_prompt = (
            "You are a master storyteller for viral YouTube Shorts.\n"
            "Output ONLY a JSON object with:\n"
            "1.'paragraphs': Array of 7 strings (PT-BR), each 1-2 short sentences.\n"
            f"{tone_block}"
            'NEVER use "ningu\u00e9m sabia", "o segredo", or "a verdade" \u2014 these are vague.\n'
            "Every paragraph MUST contain a verifiable fact \u2014 no generalities, no filler.\n"
            "Keep each paragraph 1-2 sentences (~2-3 seconds audio each).\n"
            'End with a strong conclusion, NOT "fica uma li\u00e7\u00e3o" or similar.\n'
            "Do the math yourself. If you mention a date range, calculate the years correctly."
        )
        user_prompt = (
            f"Tell a story about: {subject}. "
            f"Include origin, key facts, and specific details. "
            f"Double-check every date and number."
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
    "2.TONE: Curiosity-driven, narrative, engaging. "
    "Write like a storyteller uncovering a fascinating truth \u2014 "
    "never like Wikipedia or a corporate press release.\n"
    "3.FIRST SENTENCE: Drop the viewer right into the action \u2014 "
    "the goal, the controversy, the fact itself. "
    "NO: 'Prepare-se', 'Voc\u00ea sabia', rhetorical questions. "
    "Never waste the first 2 seconds on setup.\n"
    "4.STRUCTURE: Hook (the fact itself) \u2192 Context \u2192 Revelation \u2192 Strong conclusion\n"
    '5.NEVER start with "ningu\u00e9m sabia", "o segredo", "a verdade escondida" '
    'or "voc\u00ea n\u00e3o vai acreditar" \u2014 these are weak.\n'
    "6.FORBIDDEN: Legal names (Ltda, S.A.), addresses, city/state abbreviations. "
    "NO corporate language.\n"
    '6a.FORBIDDEN: Hyperboles, exaggerated claims, "designed by a god", '
    '"you won\'t believe", "shocking truth" \u2014 these sound fake.\n'
    "7.Each paragraph 1-2 punchy sentences (~2-3 seconds audio each).\n"
    "8.Write exactly 4-5 paragraphs.\n"
    "9.NO markdown formatting, NO JSON, just plain text paragraphs.\n"
    "10.Every paragraph must advance the story with a new specific fact \u2014 no filler.\n"
    "11.USE the provided web sources as your primary source of facts. "
    "Cite specific data from them.\n"
     '12.End with a punchy conclusion, NOT "fica uma li\u00e7\u00e3o" or similar generic phrases.\n'
     '13.NEVER use "no final fica uma li\u00e7\u00e3o" or "vale a pena conhecer" \u2014 these are filler.\n'
     "14.Do the math yourself. If you mention a date range or time period, calculate the years correctly."
 )


def _user_prompt_single(subject: str, search_context: str) -> str:
    tone_rules = (
        "- TOM: Curiosidade, narrativa envolvente. "
        "Conte como quem revela um fato fascinante \u2014 "
        "NUNCA como Wikipedia ou release corporativo.\n"
        "- PRIMEIRA FRASE: Jogue o espectador direto na a\u00e7\u00e3o \u2014 "
        "o gol, a pol\u00eamica, o pr\u00f3prio fato. "
        "NADA de 'Prepare-se', 'Voc\u00ea sabia', perguntas ret\u00f3ricas. "
        "N\u00c3O desperdice os primeiros 2 segundos com introdu\u00e7\u00e3o.\n"
        "- ESTRUTURA: Gancho (o fato) \u2192 Contexto \u2192 Revela\u00e7\u00e3o \u2192 Conclus\u00e3o forte\n"
        "- PROIBIDO: Nomes jur\u00eddicos (Ltda, S.A.), endere\u00e7os, siglas. "
        "NADA de linguagem corporativa.\n"
        "- PROIBIDO: Hip\u00e9rboles, exageros, 'desenhada por um deus', "
        "'voc\u00ea n\u00e3o vai acreditar', 'a verdade chocante' \u2014 soa falso.\n"
    )
    return (
        f'Crie uma hist\u00f3ria envolvente em 4-5 par\u00e1grafos sobre "{subject}".\n\n'
        f"{search_context}\n\n"
        f"REGRAS CR\u00cdTICAS:\n"
        f"{tone_rules}"
        f'- NUNCA comece com "ningu\u00e9m sabia", "o segredo", '
        f'"a verdade escondida" \u2014 isso \u00e9 vago e fraco\n'
        f"- Inclua nomes, datas, lugares e n\u00fameros espec\u00edficos sempre que poss\u00edvel\n"
        f"- Conte a ORIGEM: como tudo come\u00e7ou, por que existe\n"
        f"- Cada par\u00e1grafo DEVE conter um FATO VERIFIC\u00c1VEL \u2014 nada de generaliza\u00e7\u00f5es\n"
        f"- NADA de frases de enchimento\n"
        f"- TERMINE com uma conclus\u00e3o forte, N\u00c3O com 'fica uma li\u00e7\u00e3o'\n"
        f"- Fa\u00e7a a conta voc\u00ea mesmo: se mencionar um per\u00edodo, calcule os anos corretamente\n"
        f"- Use as FONTES DA WEB fornecidas como base para sua hist\u00f3ria\n\n"
        f"Escreva cada par\u00e1grafo em uma linha separada."
    )


_SYSTEM_PROMPT_METADATA = (
    "You are a master storyteller for viral YouTube Shorts.\n"
    "CRITICAL RETENTION RULES:\n"
    "1.Write in Brazilian Portuguese (PT-BR).\n"
    "2.TONE: Curiosity-driven, narrative, engaging. "
    "Write like a storyteller uncovering a fascinating truth \u2014 "
    "never like Wikipedia or a corporate press release.\n"
    "3.FIRST SENTENCE: Drop the viewer right into the action \u2014 "
    "the goal, the controversy, the fact itself. "
    "NO: 'Prepare-se', 'Voc\u00ea sabia', rhetorical questions. "
    "Never waste the first 2 seconds on setup.\n"
    "4.STRUCTURE: Hook (the fact itself) \u2192 Context \u2192 Revelation \u2192 Strong conclusion\n"
    '5.NEVER start with "ningu\u00e9m sabia", "o segredo", '
    'or "a verdade escondida" \u2014 these are vague and weak.\n'
    "6.FORBIDDEN: Legal names (Ltda, S.A.), addresses, city/state abbreviations. "
    "NO corporate language.\n"
    '6a.FORBIDDEN: Hyperboles, exaggerated claims, "designed by a god", '
    '"you won\'t believe", "shocking truth" \u2014 these sound fake.\n'
    "7.Extract concrete details from the video metadata: "
    "dates, names, places, statistics, historical context.\n"
    "8.Include origin stories \u2014 explain HOW something started, "
    "not just THAT it happened.\n"
    "9.Each paragraph 1-2 punchy sentences (~2-3 seconds audio each).\n"
    "10.Write exactly 4-5 paragraphs.\n"
    "11.NO markdown formatting, NO JSON, just plain text paragraphs.\n"
    "12.Base your story entirely on the video's content, "
    "adding only well-known historical context.\n"
     '13.End with a punchy conclusion, NOT "fica uma li\u00e7\u00e3o" or similar.\n'
     "14.Every paragraph must contain a verifiable fact \u2014 no generalities."
 )


def _user_prompt_metadata(combined_content: str) -> str:
    tone_rules = (
        "- TOM: Curiosidade, narrativa envolvente. "
        "Conte como quem revela um fato fascinante \u2014 "
        "NUNCA como Wikipedia ou release corporativo.\n"
        "- PRIMEIRA FRASE: Jogue o espectador direto na a\u00e7\u00e3o \u2014 "
        "o gol, a pol\u00eamica, o pr\u00f3prio fato. "
        "NADA de 'Prepare-se', 'Voc\u00ea sabia', perguntas ret\u00f3ricas. "
        "N\u00c3O desperdice os primeiros 2 segundos com introdu\u00e7\u00e3o.\n"
        "- ESTRUTURA: Gancho (o fato) \u2192 Contexto \u2192 Revela\u00e7\u00e3o \u2192 Conclus\u00e3o forte\n"
        "- PROIBIDO: Nomes jur\u00eddicos (Ltda, S.A.), endere\u00e7os, siglas. "
        "NADA de linguagem corporativa.\n"
        "- PROIBIDO: Hip\u00e9rboles, exageros, 'desenhada por um deus', "
        "'voc\u00ea n\u00e3o vai acreditar' \u2014 soa falso.\n"
    )
    return (
        "Crie uma hist\u00f3ria envolvente em 4-5 par\u00e1grafos baseada "
        "neste v\u00eddeo do YouTube.\n\n"
        "REGRAS CR\u00cdTICAS:\n"
        f"{tone_rules}"
        '- NUNCA comece com "ningu\u00e9m sabia", "o segredo" '
        'ou "a verdade escondida"\n'
        "- Extraia detalhes espec\u00edficos do t\u00edtulo e descri\u00e7\u00e3o: "
        "datas, nomes, locais, estat\u00edsticas\n"
        "- Conte a ORIGEM: como tudo come\u00e7ou, por que \u00e9 importante\n"
        "- Cada par\u00e1grafo DEVE conter um FATO VERIFIC\u00c1VEL \u2014 nada de generaliza\u00e7\u00f5es\n"
        "- NADA de ganchos gen\u00e9ricos ou frases de enchimento\n"
        "- TERMINE com uma conclus\u00e3o forte, N\u00c3O com 'fica uma li\u00e7\u00e3o'\n"
        "- Fa\u00e7a a conta voc\u00ea mesmo: se mencionar um per\u00edodo, calcule os anos corretamente\n\n"
        f"V\u00eddeo:\n{combined_content}\n\n"
        "Escreva cada par\u00e1grafo em uma linha separada."
    )
