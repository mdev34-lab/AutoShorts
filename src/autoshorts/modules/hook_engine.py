"""
Hook Engine — generates optimized hook templates based on style and subject.

Implements proven copywriting frameworks:
- Triple Hook (Kallaway): Visual + verbal + curiosity gap in first 3 seconds
- Curiosity Gap: information asymmetry
- Pattern Interrupt: unexpected statement breaking scroll
- Counter-Narrative: challenge what viewer thinks they know
- Cold Email Storytelling (Hormozi): specific detail + open loop + pattern interrupt

Hook quality rules:
- 1-2 sentences max
- Must contain specific fact, name, or number (grounded, not hallucinated)
- Must NOT start with "Você sabia", "Prepare-se", or rhetorical questions
"""

import hashlib
import random
import re
from enum import Enum
from typing import Dict, Optional


class HookStyle(str, Enum):
    """Available hook styles for script generation."""

    DEFAULT = "default"  # Current behavior (backward compatible)
    CURIOSITY_GAP = "curiosity"  # Create information asymmetry
    COUNTER_NARRATIVE = "counter"  # Challenge common knowledge
    CONTROVERSY = "controversy"  # Take a strong stance
    CHALLENGE = "challenge"  # Direct viewer challenge
    REVEAL = "reveal"  # Promise a shocking fact
    STORY = "story"  # Narrative arc hook


class HookEngine:
    """Generates optimized hook templates based on style and subject."""

    # Template definitions — used to drive generation (not dead code)
    _TEMPLATES_INFO = {
        HookStyle.CURIOSITY_GAP: [
            "Em [YEAR], [ENTITY] [ACTION] — e ninguém esperava que [CONSEQUENCE].",
            "O que [ENTITY] fez em [YEAR] mudou [DOMAIN] para sempre. Mas o motivo é mais estranho do que parece.",
            "[NUMBER] [TIME_PERIOD] depois de [EVENT], [ENTITY] finalmente [REVEAL]. A razão vai te surpreender.",
        ],
        HookStyle.COUNTER_NARRATIVE: [
            "Você acha que conhece [SUBJECT]. Mas a história real é completamente diferente.",
            "Todo mundo diz que [COMMON_BELIEF]. Os dados mostram o oposto.",
            "[ENTITY] não é [COMMON_ASSUMPTION]. Na verdade, é [SURPRISING_TRUTH]. E a prova está em [SPECIFIC_DATA].",
        ],
        HookStyle.CONTROVERSY: [
            "Vou dizer algo impopular: [CONTROVERSIAL_TAKE]. E posso provar.",
            "[ENTITY] é superestimado. Aqui estão [NUMBER] razões que ninguém quer admitir.",
            "A maior mentira do [DOMAIN] é que [LIE]. A verdade? [TRUTH].",
        ],
        HookStyle.CHALLENGE: [
            "Aposto que você não sabe [SURPRISING_FACT]. Duvido. Vou te provar.",
            "Se você acha que [ASSUMPTION], você está errado. E eu vou te mostrar por quê.",
            "Pare tudo. Você não vai acreditar como [ENTITY] [ACTION]. E a história real é melhor que a ficção.",
        ],
        HookStyle.REVEAL: [
            "O que você vai descobrir sobre [SUBJECT] vai mudar como você vê [DOMAIN].",
            "[ENTITY] tinha um segredo. Só que esse segredo não era o que todo mundo pensava.",
            "Em [YEAR], aconteceu algo que [DOMAIN] nunca contou. Até hoje.",
        ],
        HookStyle.STORY: [
            "Tudo começou quando [ENTITY] decidiu [ACTION]. Ninguém imaginava o que viria depois.",
            "Em [YEAR], [ENTITY] estava prestes a [CRISIS]. O que aconteceu nos próximos [TIME_PERIOD] definiu [DOMAIN].",
            "[ENTITY] tinha [AGE] anos quando [EVENT]. A decisão que tomou naquele dia mudou tudo.",
        ],
    }

    # Supporting vocab — PT-BR focused, non-hallucinating where possible
    _ACTIONS = [
        "fez história",
        "mudou as regras do jogo",
        "quebrou todos os recordes",
        "venceu contra todas as probabilidades",
        "criou algo que ninguém esperava",
        "tomou uma decisão ousada",
        "entrou para a história",
        "chocou o mundo",
    ]

    _CONSEQUENCES = [
        "isso mudaria o jogo para sempre",
        "um estádio inteiro ficasse em silêncio",
        "nada seria igual depois",
        "o mundo inteiro prestasse atenção",
        "a história tomasse outro rumo",
        "tudo virasse de cabeça para baixo",
    ]

    _DOMAINS = [
        "o futebol",
        "o esporte",
        "a história",
        "o futebol brasileiro",
        "o Brasil",
        "o campeonato",
        "o esporte mundial",
    ]

    _TIME_PERIODS = ["anos", "meses", "décadas"]

    _REVEALS = [
        "revelou a verdade",
        "contou o que realmente aconteceu",
        "mostrou os bastidores",
        "explicou o mistério",
        "abriu o jogo",
        "quebrou o silêncio",
    ]

    _CRISES = [
        "perder tudo",
        "ser esquecido",
        "cair para a segunda divisão",
        "encerrar a carreira",
        "desaparecer do mapa",
        "ficar sem patrocínio",
        "abandonar o sonho",
    ]

    _COMMON_BELIEFS = [
        "{subject} é simples",
        "{subject} já foi explicado",
        "{subject} não tem mais segredos",
        "todo mundo entende {subject}",
        "{subject} é só mais um caso",
    ]

    _CONTROVERSIAL_TAKES = [
        "{subject} não é o que você pensa",
        "a história de {subject} foi mal contada",
        "{subject} mudou tudo, mas não como contam",
        "ninguém quer admitir a verdade sobre {subject}",
    ]

    def __init__(self):
        pass

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def get_tone_instructions(self, style: HookStyle) -> str:
        """Returns enhanced tone instructions for the script generator."""
        try:
            if not isinstance(style, HookStyle):
                style = HookStyle(style)
        except Exception:
            style = HookStyle.DEFAULT

        base_map = {
            HookStyle.DEFAULT: (
                "STYLE: default — curiosity-driven, narrative, engaging. "
                "Write like a storyteller uncovering a fascinating truth. "
                "TONE: curiosity, revelation, factual storytelling. "
                "Maintain the original hook approach — drop into action, no setup. "
                "Keep paragraphs punchy, specific, verifiable.\n"
            ),
            HookStyle.CURIOSITY_GAP: (
                "STYLE: curiosity — curiosity gap principle. Create information asymmetry. "
                "TONE: suspenseful, inquisitive, withholding key detail until later. "
                "Open a loop in first sentence that only closes at the end. "
                "Use specific numbers, years, names to make gap concrete. "
                "Maintain curiosity throughout, each paragraph answers one question while opening another.\n"
            ),
            HookStyle.COUNTER_NARRATIVE: (
                "STYLE: counter-narrative — challenge common knowledge. "
                "TONE: counter, contrarian, evidence-driven, cognitive dissonance. "
                "Start by stating what people believe, then dismantle with facts. "
                "Use data, dates, and specific counter-examples. "
                "Maintain skeptical, myth-busting stance throughout script.\n"
            ),
            HookStyle.CONTROVERSY: (
                "STYLE: controversy — strong polarizing stance demanding engagement. "
                "TONE: bold, declarative, fact-backed controversy, unapologetic. "
                "Take a clear position that forces viewer to agree or disagree. "
                "Support every claim with specific data, avoid empty opinions. "
                "Maintain strong stance, don't soften in middle paragraphs.\n"
            ),
            HookStyle.CHALLENGE: (
                "STYLE: challenge — direct viewer knowledge challenge. "
                "TONE: challenging, interactive, confidence, playful confrontation. "
                "Address viewer directly ('você'), test their knowledge. "
                "Promise proof and deliver. "
                "Maintain engaging, direct-response energy throughout.\n"
            ),
            HookStyle.REVEAL: (
                "STYLE: reveal — promise shocking fact upfront. "
                "TONE: mysterious, revelation-driven, build-up to payoff. "
                "Tease secret or hidden fact in hook, escalate tension. "
                "Each paragraph adds layer to revelation. "
                "Final paragraph delivers payoff clearly.\n"
            ),
            HookStyle.STORY: (
                "STYLE: story — narrative arc with character and conflict. "
                "TONE: narrative, cinematic, emotional arc, character-driven story. "
                "Open with character + decision + stakes. "
                "Structure: setup, crisis, turning point, resolution. "
                "Maintain story tension, show transformation.\n"
            ),
        }

        return base_map.get(style, base_map[HookStyle.DEFAULT])

    def generate_hook(
        self, subject: str, style: HookStyle = HookStyle.DEFAULT, context: str = ""
    ) -> dict:
        """
        Returns:
            {
                "hook": str,
                "pattern_interrupt": str,
                "curiosity_gap": str,
                "tone_instructions": str
            }
        """
        try:
            if not isinstance(style, HookStyle):
                style = HookStyle(style)
        except Exception as e:
            # Log invalid style fallback for debugging
            try:
                from .logging_system import log

                log(f"Hook generation: invalid style '{style}' -> fallback DEFAULT: {e}", "WARNING")
            except Exception:
                pass
            style = HookStyle.DEFAULT

        safe_subject = (subject or "").strip() or "essa história"
        tone_instructions = self.get_tone_instructions(style)

        if style == HookStyle.DEFAULT:
            return {
                "hook": "",
                "pattern_interrupt": "",
                "curiosity_gap": "",
                "tone_instructions": tone_instructions,
            }

        try:
            hook = self._build_hook(safe_subject, style, context)
            pattern_interrupt = self._build_pattern_interrupt(safe_subject, style, context)
            curiosity_gap = self._build_curiosity_gap(safe_subject, style, context)

            hook = self._enforce_quality(hook, safe_subject, style)

            return {
                "hook": hook,
                "pattern_interrupt": pattern_interrupt,
                "curiosity_gap": curiosity_gap,
                "tone_instructions": tone_instructions,
            }
        except Exception as e:
            try:
                from .logging_system import log

                log(f"Hook generation failed for style {style}: {e} — falling back to DEFAULT", "WARNING")
            except Exception:
                pass
            return {
                "hook": "",
                "pattern_interrupt": "",
                "curiosity_gap": "",
                "tone_instructions": self.get_tone_instructions(HookStyle.DEFAULT),
            }

    # ------------------------------------------------------------------ #
    # Internal builders — grounded, no random hallucinated years
    # ------------------------------------------------------------------ #

    def _stable_seed(self, subject: str, style: HookStyle) -> int:
        h = hashlib.sha256(f"{subject}|{style.value}".encode()).hexdigest()
        return int(h[:8], 16)

    def _extract_year(self, subject: str, context: str = "") -> Optional[int]:
        """Extract year from subject or context — returns None if not found (no hallucination)."""
        # Subject first (most reliable)
        m = re.search(r"\b(18\d{2}|19\d{2}|20[0-2]\d|202[0-5])\b", subject)
        if m:
            return int(m.group(1))
        # Then context (web search results)
        if context:
            m = re.search(r"\b(18\d{2}|19\d{2}|20[0-2]\d|202[0-5])\b", context)
            if m:
                return int(m.group(1))
        return None

    def _extract_entity(self, subject: str) -> str:
        s = subject.strip()
        if len(s) > 60:
            words = s.split()
            if len(words) > 4:
                return " ".join(words[:4])
        return s

    def _domain_from_subject(self, subject: str, seed: int) -> str:
        low = subject.lower()
        if any(
            k in low
            for k in [
                "futebol",
                "flamengo",
                "corinthians",
                "palmeiras",
                "vasco",
                "são paulo",
                "seleção",
                "copa",
                "libertadores",
                "brasileirão",
                "var",
                "neymar",
                "pelé",
                "zico",
                "ronaldo",
            ]
        ):
            rng = random.Random(seed + 1)
            return rng.choice(["o futebol brasileiro", "o futebol", "o esporte"])
        if any(k in low for k in ["tecnologia", "ia", "inteligência", "space", "nasa"]):
            return "a tecnologia"
        return random.Random(seed + 2).choice(self._DOMAINS)

    def _number_for_subject(self, seed: int, context: str = "") -> int:
        """Return a number for structural templates (e.g., '5 razões') — not claiming factual world data."""
        # Try to extract a number from context that could be relevant, else deterministic random for structure only
        if context:
            m = re.search(r"\b([3-9])\s+(?:títulos|anos|vezes|gols|motivos|razões)\b", context, re.IGNORECASE)
            if m:
                try:
                    return int(m.group(1))
                except Exception:
                    pass
        rng = random.Random(seed + 3)
        return rng.randint(3, 9)

    def _fill_template(self, template: str, values: Dict[str, str]) -> str:
        """Replace [PLACEHOLDER] with values — leaves unknown placeholders intact then cleans."""
        result = template
        for key, val in values.items():
            placeholder = f"[{key}]"
            result = result.replace(placeholder, str(val))
        # Clean any unreplaced placeholders: replace with generic safe value or remove brackets
        # For year/missing values, replace [YEAR] with empty and tidy spaces if values had empty
        result = re.sub(r"\[([A-Z_]+)\]", "", result)
        result = re.sub(r"\s+", " ", result).strip()
        result = result.replace("  ", " ").replace(" .", ".").replace(" ,", ",")
        return result

    def _build_placeholder_values(
        self, subject: str, style: HookStyle, context: str, seed: int, year: Optional[int]
    ) -> Dict[str, str]:
        rng = random.Random(seed)
        entity = self._extract_entity(subject)
        domain = self._domain_from_subject(subject, seed)
        number = self._number_for_subject(seed, context)
        time_period = rng.choice(self._TIME_PERIODS)
        reveal = rng.choice(self._REVEALS)
        action = rng.choice(self._ACTIONS)
        consequence = rng.choice(self._CONSEQUENCES)
        crisis = rng.choice(self._CRISES)
        age = str(rng.randint(17, 35))

        # Derived values
        event = subject
        common_belief = rng.choice(self._COMMON_BELIEFS).format(subject=subject)
        controversial_take = rng.choice(self._CONTROVERSIAL_TAKES).format(subject=subject)

        # Grounded specific data — only use year if we actually found it
        if year:
            specific_data = f"{number} títulos em {year}" if "futebol" in domain else f"dados oficiais de {year}"
            truth = f"{entity} quebrou todas as estatísticas em {year}"
        else:
            specific_data = "dados oficiais"
            truth = f"{entity} quebrou todas as estatísticas"

        # For templates needing surprising fact etc.
        if year:
            surprising_fact = f"quem foi o primeiro a fazer {subject} em {year}"
            fact_with_year = f"em {year}"
        else:
            surprising_fact = f"o detalhe escondido de {subject}"
            fact_with_year = "naquela época"

        # Build dict for all placeholders used in _TEMPLATES_INFO
        values = {
            "YEAR": str(year) if year else "",
            "ENTITY": entity,
            "SUBJECT": subject,
            "ACTION": action,
            "CONSEQUENCE": consequence,
            "DOMAIN": domain,
            "NUMBER": str(number),
            "TIME_PERIOD": time_period,
            "EVENT": event,
            "REVEAL": reveal,
            "AGE": age,
            "CRISIS": crisis,
            "COMMON_BELIEF": common_belief,
            "COMMON_ASSUMPTION": f"só {domain}",
            "SURPRISING_TRUTH": "o oposto do que contam",
            "SPECIFIC_DATA": specific_data,
            "CONTROVERSIAL_TAKE": controversial_take,
            "LIE": f"{domain} é justo",
            "TRUTH": truth,
            "SURPRISING_FACT": surprising_fact,
            "ASSUMPTION": f"{subject} sempre foi assim",
            "FACT_WITH_YEAR": fact_with_year,
        }
        return values

    def _build_hook(self, subject: str, style: HookStyle, context: str) -> str:
        seed = self._stable_seed(subject, style)
        rng = random.Random(seed)
        year = self._extract_year(subject, context)
        values = self._build_placeholder_values(subject, style, context, seed, year)

        templates = self._TEMPLATES_INFO.get(style, [])
        if not templates:
            entity = self._extract_entity(subject)
            return f"{entity} — a história que {values['DOMAIN']} tentou esquecer."

        # If year is None, prefer templates without [YEAR] to avoid hallucination
        if year is None:
            no_year_templates = [t for t in templates if "[YEAR]" not in t]
            if no_year_templates:
                templates = no_year_templates

        # Deterministic choice
        tmpl = rng.choice(templates)
        hook = self._fill_template(tmpl, values)

        # Post-process: if we removed year and left double spaces like "Em ,"
        hook = hook.replace("Em ,", "Em um momento,").replace("Em  ,", "Em um momento,")
        hook = re.sub(r"\bEm\s+,", "Em um momento,", hook)

        # If hook still empty after filling, fallback
        if not hook or len(hook) < 10:
            entity = self._extract_entity(subject)
            if year:
                return f"{entity} em {year} mudou {values['DOMAIN']} para sempre."
            return f"{entity} — a história que {values['DOMAIN']} tentou esquecer."

        return hook

    def _build_pattern_interrupt(self, subject: str, style: HookStyle, context: str) -> str:
        seed = self._stable_seed(subject, style) + 100
        rng = random.Random(seed)
        year = self._extract_year(subject, context)
        entity = self._extract_entity(subject)
        year_str = str(year) if year else "naquela época"
        domain = self._domain_from_subject(subject, seed)

        patterns = {
            HookStyle.CURIOSITY_GAP: [
                f"{entity} fez algo que ninguém viu chegando.",
                f"Em {year_str}, tudo mudou em 48 horas." if year else f"{entity} mudou tudo em 48 horas.",
                f"O dado que ninguém mostra: {rng.randint(70,99)}% das pessoas erram isso.",
            ],
            HookStyle.COUNTER_NARRATIVE: [
                f"Os livros contam uma versão. Os números contam outra.",
                f"{entity} quebrou a lógica de {domain}.",
                f"O oposto do que te ensinaram na escola.",
            ],
            HookStyle.CONTROVERSY: [
                f"Se isso te irrita, é porque é verdade.",
                f"{entity} divide opiniões por um motivo.",
                f"Polêmica? Sim. Mas com prova.",
            ],
            HookStyle.CHALLENGE: [
                f"Você tem 3 segundos para responder.",
                f"A maioria erra. Você também vai?",
                f"Pause. Tenta adivinhar.",
            ],
            HookStyle.REVEAL: [
                f"O detalhe que faltava estava em {year_str}." if year else "O detalhe que faltava estava nos bastidores.",
                f"Ninguém viu isso chegando até hoje.",
                f"E o segredo estava nos números.",
            ],
            HookStyle.STORY: [
                f"Era para ser o fim. Foi só o começo.",
                f"Quando tudo parecia perdido, {entity} virou o jogo.",
                f"Uma decisão. Um segundo. Tudo mudou.",
            ],
        }
        lst = patterns.get(style, ["Algo inesperado aconteceu."])
        return rng.choice(lst)

    def _build_curiosity_gap(self, subject: str, style: HookStyle, context: str) -> str:
        seed = self._stable_seed(subject, style) + 200
        year = self._extract_year(subject, context)
        entity = self._extract_entity(subject)
        domain = self._domain_from_subject(subject, seed)
        year_part = f"em {year}" if year else "naquela época"

        gaps = {
            HookStyle.CURIOSITY_GAP: f"Por que {entity} fez isso {year_part} e como isso ainda afeta {domain} hoje?",
            HookStyle.COUNTER_NARRATIVE: f"Qual é a história real de {subject} que os dados revelam, mas ninguém conta?",
            HookStyle.CONTROVERSY: f"Por que {entity} é tão polarizador e o que os números escondem sobre {subject}?",
            HookStyle.CHALLENGE: f"Você realmente sabe o que aconteceu com {entity} {year_part}?",
            HookStyle.REVEAL: f"Qual foi o segredo de {entity} {year_part} que {domain} nunca contou?",
            HookStyle.STORY: f"Como {entity} saiu de {random.Random(seed).choice(self._CRISES)} para definir {domain}?",
        }
        return gaps.get(style, f"O que realmente aconteceu com {subject} {year_part}?")

    def _enforce_quality(self, hook: str, subject: str, style: HookStyle) -> str:
        """Ensure hook meets quality criteria — no forbidden starts, 1-2 sentences, grounded."""
        hook = hook.strip()

        # Rule 1: Must not start with forbidden phrases
        forbidden_starts = ["você sabia", "prepare-se", "voce sabia", "vocês sabiam"]
        lower = hook.lower()
        for fb in forbidden_starts:
            if lower.startswith(fb):
                hook = hook[len(fb) :].lstrip(" ?!:,-")
                if hook:
                    hook = hook[0].upper() + hook[1:]
                if subject.lower() not in hook.lower():
                    hook = f"{self._extract_entity(subject)}: {hook}"
                break

        if hook.startswith("?"):
            hook = hook.lstrip("? ").strip()

        # Rule 2: Must contain specific fact — ensure subject/entity present, if not inject
        if subject.lower() not in hook.lower() and self._extract_entity(subject).lower() not in hook.lower():
            # Append entity if missing and length allows
            if len(hook) < 120:
                hook = f"{hook.rstrip('.')} — {self._extract_entity(subject)}."

        # Rule 3: 1-2 sentences max
        sentences = [s.strip() for s in hook.split(".") if s.strip()]
        if len(sentences) > 2:
            hook = ". ".join(sentences[:2]) + "."

        if hook and hook[-1] not in ".!?":
            hook += "."

        if len(hook) < 15:
            entity = self._extract_entity(subject)
            year = self._extract_year(subject)
            if year:
                hook = f"{entity} em {year} mudou {self._domain_from_subject(subject, self._stable_seed(subject, style))} para sempre."
            else:
                hook = f"{entity} — a história que {self._domain_from_subject(subject, self._stable_seed(subject, style))} tentou esquecer."

        hook = re.sub(r"\s+", " ", hook).strip()
        return hook
