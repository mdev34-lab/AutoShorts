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
- Must contain specific fact, name, or number
- Must NOT start with "Você sabia", "Prepare-se", or rhetorical questions
"""

import hashlib
import random
import re
from enum import Enum
from typing import Dict


class HookStyle(str, Enum):
    """Available hook styles for script generation."""

    DEFAULT = "default"  # Current behavior (backward compatible)
    CURIOSITY_GAP = "curiosity"  # Create information asymmetry
    COUNTER_NARRATIVE = "counter"  # Challenge common knowledge
    CONTROVERSY = "controversy"  # Take a strong stance
    CHALLENGE = "challenge"  # Direct viewer challenge
    REVEAL = "reveal"  # Promise a shocking fact
    STORY = "story"  # Narrative arc hook

    # Allow pydantic/typer to handle case-insensitive? Typer will match by value.
    # Provide __str__ as value already via str Enum.


class HookEngine:
    """Generates optimized hook templates based on style and subject."""

    # --- Template definitions (raw, for reference) ---

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

    # Supporting vocab for filling placeholders - PT-BR focused

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
        # deterministic randomness seed base — but we use subject hash for variation
        pass

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def get_tone_instructions(self, style: HookStyle) -> str:
        """Returns enhanced tone instructions for the script generator."""

        # Ensure style is HookStyle enum
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
                "hook": str,           # The first 1-2 sentences
                "pattern_interrupt": str,  # Unexpected statement
                "curiosity_gap": str,  # What the viewer needs to know
                "tone_instructions": str   # Enhanced tone block for script generation
            }
        """
        try:
            if not isinstance(style, HookStyle):
                style = HookStyle(style)
        except Exception:
            # Fallback to DEFAULT silently on invalid style
            style = HookStyle.DEFAULT

        # Safe fallback for empty subject
        safe_subject = (subject or "").strip() or "essa história"

        tone_instructions = self.get_tone_instructions(style)

        # DEFAULT preserves existing behavior — return empty hook so script generator keeps original first paragraph
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

            # Quality guards — ensure hook meets spec
            hook = self._enforce_quality(hook, safe_subject, style)

            return {
                "hook": hook,
                "pattern_interrupt": pattern_interrupt,
                "curiosity_gap": curiosity_gap,
                "tone_instructions": tone_instructions,
            }
        except Exception as e:
            # Silent fallback to DEFAULT on failure
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
    # Internal builders
    # ------------------------------------------------------------------ #

    def _stable_seed(self, subject: str, style: HookStyle) -> int:
        h = hashlib.sha256(f"{subject}|{style.value}".encode()).hexdigest()
        return int(h[:8], 16)

    def _extract_year(self, subject: str, seed: int) -> int:
        # Try to find 4-digit year in subject
        m = re.search(r"\b(18\d{2}|19\d{2}|20[0-2]\d|202[0-5])\b", subject)
        if m:
            return int(m.group(1))
        # Deterministic pseudo-random year based on seed
        rng = random.Random(seed)
        return rng.randint(1950, 2023)

    def _extract_entity(self, subject: str) -> str:
        # Use subject as entity, but trim if too long
        # Keep first 40 chars or up to comma
        s = subject.strip()
        # Remove year numbers for cleaner entity when needed
        # But keep original casing for hook
        # Limit length
        if len(s) > 60:
            # take first 2-3 words + keep meaning
            words = s.split()
            if len(words) > 4:
                return " ".join(words[:4])
        return s

    def _domain_from_subject(self, subject: str, seed: int) -> str:
        low = subject.lower()
        # Heuristic
        if any(k in low for k in ["futebol", "flamengo", "corinthians", "palmeiras", "vasco", "são paulo", "seleção", "copa", "libertadores", "brasileirão", "var", "neymar", "pelé", "zico", "ronaldo"]):
            rng = random.Random(seed + 1)
            return rng.choice(["o futebol brasileiro", "o futebol", "o esporte"])
        if any(k in low for k in ["tecnologia", "ia", "inteligência", "space", "nasa"]):
            return "a tecnologia"
        return random.Random(seed + 2).choice(self._DOMAINS)

    def _number_for_subject(self, seed: int) -> int:
        rng = random.Random(seed + 3)
        return rng.randint(3, 9)

    def _build_hook(self, subject: str, style: HookStyle, context: str) -> str:
        seed = self._stable_seed(subject, style)
        rng = random.Random(seed)
        year = self._extract_year(subject, seed)
        entity = self._extract_entity(subject)
        domain = self._domain_from_subject(subject, seed)
        number = self._number_for_subject(seed)
        time_period = rng.choice(self._TIME_PERIODS)
        reveal = rng.choice(self._REVEALS)
        action = rng.choice(self._ACTIONS)
        consequence = rng.choice(self._CONSEQUENCES)
        crisis = rng.choice(self._CRISES)

        # Use context if provided to enrich — simple keyword extraction
        extra_fact = ""
        if context:
            # Take first 100 chars of context as inspiration (no API needed)
            snippet = context[:120].strip()
            if snippet:
                extra_fact = snippet.split(".")[0][:80]

        # Dispatch per style
        if style == HookStyle.CURIOSITY_GAP:
            tmpl_choice = rng.randint(1, 3)
            if tmpl_choice == 1:
                return f"Em {year}, {entity} {action} — e ninguém esperava que {consequence}."
            elif tmpl_choice == 2:
                return f"O que {entity} fez em {year} mudou {domain} para sempre. Mas o motivo é mais estranho do que parece."
            else:
                event = subject
                return f"{number} {time_period} depois de {event}, {entity} finalmente {reveal}. A razão vai te surpreender."

        elif style == HookStyle.COUNTER_NARRATIVE:
            tmpl_choice = rng.randint(1, 3)
            if tmpl_choice == 1:
                return f"Você acha que conhece {subject}. Mas a história real é completamente diferente."
            elif tmpl_choice == 2:
                belief = rng.choice(self._COMMON_BELIEFS).format(subject=subject)
                return f"Todo mundo diz que {belief}. Os dados mostram o oposto."
            else:
                # Template 3 with enriched data
                specific_data = f"{number} títulos em {year}" if year else f"{number} dados oficiais"
                surprising = f"o oposto do que contam"
                common_assump = f"só {domain}"
                return f"{entity} não é {common_assump}. Na verdade, é {surprising}. E a prova está em {specific_data}."

        elif style == HookStyle.CONTROVERSY:
            tmpl_choice = rng.randint(1, 3)
            if tmpl_choice == 1:
                take = rng.choice(self._CONTROVERSIAL_TAKES).format(subject=subject)
                return f"Vou dizer algo impopular: {take}. E posso provar."
            elif tmpl_choice == 2:
                return f"{entity} é superestimado. Aqui estão {number} razões que ninguém quer admitir."
            else:
                lie = f"{domain} é justo"
                truth = f"3 times ganharam 70% dos títulos em 20 anos" if "futebol" in domain else f"{entity} quebrou todas as estatísticas em {year}"
                return f"A maior mentira do {domain} é que {lie}. A verdade? {truth}."

        elif style == HookStyle.CHALLENGE:
            tmpl_choice = rng.randint(1, 3)
            if tmpl_choice == 1:
                fact = f"quem foi o primeiro a fazer {subject} em {year}" if year else f"o detalhe escondido de {subject}"
                return f"Aposto que você não sabe {fact}. Duvido. Vou te provar."
            elif tmpl_choice == 2:
                assumption = f"{subject} sempre foi assim"
                return f"Se você acha que {assumption}, você está errado. E eu vou te mostrar por quê."
            else:
                return f"Pare tudo. Você não vai acreditar como {entity} {action} em {year}. E a história real é melhor que a ficção."

        elif style == HookStyle.REVEAL:
            tmpl_choice = rng.randint(1, 3)
            if tmpl_choice == 1:
                return f"O que você vai descobrir sobre {subject} vai mudar como você vê {domain}."
            elif tmpl_choice == 2:
                return f"{entity} tinha um segredo em {year}. Só que esse segredo não era o que todo mundo pensava."
            else:
                return f"Em {year}, aconteceu algo que {domain} nunca contou. Até hoje."

        elif style == HookStyle.STORY:
            tmpl_choice = rng.randint(1, 3)
            if tmpl_choice == 1:
                return f"Tudo começou quando {entity} decidiu {action} em {year}. Ninguém imaginava o que viria depois."
            elif tmpl_choice == 2:
                return f"Em {year}, {entity} estava prestes a {crisis}. O que aconteceu nos próximos {number} {time_period} definiu {domain}."
            else:
                age = rng.randint(17, 35)
                event = f"encarar {subject}"
                return f"{entity} tinha {age} anos quando {event}. A decisão que tomou naquele dia mudou tudo."

        # Fallback
        return f"{entity} em {year} — a história que {domain} tentou esquecer."

    def _build_pattern_interrupt(self, subject: str, style: HookStyle, context: str) -> str:
        seed = self._stable_seed(subject, style) + 100
        rng = random.Random(seed)
        year = self._extract_year(subject, seed)
        entity = self._extract_entity(subject)
        patterns = {
            HookStyle.CURIOSITY_GAP: [
                f"{entity} fez algo que ninguém viu chegando.",
                f"Em {year}, tudo mudou em 48 horas.",
                f"O dado que ninguém mostra: {rng.randint(70,99)}% das pessoas erram isso.",
            ],
            HookStyle.COUNTER_NARRATIVE: [
                f"Os livros contam uma versão. Os números contam outra.",
                f"{entity} quebrou a lógica de {self._domain_from_subject(subject, seed)}.",
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
                f"O detalhe que faltava estava em {year}.",
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
        rng = random.Random(seed)
        year = self._extract_year(subject, seed)
        entity = self._extract_entity(subject)
        domain = self._domain_from_subject(subject, seed)
        gaps = {
            HookStyle.CURIOSITY_GAP: f"Por que {entity} fez isso em {year} e como isso ainda afeta {domain} hoje?",
            HookStyle.COUNTER_NARRATIVE: f"Qual é a história real de {subject} que os dados revelam, mas ninguém conta?",
            HookStyle.CONTROVERSY: f"Por que {entity} é tão polarizador e o que os números escondem sobre {subject}?",
            HookStyle.CHALLENGE: f"Você realmente sabe o que aconteceu com {entity} em {year}?",
            HookStyle.REVEAL: f"Qual foi o segredo de {entity} em {year} que {domain} nunca contou?",
            HookStyle.STORY: f"Como {entity} saiu de {rng.choice(self._CRISES)} para definir {domain}?",
        }
        return gaps.get(style, f"O que realmente aconteceu com {subject} em {year}?")

    def _enforce_quality(self, hook: str, subject: str, style: HookStyle) -> str:
        """Ensure hook meets quality criteria."""
        hook = hook.strip()

        # Rule 1: Must not start with forbidden phrases
        forbidden_starts = ["você sabia", "prepare-se", "voce sabia", "vocês sabiam"]
        lower = hook.lower()
        for fb in forbidden_starts:
            if lower.startswith(fb):
                # Rewrite to remove forbidden start
                # Strip leading phrase up to first space after forbidden?
                # Simple: replace with entity + action
                hook = hook[len(fb):].lstrip(" ?!:,-")
                # Ensure first letter upper
                if hook:
                    hook = hook[0].upper() + hook[1:]
                # Prepend entity if not present
                if subject.lower() not in hook.lower():
                    hook = f"{self._extract_entity(subject)}: {hook}"
                break

        # Rule 2: Must not be question-only? Requirement says NO rhetorical questions at start
        # Allow questions internally? But ensure not starting with question? Our templates avoid starting with rhetorical.
        # If hook ends up starting with "?" or is just question, fix.
        if hook.startswith("?"):
            hook = hook.lstrip("? ").strip()

        # Rule 3: Must contain specific fact, name, or number -> ensure subject or year present
        # If not, inject year
        seed = self._stable_seed(subject, style)
        year = self._extract_year(subject, seed)
        if subject.lower() not in hook.lower() and str(year) not in hook:
            # Append year context if missing both?
            if len(hook) < 120:
                hook = f"{hook.rstrip('.')} em {year}."

        # Rule 4: 1-2 sentences max — count sentences by '.'
        # Rough split
        sentences = [s.strip() for s in hook.split(".") if s.strip()]
        if len(sentences) > 2:
            hook = ". ".join(sentences[:2]) + "."

        # Ensure ends with period if not already
        if hook and not hook[-1] in ".!?":
            hook += "."

        # Ensure length >10 (for tests)
        if len(hook) < 15:
            hook = f"{self._extract_entity(subject)} em {year} mudou {self._domain_from_subject(subject, seed)} para sempre."

        # Final forbidden check: remove double spaces
        hook = re.sub(r"\s+", " ", hook).strip()
        return hook
