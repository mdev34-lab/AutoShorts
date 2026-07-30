"""Hook Engine for AutoShorts - generates optimized hooks based on style."""

from enum import Enum


class HookStyle(str, Enum):
    """Available hook styles for script generation."""
    DEFAULT = "default"           # Current behavior (backward compatible)
    CURIOSITY_GAP = "curiosity"   # Create information asymmetry
    COUNTER_NARRATIVE = "counter" # Challenge common knowledge
    CONTROVERSY = "controversy"   # Take a strong stance
    CHALLENGE = "challenge"       # Direct viewer challenge
    REVEAL = "reveal"            # Promise a shocking fact
    STORY = "story"              # Narrative arc hook


class HookEngine:
    """Generates optimized hook templates based on style and subject."""
    
    def __init__(self):
        self._templates = {
            HookStyle.CURIOSITY_GAP: [
                "Em {year}, {entity} {action} — e ninguém esperava que {consequence}.",
                "O que {entity} fez em {year} mudou {domain} para sempre. Mas o motivo é mais estranho do que parece.",
                "{number} {time_period} depois de {event}, {entity} finalmente {reveal}. A razão vai te surpreender.",
            ],
            HookStyle.COUNTER_NARRATIVE: [
                "Você acha que conhece {subject}. Mas a história real é completamente diferente.",
                "Todo mundo diz que {common_belief}. Os dados mostram o oposto.",
                "{entity} não é {common_assumption}. Na verdade, é {surprising_truth}. E a prova está em {specific_data}.",
            ],
            HookStyle.CONTROVERSY: [
                "Vou dizer algo impopular: {controversial_take}. E posso provar.",
                "{entity} é superestimado. Aqui estão {number} razões que ninguém quer admitir.",
                "A maior mentira do {domain} é que {lie}. A verdade? {truth}.",
            ],
            HookStyle.CHALLENGE: [
                "Aposto que você não sabe {surprising_fact}. Duvido. Vou te provar.",
                "Se você acha que {assumption}, você está errado. E eu vou te mostrar por quê.",
                "Pare tudo. Você não vai acreditar como {entity} {action}. E a história real é melhor que a ficção.",
            ],
            HookStyle.REVEAL: [
                "O que você vai descobrir sobre {subject} vai mudar como você vê {domain}.",
                "{entity} tinha um segredo. Só que esse segredo não era o que todo mundo pensava.",
                "Em {year}, aconteceu algo que {domain} nunca contou. Até hoje.",
            ],
            HookStyle.STORY: [
                "Tudo começou quando {entity} decidiu {action}. Ninguém imaginava o que viria depois.",
                "Em {year}, {entity} estava prestes a {crisis}. O que aconteceu nos próximos {time_period} definiu {domain}.",
                "{entity} tinha {age} anos quando {event}. A decisão que tomou naquele dia mudou tudo.",
            ],
        }
        
        self._tone_instructions_map = {
            HookStyle.DEFAULT: (
                "TONE: Curiosity-driven, narrative, engaging. "
                "Write like a storyteller uncovering a fascinating truth.\n"
            ),
            HookStyle.CURIOSITY_GAP: (
                "HOOK STYLE: Curiosity Gap — Create information asymmetry. "
                "The viewer knows something is important but not what it is. "
                "Open a loop in the first sentence that the rest of the script closes. "
                "Use specific facts, numbers, or names to create intrigue.\n"
            ),
            HookStyle.COUNTER_NARRATIVE: (
                "HOOK STYLE: Counter-Narrative — Challenge what the viewer thinks they know. "
                "Create cognitive dissonance by presenting evidence against common beliefs. "
                "Maintain a confident, evidence-based tone throughout.\n"
            ),
            HookStyle.CONTROVERSY: (
                "HOOK STYLE: Controversy — Take a strong stance that demands engagement. "
                "Present an unpopular opinion backed by data. "
                "Maintain conviction without being aggressive. Invite disagreement through evidence.\n"
            ),
            HookStyle.CHALLENGE: (
                "HOOK STYLE: Challenge — Direct challenge to the viewer's knowledge. "
                "Dare them to learn something new. Use confident, confrontational but friendly tone. "
                "Prove your point with specific facts.\n"
            ),
            HookStyle.REVEAL: (
                "HOOK STYLE: Reveal — Promise a shocking fact upfront. "
                "Build anticipation for the revelation. Deliver the payoff with impact. "
                "Keep tension high until the reveal.\n"
            ),
            HookStyle.STORY: (
                "HOOK STYLE: Story — Open a narrative arc with character and conflict. "
                "Set up a protagonist, a challenge, and promise resolution. "
                "Maintain narrative tension throughout. Use vivid details.\n"
            ),
        }

    def generate_hook(
        self,
        subject: str,
        style: HookStyle = HookStyle.DEFAULT,
        context: str = ""
    ) -> dict:
        """
        Generate an optimized hook based on style and subject.
        
        Args:
            subject: The main topic/subject of the video
            style: The hook style to use
            context: Additional context from web search or metadata
            
        Returns:
            dict with keys:
                - hook: str (the first 1-2 sentences)
                - pattern_interrupt: str (unexpected statement)
                - curiosity_gap: str (what the viewer needs to know)
                - tone_instructions: str (enhanced tone block for script generation)
        """
        if style == HookStyle.DEFAULT:
            return self._generate_default_hook(subject, context)
        
        # For non-default styles, generate hook using AI
        hook_data = self._generate_hook_with_ai(subject, style, context)
        return hook_data
    
    def _generate_default_hook(self, subject: str, context: str = "") -> dict:
        """Generate default hook (backward compatible with existing behavior)."""
        hook = f"Tudo começou quando {subject} se tornou um fenômeno. Ninguém imaginava o que viria depois."
        
        return {
            "hook": hook,
            "pattern_interrupt": "Mas a história real é completamente diferente do que você imagina.",
            "curiosity_gap": "O que aconteceu nos bastidores mudou tudo.",
            "tone_instructions": self.get_tone_instructions(HookStyle.DEFAULT),
        }
    
    def _generate_hook_with_ai(self, subject: str, style: HookStyle, context: str = "") -> dict:
        """Generate hook using Pollinations API with template guidance."""
        try:
            from .config import API_KEY, API_URL, MODEL_TEXT
            import requests
            
            templates = self._templates.get(style, [])
            tone_instruction = self._tone_instructions_map.get(style, "")
            
            system_prompt = (
                f"You are a copywriting expert specializing in YouTube Shorts hooks for Brazilian Portuguese audience. "
                f"Generate a compelling hook using these principles:\n"
                f"{tone_instruction}\n"
                f"AVAILABLE TEMPLATES (adapt them creatively):\n"
                + "\n".join(f"- {t}" for t in templates) + "\n"
                f"RULES:\n"
                f"- Hook must be 1-2 sentences maximum\n"
                f"- Include specific facts, names, years, or numbers\n"
                f"- NEVER start with 'Você sabia', 'Prepare-se', or rhetorical questions\n"
                f"- NO hyperboles like 'shocking truth', 'you won't believe'\n"
                f"- Output ONLY valid JSON with keys: hook, pattern_interrupt, curiosity_gap\n"
            )
            
            user_prompt = (
                f"Subject: {subject}\n"
                f"Context: {context[:500] if context else 'No additional context'}\n"
                f"Style: {style.value}\n"
                f"Generate a hook in Brazilian Portuguese following the templates and rules above."
            )
            
            payload = {
                "model": MODEL_TEXT,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "response_format": {"type": "json_object"},
            }
            
            headers = {
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(API_URL, json=payload, headers=headers, timeout=10)
            response.raise_for_status()
            
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            import json as json_module
            hook_data = json_module.loads(content)
            
            # Ensure all required keys exist
            hook_data.setdefault("hook", f"{subject} tem uma história que poucos conhecem.")
            hook_data.setdefault("pattern_interrupt", "Mas a realidade é diferente do que todos pensam.")
            hook_data.setdefault("curiosity_gap", "O que realmente aconteceu vai te surpreender.")
            hook_data["tone_instructions"] = self.get_tone_instructions(style)
            
            return hook_data
            
        except Exception as e:
            # Fallback to template-based generation
            return self._generate_hook_from_template(subject, style)
    
    def _generate_hook_from_template(self, subject: str, style: HookStyle) -> dict:
        """Fallback: Generate hook from template without API call."""
        templates = self._templates.get(style, self._templates[HookStyle.CURIOSITY_GAP])
        template = templates[0]  # Use first template as base
        
        # Simple variable substitution for fallback
        hook = template.format(
            year="2023",
            entity=subject,
            action="fez história",
            consequence="tudo mudasse",
            domain="o cenário",
            number="3",
            time_period="anos",
            event="o início",
            reveal="revelou a verdade",
            subject=subject,
            common_belief="uma coisa sobre isso",
            common_assumption="o que todos pensam",
            surprising_truth="diferente",
            specific_data="os números",
            controversial_take=f"{subject} é mal compreendido",
            lie="todos acreditavam",
            truth="a realidade é outra",
            surprising_fact="isso sobre ele",
            assumption="você entende o assunto",
            age="20",
            crisis="um grande desafio",
        )
        
        return {
            "hook": hook,
            "pattern_interrupt": "Mas a história real é completamente diferente.",
            "curiosity_gap": "O que aconteceu nos bastidores mudou tudo.",
            "tone_instructions": self.get_tone_instructions(style),
        }
    
    def get_tone_instructions(self, style: HookStyle) -> str:
        """Returns enhanced tone instructions for the script generator."""
        return self._tone_instructions_map.get(
            style, 
            self._tone_instructions_map[HookStyle.DEFAULT]
        )
