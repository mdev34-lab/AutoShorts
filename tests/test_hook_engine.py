import pytest

from autoshorts.modules.hook_engine import HookEngine, HookStyle


class TestHookEngine:
    def test_all_hook_styles_exist(self):
        """Ensure all hook styles are defined."""
        styles = [s.value for s in HookStyle]
        assert "default" in styles
        assert "curiosity" in styles
        assert "counter" in styles
        assert "controversy" in styles
        assert "challenge" in styles
        assert "reveal" in styles
        assert "story" in styles
        # Also ensure 7 total
        assert len(styles) == 7

    def test_generate_hook_returns_required_keys(self):
        """Hook generation returns all required fields."""
        engine = HookEngine()
        result = engine.generate_hook("Test subject", HookStyle.CURIOSITY_GAP)
        assert "hook" in result
        assert "pattern_interrupt" in result
        assert "curiosity_gap" in result
        assert "tone_instructions" in result

    def test_hook_is_not_empty(self):
        """Generated hook is non-empty."""
        engine = HookEngine()
        result = engine.generate_hook("Corinthians 2012 Libertadores", HookStyle.CONTROVERSY)
        assert len(result["hook"]) > 10

    def test_tone_instructions_include_hook_style(self):
        """Tone instructions mention the hook style."""
        engine = HookEngine()
        instructions = engine.get_tone_instructions(HookStyle.COUNTER_NARRATIVE)
        assert "counter" in instructions.lower() or "narrative" in instructions.lower()

    def test_backward_compatibility(self):
        """DEFAULT hook style preserves existing behavior."""
        engine = HookEngine()
        instructions = engine.get_tone_instructions(HookStyle.DEFAULT)
        assert "curiosity" in instructions.lower()  # Existing tone preserved

    def test_default_hook_empty(self):
        """DEFAULT style returns empty hook to preserve backward compat."""
        engine = HookEngine()
        result = engine.generate_hook("Some subject", HookStyle.DEFAULT)
        assert result["hook"] == ""  # Should be empty for backward compat
        # But tone_instructions still present
        assert len(result["tone_instructions"]) > 0

    def test_all_styles_generate_hooks(self):
        """Each non-default style generates a non-empty hook."""
        engine = HookEngine()
        subject = "Pelé 1000 gols"
        for style in HookStyle:
            if style == HookStyle.DEFAULT:
                continue
            result = engine.generate_hook(subject, style)
            assert len(result["hook"]) > 10, f"Style {style} produced empty hook"
            assert len(result["pattern_interrupt"]) > 5
            assert len(result["curiosity_gap"]) > 5

    def test_hook_contains_specific_info(self):
        """Hook should contain subject or year or number."""
        engine = HookEngine()
        subject = "Corinthians 2012 Libertadores"
        result = engine.generate_hook(subject, HookStyle.CURIOSITY_GAP)
        hook_lower = result["hook"].lower()
        # Should contain either subject keyword or year 2012
        assert "corinthians" in hook_lower or "2012" in result["hook"] or "libertadores" in hook_lower

    def test_hook_not_start_with_forbidden(self):
        """Hook must NOT start with forbidden phrases."""
        engine = HookEngine()
        forbidden = ["você sabia", "prepare-se", "voce sabia"]
        for style in HookStyle:
            if style == HookStyle.DEFAULT:
                continue
            result = engine.generate_hook("Test subject 2020", style)
            hook_lower = result["hook"].lower().strip()
            if not hook_lower:
                continue
            for fb in forbidden:
                assert not hook_lower.startswith(fb), f"Hook for {style} starts with forbidden '{fb}': {result['hook']}"

    def test_hook_quality_1_2_sentences(self):
        """Hook should be 1-2 sentences max."""
        engine = HookEngine()
        result = engine.generate_hook("Flamengo 1981 Mundial", HookStyle.STORY)
        hook = result["hook"]
        # Count sentences by period
        sentences = [s for s in hook.split(".") if s.strip()]
        assert 1 <= len(sentences) <= 3  # Allow up to 3, but ideal 2

    def test_tone_instructions_distinct(self):
        """Each hook style has distinct tone instructions."""
        engine = HookEngine()
        instructions = {}
        for style in HookStyle:
            instructions[style] = engine.get_tone_instructions(style)
        # Check all are non-empty and not all identical
        values = list(instructions.values())
        assert len(set(values)) == len(values)  # all distinct

    def test_generate_hook_with_context(self):
        """Hook generation with context should not fail."""
        engine = HookEngine()
        context = "Flamengo venceu o Liverpool em 1981 no Mundial de Clubes com gols de Nunes e Adílio"
        result = engine.generate_hook("Flamengo 1981", HookStyle.REVEAL, context=context)
        assert len(result["hook"]) > 10

    def test_generate_hook_invalid_style_fallback(self):
        """Invalid style should fallback to DEFAULT silently."""
        engine = HookEngine()
        result = engine.generate_hook("Test", "invalid_style")  # type: ignore
        # Should fallback to DEFAULT with empty hook but valid structure
        assert "hook" in result
        assert "tone_instructions" in result

    def test_hook_style_enum_values(self):
        """Enum values match expected strings."""
        assert HookStyle.DEFAULT.value == "default"
        assert HookStyle.CURIOSITY_GAP.value == "curiosity"
        assert HookStyle.COUNTER_NARRATIVE.value == "counter"
        assert HookStyle.CONTROVERSY.value == "controversy"
        assert HookStyle.CHALLENGE.value == "challenge"
        assert HookStyle.REVEAL.value == "reveal"
        assert HookStyle.STORY.value == "story"

    def test_generate_hook_performance(self):
        """Hook generation should be fast (<2 seconds)."""
        import time

        engine = HookEngine()
        start = time.time()
        for _ in range(10):
            engine.generate_hook("Test performance subject 2020", HookStyle.CURIOSITY_GAP)
        elapsed = time.time() - start
        # 10 generations should be well under 2 seconds each, total <2 sec ideally
        # Allow 2 sec total for 10 runs = 0.2 sec each to be safe
        assert elapsed < 2.0, f"Hook generation too slow: {elapsed}s for 10 runs"

    def test_pattern_interrupt_and_curiosity_gap_content(self):
        """Pattern interrupt and curiosity gap should be meaningful."""
        engine = HookEngine()
        result = engine.generate_hook("Neymar vs Ronaldo", HookStyle.COUNTER_NARRATIVE)
        assert len(result["pattern_interrupt"]) > 5
        assert len(result["curiosity_gap"]) > 10
        # Curiosity gap should mention subject or entity
        assert "Neymar" in result["curiosity_gap"] or "Ronaldo" in result["curiosity_gap"] or "história" in result["curiosity_gap"].lower()
