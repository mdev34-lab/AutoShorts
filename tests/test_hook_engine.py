"""Test Hook Engine functionality."""

import pytest

from autoshorts.modules.hook_engine import HookEngine, HookStyle


class TestHookEngine:
    """Test cases for HookEngine class."""

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

    def test_default_hook_generation(self):
        """DEFAULT style generates valid hook."""
        engine = HookEngine()
        result = engine.generate_hook("Test subject", HookStyle.DEFAULT)
        assert isinstance(result["hook"], str)
        assert len(result["hook"]) > 0

    def test_all_styles_generate_valid_hooks(self):
        """All hook styles generate valid hooks with required keys."""
        engine = HookEngine()
        for style in HookStyle:
            result = engine.generate_hook("Flamengo 1981", style)
            assert "hook" in result
            assert "pattern_interrupt" in result
            assert "curiosity_gap" in result
            assert "tone_instructions" in result
            assert isinstance(result["hook"], str)
            assert len(result["hook"]) > 0

    def test_hook_templates_are_used(self):
        """Verify templates are being used in fallback generation."""
        engine = HookEngine()
        # Test curiosity gap template
        result = engine._generate_hook_from_template("Teste", HookStyle.CURIOSITY_GAP)
        assert "hook" in result
        assert isinstance(result["hook"], str)
        
        # Test counter narrative template
        result = engine._generate_hook_from_template("Teste", HookStyle.COUNTER_NARRATIVE)
        assert "hook" in result
        
        # Test controversy template
        result = engine._generate_hook_from_template("Teste", HookStyle.CONTROVERSY)
        assert "hook" in result

    def test_tone_instructions_are_style_specific(self):
        """Each hook style has distinct tone instructions."""
        engine = HookEngine()
        
        default_instructions = engine.get_tone_instructions(HookStyle.DEFAULT)
        curiosity_instructions = engine.get_tone_instructions(HookStyle.CURIOSITY_GAP)
        controversy_instructions = engine.get_tone_instructions(HookStyle.CONTROVERSY)
        
        # Each should be different
        assert default_instructions != curiosity_instructions
        assert curiosity_instructions != controversy_instructions
        
        # Curiosity should mention "curiosity" or "asymmetry"
        assert "curiosity" in curiosity_instructions.lower() or "asymmetry" in curiosity_instructions.lower()
        
        # Controversy should mention "controversy" or "stance"
        assert "controversy" in controversy_instructions.lower() or "stance" in controversy_instructions.lower()

    def test_fallback_generation_works(self):
        """Fallback template-based generation works without API."""
        engine = HookEngine()
        
        for style in HookStyle:
            result = engine._generate_hook_from_template("Subject Test", style)
            assert "hook" in result
            assert len(result["hook"]) > 0
            assert "tone_instructions" in result
