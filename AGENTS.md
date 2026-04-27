<guidance>
You maintain project-specific memory for AutoShorts.
Record recurring patterns, architectural decisions, and lessons that improve consistency and quality.
Focus on decisions that affect multiple parts of the system.
Keep entries concise and practical. Avoid duplication.
Do not blindly follow entries — apply judgment based on context.
</guidance>

<memory>
- Video background logic must be unified: blurred YouTube bg is the base, AI images are overlays on top (not replacements)
- Use VideoCompositor for all video processing (yt_summarizer + experimental share same bg logic)
- PIL for text measurement is 10-50x faster than TextClip - always use lru_cache
- Use VideoBackgroundManager for YouTube downloads instead of duplicating logic
- DRY: modularize reusable code in modules/
- Catch KeyboardInterrupt in CLI entry points - print "\nCancelled." and exit with code 130 using `raise SystemExit(130) from None`
</memory>