# Structured portrait evidence prototype

This prototype extracts authored source evidence. It is deliberately NOT called
by the image searcher and does not admit any additional web images.

## Source contract

- Independently resolved, unambiguous exact Wikidata entity ID, not name overlap.
- Commons file page ID must bind the supplied MediaInfo entity ID.
- A non-deprecated exact depicts(P180) statement, not generic person or category.
- HTTPS Commons file page, Wikimedia original-image URL, reusable license from
  the existing allowlist, creator metadata retained.
- Any explicit date, event or team constraint rejects this portrait-only route.

The caller must extract those constraints; an empty list is not proof that an
arbitrary prompt is unconstrained. The extractor does not resolve names, inspect
pixels, identify faces, validate source authors, or satisfy license obligations
by itself. Multiple depicted people and misleading authored metadata remain
possible. Output records this uncertainty explicitly.

## Live sample and limits

On October24,2026, the file
https://commons.wikimedia.org/wiki/File:Clarence_Seedorf_2021.jpg
resolved through the public API to file111425118 / MediaInfoM111425118 /
depictsQ484909. The entity source
https://www.wikidata.org/wiki/Special:EntityData/Q484909.json
labelsQ484909 ClarenceSeedorf. The file metadata names its creator and
CCBY-SA3.0 license. Its2021date does NOT establish any2012match context.
Downloading that original image returnedHTTP429; no retry or visual-validation
claim was made. Tests use a minimal source-shaped fixture, not an image fixture.

Official modeling/API references:
- https://commons.wikimedia.org/wiki/Commons:Structured_data/Modeling/Depiction
- https://commons.wikimedia.org/wiki/Commons:Structured_data/Lua

## Before production admission

1. Independently resolve identity and all contextual constraints from the
   pipeline's actual narration/image request; fail closed on ambiguity.
2. Validate representative real pixels against an independent reference and
   challenge wrong-person, artwork, crowd, outdated-event and mislabeled sources.
3. Review the evidence contract without redefining source metadata as visual
   verification. Preserve current license, watermark, OCR, cache and credit gates.
4. Only then propose a separate admission change. Until then specific web-image
   requests keep the existing safe fallback behavior.
