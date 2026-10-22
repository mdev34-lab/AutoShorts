"""Product-link video generation (ported from the shopee-short CLI).

Self-contained building blocks for `autoshorts new product <link>`:
link parsing, pronunciation respellings for TTS, guarded HTTP access and
fail-closed source checks. No Shopee page is ever fetched: the slug is
the only source of the product name.
"""
