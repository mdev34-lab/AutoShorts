from .logging_system import log


class WebSearcher:
    def __init__(self, max_results_per_query: int = 5):
        self.max_results_per_query = max_results_per_query

    @staticmethod
    def generate_queries(subject: str) -> list[str]:
        s = subject.strip().strip('"').strip("'")
        return [
            s,
            f"{s} hist\u00f3ria origem",
            f"{s} fatos importantes",
            f"{s} contexto hist\u00f3rico",
        ]

    def search(self, subject: str) -> list[dict] | None:
        if not subject:
            return None
        queries = self.generate_queries(subject)
        return self.search_with_queries(queries)

    def search_with_queries(self, queries: list[str]) -> list[dict] | None:
        if not queries:
            log("Web search skipped: no queries provided", "INFO")
            return None

        try:
            from ddgs import DDGS

            log(
                f"Web search starting: {len(queries)} queries",
                "INFO",
            )
            seen_urls: set[str] = set()
            results: list[dict] = []

            for i, q in enumerate(queries, 1):
                try:
                    log(f"Web search query {i}/{len(queries)}: '{q[:80]}'", "INFO")
                    batch = list(DDGS().text(q, max_results=self.max_results_per_query))
                    log(f"Web search query {i}: {len(batch)} raw results", "INFO")
                    new_count = 0
                    for r in batch:
                        url = r.get("href", "")
                        if url and url not in seen_urls:
                            seen_urls.add(url)
                            results.append(
                                {
                                    "title": r.get("title", ""),
                                    "url": url,
                                    "snippet": r.get("body", ""),
                                }
                            )
                            new_count += 1
                    log(
                        f"Web search query {i}: {new_count} new unique URLs",
                        "INFO",
                    )
                except Exception as e:
                    log(f"Web search query {i} failed: {e}", "WARNING")
                    continue

            if results:
                domains = set(
                    r["url"].split("/")[2] for r in results if "//" in r["url"]
                )
                log(
                    f"Web search complete: {len(results)} unique sources from "
                    f"{len(domains)} domains: {', '.join(sorted(domains))}",
                    "SUCCESS",
                )
            else:
                log("Web search returned no results", "WARNING")

            return results if results else None

        except ImportError:
            log("Web search unavailable: ddgs library not installed", "WARNING")
            return None
        except Exception as e:
            log(f"Web search failed: {e}", "WARNING")
            return None

    @staticmethod
    def format_context(results: list[dict]) -> str:
        if not results:
            return ""
        log(f"Formatting {len(results)} web sources for prompt injection", "INFO")
        lines = [
            "FONTES DA WEB (use como base para sua hist\u00f3ria, citando fatos espec\u00edficos):",
            "\u2500" * 60,
        ]
        for i, r in enumerate(results, 1):
            snippet = r["snippet"][:200]
            lines.append(f"[{i}] {r['title']}")
            lines.append(f"    Fonte: {r['url']}")
            lines.append(f"    {snippet}")
            lines.append("")
        lines.append("\u2500" * 60)
        return "\n".join(lines)
