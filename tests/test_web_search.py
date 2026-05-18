from unittest.mock import MagicMock, patch

from autoshorts.modules.web_search import WebSearcher


class TestWebSearcher:
    def setup_method(self):
        self.searcher = WebSearcher(max_results_per_query=3)

    def test_generate_queries(self):
        queries = WebSearcher.generate_queries("Guerra Fria")
        assert len(queries) == 4
        assert "Guerra Fria" in queries
        assert all("Guerra Fria" in q for q in queries)

    def test_generate_queries_strips_quotes(self):
        queries = WebSearcher.generate_queries('"test subject"')
        assert "test subject" in queries[0]

    def test_search_empty_subject(self):
        assert self.searcher.search("") is None
        assert self.searcher.search(None) is None

    @patch("ddgs.DDGS")
    def test_search_success(self, mock_ddgs_class):
        mock_ddgs = MagicMock()
        mock_ddgs_class.return_value = mock_ddgs
        mock_ddgs.text.return_value = [
            {"href": "https://example.com/1", "title": "Result 1", "body": "Snippet one"},
            {"href": "https://example.com/2", "title": "Result 2", "body": "Snippet two"},
        ]

        results = self.searcher.search("test subject")

        assert results is not None
        assert len(results) == 2
        assert results[0]["title"] == "Result 1"
        assert results[0]["url"] == "https://example.com/1"

    @patch("ddgs.DDGS")
    def test_search_deduplicates_by_url(self, mock_ddgs_class):
        mock_ddgs = MagicMock()
        mock_ddgs_class.return_value = mock_ddgs
        mock_ddgs.text.return_value = [
            {"href": "https://example.com/1", "title": "Result 1", "body": "Snippet"},
            {"href": "https://example.com/1", "title": "Result 1 dup", "body": "Duplicate"},
        ]

        results = self.searcher.search("test subject")

        assert results is not None
        assert len(results) == 1

    @patch("ddgs.DDGS")
    def test_search_skips_missing_href(self, mock_ddgs_class):
        mock_ddgs = MagicMock()
        mock_ddgs_class.return_value = mock_ddgs
        mock_ddgs.text.return_value = [
            {"title": "No URL", "body": "Snippet"},
            {"href": "https://example.com/1", "title": "Has URL", "body": "Snippet"},
        ]

        results = self.searcher.search("test subject")

        assert results is not None
        assert len(results) == 1
        assert results[0]["title"] == "Has URL"

    @patch("ddgs.DDGS")
    def test_search_ddg_error_per_query_continues(self, mock_ddgs_class):
        mock_ddgs = MagicMock()
        mock_ddgs_class.return_value = mock_ddgs
        mock_ddgs.text.side_effect = [
            [{"href": "https://example.com/1", "title": "R1", "body": "S1"}],
            Exception("Rate limited"),
            [{"href": "https://example.com/2", "title": "R2", "body": "S2"}],
            [{"href": "https://example.com/3", "title": "R3", "body": "S3"}],
        ]

        results = self.searcher.search("test subject")

        assert results is not None
        assert len(results) == 3

    @patch("ddgs.DDGS")
    def test_search_all_queries_fail(self, mock_ddgs_class):
        mock_ddgs = MagicMock()
        mock_ddgs_class.return_value = mock_ddgs
        mock_ddgs.text.side_effect = Exception("Network error")

        results = self.searcher.search("test subject")
        assert results is None

    def test_format_context(self):
        results = [
            {"title": "Guerra Fria", "url": "https://ex.com/gf", "snippet": "A Guerra Fria foi um per\u00edodo..."},
            {"title": "Origem", "url": "https://ex.com/origem", "snippet": "Tudo comecou em 1945..."},
        ]
        formatted = WebSearcher.format_context(results)

        assert "FONTES DA WEB" in formatted
        assert "Guerra Fria" in formatted
        assert "https://ex.com/gf" in formatted
        assert "[1]" in formatted
        assert "[2]" in formatted

    def test_format_context_empty(self):
        assert WebSearcher.format_context([]) == ""
        assert WebSearcher.format_context(None) == ""
