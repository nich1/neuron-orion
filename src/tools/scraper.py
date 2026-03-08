"""Web search (DuckDuckGo) and scraping (httpx + BeautifulSoup) tools."""
from __future__ import annotations

import asyncio
import json
import logging
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from pydantic_ai import RunContext

from ..deps import AgentDeps

log = logging.getLogger(__name__)

_SEARCH_MAX_RESULTS = 6
_SCRAPE_MAX_CHARS = 6000
_STRIP_TAGS = {"script", "style", "nav", "footer", "header", "noscript", "svg", "img"}


def _sync_search(query: str, max_results: int) -> list[dict[str, str]]:
    return DDGS().text(query, max_results=max_results)


async def web_search_impl(
    ctx: RunContext[AgentDeps],
    query: str,
) -> str:
    """Search the web via DuckDuckGo. Returns a JSON array of {title, url, snippet} results."""
    try:
        raw = await asyncio.to_thread(_sync_search, query, _SEARCH_MAX_RESULTS)
        results = [
            {"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")}
            for r in raw
        ]
        return json.dumps(results, indent=2) if results else f"No results found for: {query}"
    except Exception:
        log.exception("DuckDuckGo search failed for: %s", query)
        return f"Search failed for: {query}"


async def web_scrape_impl(
    ctx: RunContext[AgentDeps],
    url: str,
) -> str:
    """Fetch a URL and return its main text content (HTML stripped)."""
    try:
        resp = await ctx.deps.http_client.get(
            url, timeout=30.0, follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; NichNeuron/0.1)"},
        )
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup.find_all(_STRIP_TAGS):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)
        if len(text) > _SCRAPE_MAX_CHARS:
            text = text[:_SCRAPE_MAX_CHARS] + "\n…[truncated]"

        return text if text else "Page returned no extractable text."
    except Exception:
        log.exception("Scrape failed: %s", url)
        return f"Failed to scrape {url}"
