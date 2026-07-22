"""Web-oriented tools: web_search, fetch_url, web_search_fetch.

All three are **read-only, network-only** tools - they never touch the
local filesystem, so they bypass the workspace-path validation used by
the mutating file tools.  They use only ``requests`` (already a project
dependency) plus the standard library, so no extra packages are needed.

* ``web_search``      - DuckDuckGo HTML search, returns titles/URLs/snippets.
* ``fetch_url``        - HTTP GET a single page, strip HTML, return readable text.
* ``web_search_fetch`` - Run a search, then fetch and concatenate the top pages.

Security / robustness notes
---------------------------
* Only ``http`` and ``https`` schemes are accepted (no ``file://``, ``ftp`` …).
* Responses are streamed and capped at ``MAX_BYTES`` to bound memory use.
* A generous timeout is set on every request.
* A normal browser ``User-Agent`` is sent so most sites return real content.
"""

from __future__ import annotations

import html
import re
from urllib.parse import parse_qs, urlparse

import requests

from agentThree.logging_setup import logger
from agentThree.tools_registry import tool

# --------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------- #

#: DuckDuckGo HTML endpoint - no API key, no JSON, just parseable HTML.
DDG_URL: str = "https://html.duckduckgo.com/html/"

#: Browser-ish UA so sites don't return minimal/blank pages.
HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

#: Hard cap on how many bytes we read from a single HTTP response.
MAX_BYTES: int = 2_000_000

#: Per-request timeout (seconds): (connect, read).
REQUEST_TIMEOUT: int = 15

#: Only these URL schemes are accepted by the fetch tools.
_ALLOWED_SCHEMES: tuple[str, ...] = ("http", "https")


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #

def _validate_url(url: str) -> str:
    """Return ``url`` if it is a safe http(s) URL, else raise ValueError."""
    if not isinstance(url, str) or not url.strip():
        raise ValueError("URL is required and must be a non-empty string")
    parsed = urlparse(url.strip())
    if parsed.scheme.lower() not in _ALLOWED_SCHEMES:
        raise ValueError(
            f"Refused: only {('/'.join(_ALLOWED_SCHEMES))} URLs are allowed, "
            f"got scheme {parsed.scheme!r}."
        )
    if not parsed.netloc:
        raise ValueError(f"Refused: URL {url!r} has no host.")
    return url.strip()


def _http_get(url: str) -> tuple[str, str]:
    """GET ``url`` with a byte cap and return ``(text, content_type)``.

    Reads at most :data:`MAX_BYTES` bytes.  Raises ``requests`` exceptions
    on failure - callers are expected to wrap them.
    """
    resp = requests.get(
        url, headers=HEADERS, timeout=REQUEST_TIMEOUT, stream=True,
        allow_redirects=True,
    )
    resp.raise_for_status()
    chunks: list[bytes] = []
    total = 0
    for chunk in resp.iter_content(chunk_size=8192):
        if not chunk:
            continue
        chunks.append(chunk)
        total += len(chunk)
        if total >= MAX_BYTES:
            break
    raw = b"".join(chunks)
    # Try to decode using the charset the server declared, fall back to utf-8
    # (errors replaced so a few bad bytes never abort the whole page).
    charset = resp.encoding or "utf-8"
    text = raw.decode(charset, errors="replace")
    content_type = resp.headers.get("Content-Type", "")
    logger.debug("Fetched %s -> %d bytes, content-type=%s", url, total, content_type)
    return text, content_type


_TAG_STRIP_RE = re.compile(r"(?s)<[^>]+>")
_SCRIPT_STYLE_RE = re.compile(r"(?is)<(script|style|noscript|template)[^>]*>.*?</\1>")
_BR_RE = re.compile(r"(?i)<br\s*/?>")
_BLOCK_RE = re.compile(
    r"(?i)</(p|div|li|h[1-6]|tr|table|section|article|header|footer|blockquote)>"
)
_WHITESPACE_RE = re.compile(r"[ \t]+")


def _html_to_text(raw_html: str) -> str:
    """Best-effort conversion of an HTML string to readable plain text."""
    # Drop script/style/noscript/template blocks entirely.
    text = _SCRIPT_STYLE_RE.sub(" ", raw_html)
    # <br> -> newline, closing block tags -> newline.
    text = _BR_RE.sub("\n", text)
    text = _BLOCK_RE.sub("\n", text)
    # Strip every remaining tag.
    text = _TAG_STRIP_RE.sub("", text)
    # Unescape entities (&amp; -> &).
    text = html.unescape(text)
    # Collapse runs of spaces/tabs, but keep line structure.
    lines: list[str] = []
    prev_blank = False
    for line in text.splitlines():
        line = _WHITESPACE_RE.sub(" ", line).strip()
        if not line:
            if not prev_blank and lines:
                lines.append("")
                prev_blank = True
        else:
            lines.append(line)
            prev_blank = False
    return "\n".join(lines).strip()


def _resolve_ddg_link(href: str) -> str:
    """DuckDuckGo HTML wraps real URLs in a redirect (``?uddg=<encoded>``).

    Return the underlying URL if we can decode it, otherwise the raw href.
    """
    parsed = urlparse(href)
    qs = parse_qs(parsed.query)
    if "uddg" in qs and qs["uddg"]:
        return qs["uddg"][0]
    return href


_RESULT_A_RE = re.compile(
    r'<a[^>]+class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', re.S
)
_SNIPPET_RE = re.compile(
    r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>', re.S
)


def _ddg_search(query: str, max_results: int) -> list[dict[str, str]]:
    """Run a DuckDuckGo HTML search and return a list of result dicts.

    Each dict has keys: ``title``, ``url``, ``snippet``.
    """
    if max_results < 1:
        max_results = 1
    if max_results > 20:
        max_results = 20

    resp = requests.post(
        DDG_URL,
        data={"q": query, "b": ""},  # 'b' disables JS redirect
        headers=HEADERS,
        timeout=REQUEST_TIMEOUT,
        allow_redirects=True,
    )
    resp.raise_for_status()
    page = resp.text

    titles = _RESULT_A_RE.findall(page)
    snippets_raw = _SNIPPET_RE.findall(page)

    results: list[dict[str, str]] = []
    for i, (raw_href, raw_title) in enumerate(titles):
        title = _html_to_text(raw_title)
        url = _resolve_ddg_link(raw_href)
        snippet = ""
        if i < len(snippets_raw):
            snippet = _html_to_text(snippets_raw[i])
        if not title and not url:
            continue
        results.append({"title": title, "url": url, "snippet": snippet})
        if len(results) >= max_results:
            break

    logger.debug("DDG search for %r returned %d usable results.", query, len(results))
    return results


# --------------------------------------------------------------------- #
# Tool parameter schemas (compact, with minimal descriptions)
# --------------------------------------------------------------------- #

_WEB_SEARCH_PARAMS: dict = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "The search query string."},
        "max_results": {"type": "integer", "description": "Maximum number of results to return (1-10). Default 5."},
    },
    "required": ["query"],
}

_FETCH_URL_PARAMS: dict = {
    "type": "object",
    "properties": {
        "url": {"type": "string", "description": "Absolute http(s) URL of the page to fetch."},
        "max_chars": {"type": "integer", "description": "Truncate returned text to this many characters. Default 20000."},
    },
    "required": ["url"],
}

_WEB_SEARCH_FETCH_PARAMS: dict = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "The search query string."},
        "max_results": {"type": "integer", "description": "How many of the top search results to fetch (1-5). Default 3."},
        "max_chars_per_page": {"type": "integer", "description": "Truncate each fetched page to this many characters. Default 4000."},
    },
    "required": ["query"],
}


# --------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------- #

@tool(name="web_search", description="Search the web (DuckDuckGo) and return titles, URLs and short snippets for the top results.", parameters=_WEB_SEARCH_PARAMS)
def web_search(query: str, max_results: int = 5) -> str:
    try:
        results = _ddg_search(query, int(max_results))
    except ValueError as exc:
        return f"Error: {exc}"
    except requests.exceptions.RequestException as exc:
        logger.warning("web_search request failed: %s", exc)
        return f"Error performing web search: {exc}"
    except Exception as exc:
        logger.exception("web_search unexpected error")
        return f"Error performing web search: {exc}"

    if not results:
        return f"No results found for {query!r}."

    lines = [f"Web search results for {query!r} ({len(results)}):"]
    for i, r in enumerate(results, 1):
        lines.append(f"\n[{i}] {r['title']}")
        lines.append(f"    URL: {r['url']}")
        if r["snippet"]:
            lines.append(f"    {r['snippet']}")
    return "\n".join(lines)


@tool(name="fetch_url", description="Fetch a web page (http/https) and return its text content with HTML tags stripped. Truncates to max_chars.", parameters=_FETCH_URL_PARAMS)
def fetch_url(url: str, max_chars: int = 20000) -> str:
    try:
        url = _validate_url(url)
    except ValueError as exc:
        return f"Error: {exc}"

    try:
        max_chars = int(max_chars) if max_chars is not None else 20000
    except (TypeError, ValueError):
        max_chars = 20000
    if max_chars < 1:
        max_chars = 20000

    try:
        raw, content_type = _http_get(url)
    except requests.exceptions.RequestException as exc:
        logger.warning("fetch_url request failed for %s: %s", url, exc)
        return f"Error fetching {url}: {exc}"
    except Exception as exc:
        logger.exception("fetch_url unexpected error for %s", url)
        return f"Error fetching {url}: {exc}"

    is_html = "html" in content_type.lower() or "<html" in raw[:2048].lower()
    if is_html:
        text = _html_to_text(raw)
    else:
        # Not HTML (e.g. JSON, plain text, CSV) - return as-is.
        text = raw

    if len(text) > max_chars:
        text = text[:max_chars] + f"\n... [truncated, content is {len(text)} chars]"
    if not text.strip():
        return f"Fetched {url} but the page had no extractable text content."
    return text


@tool(name="web_search_fetch", description="Search the web, then fetch and concatenate the text of the top results. Useful for getting substantive content, not just snippets.", parameters=_WEB_SEARCH_FETCH_PARAMS)
def web_search_fetch(
    query: str,
    max_results: int = 3,
    max_chars_per_page: int = 4000,
) -> str:
    try:
        max_results = int(max_results) if max_results is not None else 3
    except (TypeError, ValueError):
        max_results = 3
    if max_results < 1:
        max_results = 1
    if max_results > 5:
        max_results = 5

    try:
        max_chars_per_page = int(max_chars_per_page) if max_chars_per_page is not None else 4000
    except (TypeError, ValueError):
        max_chars_per_page = 4000
    if max_chars_per_page < 100:
        max_chars_per_page = 4000

    try:
        results = _ddg_search(query, max_results)
    except requests.exceptions.RequestException as exc:
        logger.warning("web_search_fetch search failed: %s", exc)
        return f"Error performing web search: {exc}"
    except Exception as exc:
        logger.exception("web_search_fetch unexpected error during search")
        return f"Error performing web search: {exc}"

    if not results:
        return f"No results found for {query!r}."

    blocks: list[str] = [
        f"Web search + fetch for {query!r} ({len(results)} page(s)):",
    ]
    for i, r in enumerate(results, 1):
        header = f"\n=== [{i}] {r['title']} ===\nURL: {r['url']}\n"
        if not r["url"].lower().startswith(("http://", "https://")):
            blocks.append(header + "(skipped: non-http URL)")
            continue
        try:
            raw, content_type = _http_get(r["url"])
        except requests.exceptions.RequestException as exc:
            logger.debug("web_search_fetch: could not fetch %s: %s", r["url"], exc)
            blocks.append(header + f"(fetch failed: {exc})")
            continue
        except Exception as exc:
            logger.debug("web_search_fetch: unexpected error for %s: %s", r["url"], exc)
            blocks.append(header + f"(fetch error: {exc})")
            continue

        is_html = "html" in content_type.lower() or "<html" in raw[:2048].lower()
        text = _html_to_text(raw) if is_html else raw
        if len(text) > max_chars_per_page:
            text = text[:max_chars_per_page] + "\n... [truncated]"
        blocks.append(header + text)

    return "\n".join(blocks)