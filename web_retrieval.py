from __future__ import annotations

from dataclasses import dataclass
from typing import List
from urllib.parse import quote_plus
import html
import re
import xml.etree.ElementTree as ET

import requests
from langchain_core.documents import Document
from nlp_utils import normalize_spaces as _normalize_spaces, normalize_tokens as _normalize_tokens


@dataclass
class WebSearchResult:
    provider: str
    title: str
    snippet: str
    url: str
    score: float

    def to_document(self) -> Document:
        return Document(
            page_content=f"{self.title}. {self.snippet}".strip(),
            metadata={
                "source": f"{self.provider}: {self.title}",
                "url": self.url,
                "provider": self.provider,
            },
        )


class WebSearchClient:
    def __init__(self, timeout: int = 12):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "ThesisRAG/1.0 (educational research app)",
                "Accept-Language": "uk,en;q=0.8",
            }
        )

    @staticmethod
    def _normalize_spaces(text: str) -> str:
        return _normalize_spaces(text)

    @staticmethod
    def _normalize_tokens(text: str) -> list[str]:
        return _normalize_tokens(text)

    def _expand_query(self, query: str) -> str:
        normalized = self._normalize_spaces(query)
        lowered = normalized.lower()
        replacements = {
            "ші": "штучний інтелект",
            "ai": "artificial intelligence штучний інтелект",
            "llm": "large language model велика мовна модель",
        }
        for short_form, expanded in replacements.items():
            pattern = rf"(^|\W){re.escape(short_form)}($|\W)"
            if re.search(pattern, lowered):
                normalized = f"{normalized} {expanded}"
        return self._normalize_spaces(normalized)

    def _query_mode(self, query: str) -> str:
        lowered = query.lower().strip()
        if any(marker in lowered for marker in ("що таке", "визначення", " це")):
            return "definition"
        if lowered.startswith("скільки") or lowered.startswith("хто") or lowered.startswith("коли"):
            return "factoid"
        if lowered.startswith("які") or lowered.startswith("які є") or lowered.startswith("яка класиф"):
            return "list"
        return "general"

    def _query_focus(self, query: str) -> str:
        lowered = self._expand_query(query).lower().strip()
        for prefix in ("що таке ", "що таке", "визначення ", "визначення", "які є ", "які ", "яка ", "який ", "яке "):
            if lowered.startswith(prefix):
                lowered = lowered[len(prefix):].strip()
        return self._normalize_spaces(lowered.strip(" ?!.:,;"))

    def _provider_bias(self, provider: str, mode: str) -> float:
        provider_lower = provider.lower()
        if mode == "definition":
            if provider_lower == "wikipedia":
                return 0.32
            if provider_lower == "duckduckgo":
                return 0.12
            if provider_lower in {"openalex", "semantic scholar"}:
                return -0.08
            if provider_lower == "arxiv":
                return -0.18
        if mode == "factoid":
            if provider_lower == "wikipedia":
                return 0.34
            if provider_lower == "duckduckgo":
                return 0.18
            if provider_lower == "semantic scholar":
                return -0.18
            if provider_lower == "openalex":
                return -0.22
            if provider_lower == "arxiv":
                return -0.35
        if mode == "list":
            if provider_lower == "wikipedia":
                return 0.22
            if provider_lower == "duckduckgo":
                return 0.08
            if provider_lower == "arxiv":
                return -0.15
        return 0.0

    def _score_result(self, query: str, item: WebSearchResult) -> float:
        mode = self._query_mode(query)
        expanded_query = self._expand_query(query)
        focus_query = self._query_focus(query)
        query_tokens = {
            token for token in self._normalize_tokens(expanded_query)
            if len(token) > 1 and token not in {"що", "таке", "які", "є", "це", "про"}
        }
        haystack = f"{item.title} {item.snippet}".lower()
        haystack_tokens = set(self._normalize_tokens(haystack))
        overlap = len(query_tokens & haystack_tokens)
        lexical = overlap / max(len(query_tokens), 1)
        score = item.score + 0.55 * lexical + self._provider_bias(item.provider, mode)

        title_lower = item.title.lower()
        snippet_lower = item.snippet.lower()
        title_tokens = set(self._normalize_tokens(title_lower))
        focus_tokens = set(self._normalize_tokens(focus_query))
        if mode == "definition":
            if focus_query and title_lower == focus_query:
                score += 0.45
            elif focus_query and title_lower.startswith(focus_query + " ("):
                score += 0.34
            elif focus_query and title_lower.startswith(focus_query + " "):
                score -= 0.16
            if focus_tokens and title_tokens == focus_tokens:
                score += 0.22
            if any(token in title_lower for token in query_tokens):
                score += 0.18
            if "— це" in snippet_lower or " це " in snippet_lower or "назива" in snippet_lower:
                score += 0.18
            if any(marker in title_lower for marker in ("вільним стилем", "синхронне", "післяпологова", "велика ")):
                score -= 0.28
            if any(noisy in title_lower for noisy in ("effect", "ефект", "повстання", "роберт", "attack", "synthesis")):
                score -= 0.35
            if lexical < 0.34:
                score -= 0.35
        elif mode == "factoid":
            if focus_query and (title_lower == focus_query or title_lower.startswith(focus_query + " (")):
                score += 0.3
            if any(marker in snippet_lower for marker in ("чинний", "обирається", "рок", "літер", "букв", "народив", "президент")):
                score += 0.14
            if any(noisy in title_lower for noisy in ("analysis", "study", "performance", "detector", "experiment", "report")):
                score -= 0.3
            if lexical < 0.28:
                score -= 0.3
        elif mode == "list":
            if any(marker in snippet_lower for marker in ("типи", "види", "класиф", ",", ";")):
                score += 0.16
            if any(noisy in title_lower for noisy in ("method", "метод", "study", "дослідж", "analysis")):
                score -= 0.18
        else:
            if lexical < 0.2:
                score -= 0.2

        if item.provider.lower() in {"openalex", "arxiv", "semantic scholar"} and lexical < 0.45:
            score -= 0.22
        return score

    def _filter_and_rank_results(self, query: str, results: List[WebSearchResult]) -> List[WebSearchResult]:
        mode = self._query_mode(query)
        rescored: List[tuple[float, WebSearchResult]] = []
        for item in results:
            rescored.append((self._score_result(query, item), item))
        rescored.sort(key=lambda pair: pair[0], reverse=True)

        top_encyclopedic_exists = any(
            item.provider.lower() in {"wikipedia", "duckduckgo"} and score >= 0.45
            for score, item in rescored
        )

        filtered: List[WebSearchResult] = []
        for score, item in rescored:
            if mode == "definition" and score < 0.45:
                continue
            if mode == "factoid" and score < 0.4:
                continue
            if mode == "list" and score < 0.35:
                continue
            if mode in {"definition", "factoid"} and top_encyclopedic_exists and item.provider.lower() in {"openalex", "arxiv", "semantic scholar"} and score < 0.8:
                continue
            item.score = max(0.0, min(1.0, score))
            filtered.append(item)

        if mode in {"definition", "factoid"} and filtered:
            encyclopedic = [item for item in filtered if item.provider.lower() in {"wikipedia", "duckduckgo"}]
            academic = [item for item in filtered if item.provider.lower() not in {"wikipedia", "duckduckgo"}]
            prioritized = encyclopedic[:5]
            if not prioritized:
                prioritized = academic[:3]
            else:
                prioritized.extend(academic[:2])
            return prioritized

        if filtered:
            return filtered
        for score, item in rescored[:5]:
            item.score = max(0.0, min(1.0, score))
            filtered.append(item)
        return filtered

    def _safe_get_json(self, url: str, params: dict | None = None) -> dict | list | None:
        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except Exception:
            return None

    def _safe_get_text(self, url: str, params: dict | None = None) -> str | None:
        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            return response.text
        except Exception:
            return None

    def _fetch_wikipedia_extract(self, title: str) -> str:
        payload = self._safe_get_json(
            "https://uk.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "prop": "extracts",
                "exintro": 1,
                "explaintext": 1,
                "titles": title,
                "format": "json",
                "utf8": 1,
            },
        )
        if not isinstance(payload, dict):
            return ""
        pages = payload.get("query", {}).get("pages", {})
        if not isinstance(pages, dict):
            return ""
        for page in pages.values():
            extract = self._normalize_spaces(page.get("extract", ""))
            if extract:
                return extract[:1200]
        return ""

    def search_wikipedia(self, query: str, limit: int = 3) -> List[WebSearchResult]:
        query = self._expand_query(query)
        payload = self._safe_get_json(
            "https://uk.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "format": "json",
                "utf8": 1,
                "srlimit": limit,
            },
        )
        if not isinstance(payload, dict):
            return []
        items = payload.get("query", {}).get("search", [])
        results: List[WebSearchResult] = []
        for idx, item in enumerate(items[:limit]):
            title = self._normalize_spaces(item.get("title", ""))
            snippet = re.sub(r"<.*?>", " ", item.get("snippet", ""))
            snippet = html.unescape(self._normalize_spaces(snippet))
            extract = self._fetch_wikipedia_extract(title) if title else ""
            if extract:
                snippet = extract
            if not title or not snippet:
                continue
            url = f"https://uk.wikipedia.org/wiki/{quote_plus(title.replace(' ', '_'))}"
            score = max(0.0, 1.0 - idx * 0.12)
            results.append(WebSearchResult("Wikipedia", title, snippet, url, score))
        return results

    def search_openalex(self, query: str, limit: int = 3) -> List[WebSearchResult]:
        query = self._expand_query(query)
        payload = self._safe_get_json(
            "https://api.openalex.org/works",
            params={
                "search": query,
                "per-page": limit,
                "select": "display_name,doi,primary_location,abstract_inverted_index,publication_year",
            },
        )
        if not isinstance(payload, dict):
            return []
        results: List[WebSearchResult] = []
        for idx, item in enumerate(payload.get("results", [])[:limit]):
            title = self._normalize_spaces(item.get("display_name", ""))
            abstract_index = item.get("abstract_inverted_index") or {}
            snippet = self._reconstruct_inverted_abstract(abstract_index)
            if not snippet:
                snippet = f"Наукова праця за темою запиту. Рік: {item.get('publication_year', 'невідомо')}."
            primary_location = item.get("primary_location") or {}
            landing = primary_location.get("landing_page_url") or item.get("doi") or ""
            if landing and landing.startswith("10."):
                landing = f"https://doi.org/{landing}"
            if not title:
                continue
            score = max(0.0, 0.95 - idx * 0.12)
            results.append(WebSearchResult("OpenAlex", title, self._normalize_spaces(snippet), landing, score))
        return results

    @staticmethod
    def _reconstruct_inverted_abstract(abstract_index: dict) -> str:
        if not abstract_index:
            return ""
        positions: dict[int, str] = {}
        for word, indexes in abstract_index.items():
            for pos in indexes:
                positions[int(pos)] = word
        ordered = [positions[pos] for pos in sorted(positions)]
        return " ".join(ordered[:80])

    def search_arxiv(self, query: str, limit: int = 3) -> List[WebSearchResult]:
        query = self._expand_query(query)
        text = self._safe_get_text(
            "https://export.arxiv.org/api/query",
            params={
                "search_query": f"all:{query}",
                "start": 0,
                "max_results": limit,
            },
        )
        if not text:
            return []
        try:
            root = ET.fromstring(text)
        except ET.ParseError:
            return []
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        results: List[WebSearchResult] = []
        for idx, entry in enumerate(root.findall("atom:entry", ns)[:limit]):
            title = self._normalize_spaces(entry.findtext("atom:title", default="", namespaces=ns))
            summary = self._normalize_spaces(entry.findtext("atom:summary", default="", namespaces=ns))
            url = self._normalize_spaces(entry.findtext("atom:id", default="", namespaces=ns))
            if not title:
                continue
            score = max(0.0, 0.92 - idx * 0.12)
            results.append(WebSearchResult("arXiv", title, summary[:800], url, score))
        return results

    def search_semantic_scholar(self, query: str, limit: int = 3) -> List[WebSearchResult]:
        query = self._expand_query(query)
        payload = self._safe_get_json(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={
                "query": query,
                "limit": limit,
                "fields": "title,abstract,url,year",
            },
        )
        if not isinstance(payload, dict):
            return []
        results: List[WebSearchResult] = []
        for idx, item in enumerate(payload.get("data", [])[:limit]):
            title = self._normalize_spaces(item.get("title", ""))
            snippet = self._normalize_spaces(item.get("abstract", ""))
            if not snippet:
                snippet = f"Наукова публікація за темою запиту. Рік: {item.get('year', 'невідомо')}."
            url = item.get("url", "") or ""
            if not title:
                continue
            score = max(0.0, 0.9 - idx * 0.12)
            results.append(WebSearchResult("Semantic Scholar", title, snippet[:800], url, score))
        return results

    def search_duckduckgo(self, query: str, limit: int = 3) -> List[WebSearchResult]:
        query = self._expand_query(query)
        payload = self._safe_get_json(
            "https://api.duckduckgo.com/",
            params={
                "q": query,
                "format": "json",
                "no_html": 1,
                "skip_disambig": 1,
            },
        )
        if not isinstance(payload, dict):
            return []
        results: List[WebSearchResult] = []
        abstract = self._normalize_spaces(payload.get("AbstractText", ""))
        heading = self._normalize_spaces(payload.get("Heading", "")) or query
        abstract_url = payload.get("AbstractURL", "") or ""
        if abstract:
            results.append(WebSearchResult("DuckDuckGo", heading, abstract, abstract_url, 0.82))
        related = payload.get("RelatedTopics", [])
        for item in related:
            if len(results) >= limit:
                break
            if "Text" in item:
                text = self._normalize_spaces(item.get("Text", ""))
                url = item.get("FirstURL", "") or ""
                title = text.split(" - ", 1)[0] if " - " in text else heading
                if text:
                    results.append(WebSearchResult("DuckDuckGo", title, text, url, max(0.0, 0.76 - 0.08 * len(results))))
            elif "Topics" in item:
                for nested in item.get("Topics", []):
                    if len(results) >= limit:
                        break
                    text = self._normalize_spaces(nested.get("Text", ""))
                    url = nested.get("FirstURL", "") or ""
                    title = text.split(" - ", 1)[0] if " - " in text else heading
                    if text:
                        results.append(WebSearchResult("DuckDuckGo", title, text, url, max(0.0, 0.76 - 0.08 * len(results))))
        return results[:limit]

    def search_provider(self, provider: str, query: str, limit: int = 3) -> List[WebSearchResult]:
        normalized = provider.lower()
        if normalized == "wikipedia":
            return self.search_wikipedia(query, limit=limit)
        if normalized == "openalex":
            return self.search_openalex(query, limit=limit)
        if normalized == "arxiv":
            return self.search_arxiv(query, limit=limit)
        if normalized == "semantic scholar":
            return self.search_semantic_scholar(query, limit=limit)
        if normalized == "duckduckgo":
            return self.search_duckduckgo(query, limit=limit)
        return []

    def search_many(self, query: str, providers: List[str], limit_per_provider: int = 3) -> List[WebSearchResult]:
        combined: List[WebSearchResult] = []
        seen = set()
        for provider in providers:
            for item in self.search_provider(provider, query, limit=limit_per_provider):
                key = (item.provider, item.title, item.url)
                if key in seen:
                    continue
                seen.add(key)
                combined.append(item)
        return self._filter_and_rank_results(query, combined)
