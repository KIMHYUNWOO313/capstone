# -*- coding: utf-8 -*-
"""OpenAI Web Search 기반 농산물·기상 뉴스 수집."""
from __future__ import annotations

import json
from datetime import timezone
from typing import Any
import re
from urllib.parse import parse_qsl, quote_plus, urlencode, unquote, urlsplit, urlunsplit
from xml.etree import ElementTree

from django.conf import settings

from .agri_store import _firestore
from .models import AgriNewsSnapshot


NEWS_CROPS = [
    "감자 수미",
    "고구마",
    "당근",
    "대파",
    "배추",
    "백다다기오이",
    "사과 부사",
    "시금치",
    "양파",
]

NEWS_WEATHER_KEYWORDS = ["고온", "강수량", "태풍", "홍수", "폭염", "집중호우"]
NEWS_RELEVANCE_KEYWORDS = [
    *NEWS_CROPS,
    "감자",
    "오이",
    "농산물",
    "채소",
    "과일",
    "가격",
    "도매",
    "출하",
    "작황",
    "기상",
    "날씨",
    *NEWS_WEATHER_KEYWORDS,
]
BAD_PAGE_KEYWORDS = [
    "페이지를 찾을 수 없습니다",
    "존재하지 않는 페이지",
    "삭제된 기사",
    "삭제된 페이지",
    "요청하신 페이지를 찾을 수 없습니다",
    "404 not found",
    "not found",
]
BAD_URL_PARTS = [
    "/search",
    "search.php",
    "search?",
    "/search/",
    "/member/",
    "/login",
]


def latest_news_snapshot() -> AgriNewsSnapshot | None:
    return AgriNewsSnapshot.objects.order_by("-fetched_at").first()


def latest_news_snapshots(limit: int = 8) -> list[AgriNewsSnapshot]:
    return list(AgriNewsSnapshot.objects.order_by("-fetched_at")[:limit])


def latest_reachable_news_snapshots(limit: int = 8) -> list[AgriNewsSnapshot]:
    """Return recent snapshots after removing obviously unreachable article links."""
    out: list[AgriNewsSnapshot] = []
    for snapshot in AgriNewsSnapshot.objects.order_by("-fetched_at")[: max(limit * 3, limit)]:
        articles = []
        for article in snapshot.articles or []:
            if not isinstance(article, dict):
                continue
            url = article.get("url")
            if _valid_url(url):
                article = dict(article)
                article["url"] = _clean_url(str(url))
                articles.append(article)
        if not articles:
            continue
        snapshot.articles = articles
        out.append(snapshot)
        if len(out) >= limit:
            break
    return out


def _news_query() -> str:
    crops = ", ".join(NEWS_CROPS)
    weather = ", ".join(NEWS_WEATHER_KEYWORDS)
    return (
        "한국 농산물 가격에 영향을 줄 수 있는 최신 뉴스와 기상 예보를 찾아라. "
        f"대상 작물: {crops}. "
        f"기상 이슈: {weather}. "
        "최근 기사와 예보 중심으로, 작물 가격 영향 가능성을 함께 요약하라."
    )


def _extract_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if text:
        return str(text).strip()
    chunks: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            value = getattr(content, "text", None)
            if value:
                chunks.append(str(value))
    return "\n".join(chunks).strip()


def _iter_response_content(response: Any):
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            yield content


def _extract_citations(response: Any) -> list[dict[str, str]]:
    citations: list[dict[str, str]] = []
    seen: set[str] = set()
    for content in _iter_response_content(response):
        annotations = getattr(content, "annotations", None) or []
        for ann in annotations:
            ann_type = getattr(ann, "type", "") or ""
            url = getattr(ann, "url", "") or ""
            title = getattr(ann, "title", "") or ""
            if not url and isinstance(ann, dict):
                ann_type = ann.get("type", ann_type)
                url = ann.get("url", "")
                title = ann.get("title", "")
            if ann_type in {"url_citation", "citation"} and url and url not in seen:
                seen.add(url)
                citations.append({"url": str(url), "title": str(title or "").strip()})
    return citations


def _parse_json_text(text: str) -> dict[str, Any]:
    raw = text.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        raw = "\n".join(lines).strip()
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        raw = raw[start : end + 1]
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {"summary": text, "articles": []}
    if not isinstance(data, dict):
        return {"summary": text, "articles": []}
    articles = data.get("articles") or []
    if not isinstance(articles, list):
        articles = []
    return {
        "summary": str(data.get("summary") or "").strip(),
        "articles": articles[:12],
    }


def _valid_url(url: Any) -> bool:
    if not isinstance(url, str):
        return False
    return url.startswith("http://") or url.startswith("https://")


def _clean_url(url: str) -> str:
    """Remove tracking parameters while preserving the actual article URL."""
    parts = urlsplit(url)
    query = [
        (k, v)
        for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if not k.lower().startswith("utm_")
    ]
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


def _html_to_text(html: str) -> str:
    text = re.sub(r"(?is)<script.*?</script>", " ", html)
    text = re.sub(r"(?is)<style.*?</style>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _url_is_reachable(url: str) -> bool:
    """Return True only for reachable, relevant, article-like pages."""
    if not _valid_url(url):
        return False
    lowered_url = unquote(url).lower()
    if any(part in lowered_url for part in BAD_URL_PARTS):
        return False
    try:
        import requests

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0 Safari/537.36"
            )
        }
        res = requests.get(url, headers=headers, timeout=8, allow_redirects=True)
        if not (200 <= res.status_code < 400):
            return False
        final_url = unquote(str(res.url or url)).lower()
        if any(part in final_url for part in BAD_URL_PARTS):
            return False
        if "news.google.com" in final_url:
            return True
        content_type = (res.headers.get("content-type") or "").lower()
        if "text/html" not in content_type and "text/plain" not in content_type:
            return False
        text = _html_to_text(res.text[:300000])
        lowered_text = text.lower()
        if any(bad in lowered_text for bad in BAD_PAGE_KEYWORDS):
            return False
        return any(keyword in text for keyword in NEWS_RELEVANCE_KEYWORDS)
    except Exception:
        return False


def _split_title_source(title: str, fallback_source: str = "") -> tuple[str, str]:
    clean = re.sub(r"\s+", " ", str(title or "")).strip()
    source = fallback_source
    # Google News RSS titles commonly end with " - 언론사".
    if " - " in clean:
        left, right = clean.rsplit(" - ", 1)
        if left.strip() and right.strip():
            clean = left.strip()
            source = right.strip()
    return clean or "관련 뉴스", source


def _infer_related_crops(title: str) -> list[str]:
    crops = []
    for crop in NEWS_CROPS:
        if crop in title:
            crops.append(crop)
    aliases = {
        "감자": "감자 수미",
        "오이": "백다다기오이",
    }
    for key, crop in aliases.items():
        if key in title and crop not in crops:
            crops.append(crop)
    return crops


def _infer_category(title: str) -> str:
    if any(k in title for k in NEWS_WEATHER_KEYWORDS + ["기상", "날씨", "농작물 피해"]):
        return "기상예보"
    return "작물뉴스"


def _infer_impact(title: str) -> str:
    if any(k in title for k in ["고온", "폭염"]):
        return "고온 이슈는 생육 스트레스와 품질 저하로 이어질 수 있어 가격 변동 요인으로 확인합니다."
    if any(k in title for k in ["강수", "집중호우", "홍수", "태풍"]):
        return "강수와 재해성 기상은 출하량과 산지 작업 여건에 영향을 줄 수 있어 가격 변동 요인으로 확인합니다."
    if any(k in title for k in ["가격", "도매", "출하", "작황", "수급"]):
        return "농산물 가격, 출하, 수급과 직접 관련된 기사로 예측 판단의 참고 정보로 사용합니다."
    return "관련 뉴스로 확인된 실제 기사이며, 가격 예측 해석 시 보조 정보로 참고합니다."


def _summary_from_articles(articles: list[dict[str, Any]], fallback: str) -> str:
    if not articles:
        return fallback
    categories = sorted({str(a.get("category") or "뉴스") for a in articles})
    crops = []
    for article in articles:
        for crop in article.get("related_crops") or []:
            if crop not in crops:
                crops.append(crop)
    lines = [
        f"총 {len(articles)}개의 실제 접근 가능한 관련 뉴스 링크를 확인했습니다.",
        f"분류: {', '.join(categories)}",
    ]
    if crops:
        lines.append(f"관련 작물: {', '.join(crops)}")
    lines.append("각 카드는 실제 기사 제목과 링크를 기준으로 표시됩니다.")
    return "\n".join(lines)


def _merge_citations(parsed: dict[str, Any], citations: list[dict[str, str]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    used_urls: set[str] = set()

    if citations:
        for citation in citations:
            url = citation.get("url")
            if not _valid_url(url):
                continue
            url = _clean_url(url)
            if url in used_urls or not _url_is_reachable(url):
                continue
            used_urls.add(url)
            host = urlsplit(url).netloc.replace("www.", "")
            title, source = _split_title_source(citation.get("title") or "", host)
            merged.append(
                {
                    "title": title,
                    "source": source or host,
                    "url": url,
                    "published_at": "",
                    "category": _infer_category(title),
                    "related_crops": _infer_related_crops(title),
                    "impact": _infer_impact(title),
                }
            )
        return merged[:12]

    return []


def _citation_search(client: Any, model: str, query: str) -> tuple[str, list[dict[str, str]]]:
    prompts = [
        "한국 배추 양파 대파 사과 농산물 가격 관련 최신 뉴스 기사 3개를 찾아 제목과 URL을 간단히 알려줘.",
        "한국 감자 고구마 당근 오이 시금치 농산물 가격 관련 최신 뉴스 기사 3개를 찾아 제목과 URL을 간단히 알려줘.",
        "한국 고온 강수량 태풍 홍수 폭염 집중호우 농작물 피해 기상 예보 최신 뉴스 기사 3개를 찾아 제목과 URL을 간단히 알려줘.",
    ]
    texts: list[str] = []
    citations: list[dict[str, str]] = []
    seen: set[str] = set()
    for prompt in prompts:
        response = client.responses.create(
            model=model,
            tools=[{"type": "web_search_preview"}],
            input=f"{prompt}\n\n검색 배경: {query}",
        )
        texts.append(_extract_text(response))
        for citation in _extract_citations(response):
            url = citation.get("url")
            if _valid_url(url) and url not in seen:
                seen.add(url)
                citations.append(citation)
    return "\n\n".join(texts), citations[:12]


def _rss_news_citations() -> list[dict[str, str]]:
    queries = [
        "한국 농산물 가격 배추 양파 대파 사과",
        "한국 농산물 가격 감자 고구마 당근 오이 시금치",
        "한국 농작물 피해 고온 강수량 태풍 홍수 폭염 집중호우",
    ]
    citations: list[dict[str, str]] = []
    seen: set[str] = set()
    try:
        import requests

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0 Safari/537.36"
            )
        }
        for q in queries:
            url = (
                "https://news.google.com/rss/search?q="
                + quote_plus(q)
                + "&hl=ko&gl=KR&ceid=KR:ko"
            )
            res = requests.get(url, headers=headers, timeout=10)
            if not (200 <= res.status_code < 400):
                continue
            root = ElementTree.fromstring(res.content)
            for item in root.findall(".//item")[:5]:
                title = (item.findtext("title") or "").strip()
                link = (item.findtext("link") or "").strip()
                if not _valid_url(link) or link in seen:
                    continue
                seen.add(link)
                citations.append({"title": title, "url": link})
                if len(citations) >= 12:
                    return citations
    except Exception:
        return citations
    return citations


def _save_news_to_firestore(snapshot: AgriNewsSnapshot) -> None:
    db = _firestore()
    if db is None:
        return
    fetched = snapshot.fetched_at
    if fetched and fetched.tzinfo is None:
        fetched = fetched.replace(tzinfo=timezone.utc)
    doc_id = f"news__{snapshot.id}__{snapshot.fetched_at.strftime('%Y%m%dT%H%M%S')}"
    db.collection("agri_news").document(doc_id).set(
        {
            "id": snapshot.id,
            "query": snapshot.query,
            "summary": snapshot.summary,
            "articles": snapshot.articles or [],
            "raw_response": snapshot.raw_response or {},
            "fetched_at": fetched,
            "source": "openai_web_search",
        }
    )


def fetch_agri_news_with_web_search() -> AgriNewsSnapshot:
    api_key = getattr(settings, "OPENAI_API_KEY", "") or ""
    if not api_key.strip():
        raise RuntimeError("OPENAI_API_KEY가 설정되지 않았습니다.")

    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    model = getattr(settings, "OPENAI_NEWS_MODEL", "gpt-4o-mini")
    query = _news_query()
    prompt = f"""
OpenAI Web Search를 사용해 한국어 최신 뉴스를 조사하라.

요구사항:
- 농산물 뉴스: 감자 수미, 고구마, 당근, 대파, 배추, 백다다기오이, 사과 부사, 시금치, 양파
- 기상 뉴스/예보: 고온, 강수량, 태풍, 홍수, 폭염, 집중호우
- 가격에 영향을 줄 가능성이 있는 내용 위주로 정리
- 반드시 JSON만 출력

JSON 형식:
{{
  "summary": "전체 핵심 요약 3~5문장",
  "articles": [
    {{
      "title": "기사 또는 예보 제목",
      "source": "매체명 또는 기관명",
      "url": "URL",
      "published_at": "확인 가능한 날짜 또는 빈 문자열",
      "category": "작물뉴스 또는 기상예보",
      "related_crops": ["배추", "양파"],
      "impact": "가격 영향 가능성 1~2문장"
    }}
  ]
}}

검색 질의:
{query}
""".strip()

    response = client.responses.create(
        model=model,
        tools=[{"type": "web_search_preview"}],
        input=prompt,
    )
    text = _extract_text(response)
    citations = _extract_citations(response)
    citation_text = ""
    if not citations:
        citation_text, citations = _citation_search(client, model, query)
    rss_citations = _rss_news_citations()
    rss_urls = {c.get("url") for c in rss_citations}
    citations = rss_citations + [c for c in citations if c.get("url") not in rss_urls]
    parsed = _parse_json_text(text)
    articles = _merge_citations(parsed, citations)
    snapshot = AgriNewsSnapshot.objects.create(
        query=query,
        summary=_summary_from_articles(articles, parsed.get("summary") or text),
        articles=articles,
        raw_response={
            "output_text": text,
            "citation_text": citation_text,
            "citations": citations,
        },
    )
    _save_news_to_firestore(snapshot)
    return snapshot
