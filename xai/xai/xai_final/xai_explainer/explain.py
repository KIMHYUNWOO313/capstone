"""OpenAI Responses API (gpt-4o) + web_search 도구를 사용한 품목별 설명 생성."""
import json
import re
from typing import Any

from openai import OpenAI

from config import GPT_API_KEY, GPT_MODEL

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        if not GPT_API_KEY:
            raise RuntimeError(".env의 GPT_API_KEY가 비어 있습니다.")
        _client = OpenAI(api_key=GPT_API_KEY)
    return _client


def _extract_output_text(response) -> str:
    text = getattr(response, "output_text", None)
    if text:
        return text
    chunks = []
    for item in getattr(response, "output", []) or []:
        for c in getattr(item, "content", []) or []:
            t = getattr(c, "text", None)
            if t:
                chunks.append(t)
    return "\n".join(chunks)


def _escape_inner_whitespace(s: str) -> str:
    """JSON 문자열 리터럴 내부의 raw newline/tab을 escape.

    GPT가 긴 narrative를 만들 때 종종 string value 안에 실제 줄바꿈을 넣어
    JSON 표준 위반이 된다. 표준 파서가 실패하면 이 함수로 보정 후 재시도.
    """
    out: list[str] = []
    in_string = False
    escape = False
    for ch in s:
        if escape:
            out.append(ch)
            escape = False
            continue
        if ch == "\\":
            out.append(ch)
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            out.append(ch)
            continue
        if in_string:
            if ch == "\n":
                out.append("\\n")
                continue
            if ch == "\r":
                out.append("\\r")
                continue
            if ch == "\t":
                out.append("\\t")
                continue
        out.append(ch)
    return "".join(out)


def _parse_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    candidate = m.group(0) if m else text
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    try:
        return json.loads(_escape_inner_whitespace(candidate))
    except json.JSONDecodeError:
        pass

    return {"_raw": text, "_parse_error": True}


def explain_item(
    system_prompt: str,
    user_prompt: str,
    enable_web_search: bool = True,
) -> dict[str, Any]:
    client = _get_client()

    kwargs: dict[str, Any] = {
        "model": GPT_MODEL,
        "input": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    if enable_web_search:
        kwargs["tools"] = [{"type": "web_search"}]

    response = client.responses.create(**kwargs)
    text = _extract_output_text(response)
    parsed = _parse_json(text)

    tool_calls = 0
    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", "") in {"web_search_call", "tool_call"}:
            tool_calls += 1

    parsed["_meta"] = {
        "model": GPT_MODEL,
        "web_search_calls": tool_calls,
        "response_id": getattr(response, "id", None),
    }
    return parsed
