"""기상청 기상특보 조회서비스 호출 및 품목별 매핑.

API 명세: 기상청 'WthrWrnInfoService' Open API.
- getWthrWrnMsg : 활성 특보 통보문 (제목, 해당구역, 발효시각, 내용)
- getWthrPwn    : 예비특보
인증키는 .env의 WARN_API_KEY.
"""
import datetime as _dt
import json
from typing import Any

import requests

from config import META_PATH, WARN_API_BASE, WARN_API_KEY
from stn_code_map import (
    STATION_TO_REGION_STN,
    STN_TO_REGION_KR,
    stations_to_region_stns,
)

TIMEOUT_SEC = 8


def _get(endpoint: str, **params) -> dict[str, Any] | None:
    url = f"{WARN_API_BASE}/{endpoint}"
    params = {
        "serviceKey": WARN_API_KEY,
        "dataType": "JSON",
        "numOfRows": 20,
        "pageNo": 1,
        **params,
    }
    try:
        r = requests.get(url, params=params, timeout=TIMEOUT_SEC)
        r.raise_for_status()
        text = r.text
        if text.lstrip().startswith("<"):
            return None
        return json.loads(text)
    except Exception:
        return None


def _items(payload: dict | None) -> list[dict]:
    if not payload:
        return []
    try:
        items = payload["response"]["body"]["items"]["item"]
    except (KeyError, TypeError):
        return []
    if isinstance(items, dict):
        return [items]
    return items


def _fetch_warn(stn_id: int, from_tm: str, to_tm: str) -> list[dict]:
    msg = _get(
        "getWthrWrnMsg",
        stnId=stn_id,
        fromTmFc=from_tm,
        toTmFc=to_tm,
    )
    pwn = _get(
        "getWthrPwn",
        stnId=stn_id,
        fromTmFc=from_tm,
        toTmFc=to_tm,
    )

    out = []
    for it in _items(msg):
        out.append({
            "kind": "특보",
            "title": (it.get("t1") or "").strip(),
            "regions": (it.get("t2") or "").strip(),
            "effective": (it.get("t3") or "").strip(),
            "body": (it.get("t4") or "").strip(),
            "current": (it.get("t6") or "").strip(),
            "tmFc": it.get("tmFc"),
        })
    for it in _items(pwn):
        out.append({
            "kind": "예비특보",
            "title": (it.get("t1") or "").strip(),
            "regions": (it.get("t2") or "").strip(),
            "effective": (it.get("t3") or "").strip(),
            "body": (it.get("t4") or "").strip(),
            "current": "",
            "tmFc": it.get("tmFc"),
        })
    return out


def fetch_warnings_by_item(window_days_back: int = 7) -> dict[str, list[dict]]:
    """
    item_station_map을 광역 stnId로 묶어 한 번씩 호출 후 품목별로 분배.
    Returns: {item_id: [warning_dict, ...]}
    """
    with open(META_PATH, encoding="utf-8") as f:
        meta = json.load(f)
    item_station_map: dict[str, str] = meta["item_station_map"]

    today = _dt.date.today()
    from_tm = (today - _dt.timedelta(days=window_days_back)).strftime("%Y%m%d")
    to_tm = today.strftime("%Y%m%d")

    grouped = stations_to_region_stns(list(item_station_map.values()))

    stn_to_warns: dict[int, list[dict]] = {}
    for stn_id in grouped:
        stn_to_warns[stn_id] = _fetch_warn(stn_id, from_tm, to_tm)

    out: dict[str, list[dict]] = {}
    for item_id, station in item_station_map.items():
        stn_id = STATION_TO_REGION_STN.get(station)
        if stn_id is None:
            out[item_id] = []
            continue
        warns = stn_to_warns.get(stn_id, [])
        annotated = []
        for w in warns:
            annotated.append({
                **w,
                "region_code": stn_id,
                "region_kr": STN_TO_REGION_KR.get(stn_id, str(stn_id)),
            })
        out[item_id] = annotated
    return out


def format_warnings(warnings: list[dict]) -> str:
    if not warnings:
        return (
            "**최근 1주일(D-7 ~ 오늘) 동안 해당 광역에 발효된 활성·예비 기상특보 없음.** "
            "(기상 이벤트로 인한 단기 가격 충격 요인은 관측되지 않음)"
        )
    # 발표시각(tmFc) 기준 최신순 정렬
    sorted_w = sorted(
        warnings,
        key=lambda x: x.get("tmFc") or "",
        reverse=True,
    )
    lines = [
        f"(최근 1주일 동안 발효된 특보 총 {len(sorted_w)}건, 최신 발표순)"
    ]
    for w in sorted_w:
        title = w["title"] or w["kind"]
        tm = w.get("tmFc") or ""
        # tmFc: YYYYMMDDHHMM → YYYY-MM-DD HH:MM 로 보기 좋게
        tm_fmt = (
            f"{tm[:4]}-{tm[4:6]}-{tm[6:8]} {tm[8:10]}:{tm[10:12]}"
            if len(tm) >= 12 else tm
        )
        line = f"- [{w['kind']}] {tm_fmt} | {title} | {w['region_kr']}"
        if w.get("effective"):
            line += f" | 발효: {w['effective'][:80]}"
        if w.get("current"):
            line += f" | 현황: {w['current'][:140]}"
        lines.append(line)
    return "\n".join(lines)


if __name__ == "__main__":
    if not WARN_API_KEY:
        raise SystemExit(".env에 WARN_API_KEY를 채워주세요.")
    by_item = fetch_warnings_by_item()
    print(f"품목 수: {len(by_item)}")
    sample = "apple_fuji_box10kg_high"
    print(f"\n[{sample}]")
    print(format_warnings(by_item.get(sample, [])))
