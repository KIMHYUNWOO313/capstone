# -*- coding: utf-8 -*-
"""외부 API 자동 수집 클라이언트 — KAMIS, 오피넷, 기상청.

각 함수는 (날짜별·항목별) 스냅샷 리스트를 반환한다:
    [{"snapshot_date": date, "item_key": str, "payload": dict}, ...]
실제 호출은 requests로 수행하며, 키가 없거나 응답 실패 시 빈 리스트.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Any, Iterable

from django.conf import settings

logger = logging.getLogger(__name__)


def _safe_request_json(url: str, params: dict, timeout: int = 20) -> Any:
    import requests

    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _safe_request_xml(url: str, params: dict, timeout: int = 20) -> Any:
    import requests
    from xml.etree import ElementTree as ET

    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return ET.fromstring(r.content)


# -------------------- KAMIS (매일 업데이트) --------------------
# 일별 도·소매 가격 정보. 공공데이터포털 KAMIS API.
# 엔드포인트: http://www.kamis.or.kr/service/price/xml.do
# action=dailyPriceByCategoryList, productClsCode=01(소매)/02(도매)
# 무·배추·양파·사과·배·마늘 카테고리 코드는 KAMIS 코드표 참고. 여기서는
# 프론트(부류번호)별로 묶어서 받고, payload에 그대로 저장한다.

# (한글 표시용, 실제 호출 시는 cer_code/p_item_category_code 사용)
_KAMIS_CATEGORIES: tuple[dict[str, str], ...] = (
    {"label": "식량작물", "code": "100"},   # 마늘 등
    {"label": "채소류", "code": "200"},     # 배추, 무, 양파
    {"label": "과일류", "code": "400"},     # 사과, 배
)


def fetch_kamis_daily(target_date: date) -> list[dict[str, Any]]:
    api_key = getattr(settings, "KAMIS_API_KEY", "") or ""
    api_id = getattr(settings, "KAMIS_API_ID", "") or "tester"
    if not api_key.strip():
        logger.info("KAMIS_API_KEY not set; skip")
        return []

    url = "http://www.kamis.or.kr/service/price/xml.do"
    out: list[dict[str, Any]] = []
    for cat in _KAMIS_CATEGORIES:
        params = {
            "action": "dailyPriceByCategoryList",
            "p_product_cls_code": "01",  # 소매(01) / 도매(02)
            "p_country_code": "1101",  # 서울 기준 가격(필수)
            "p_regday": target_date.strftime("%Y-%m-%d"),
            "p_convert_kg_yn": "N",
            "p_item_category_code": cat["code"],
            "p_cert_key": api_key,
            "p_cert_id": api_id,
            "p_returntype": "json",
        }
        try:
            data = _safe_request_json(url, params)
        except Exception as exc:
            logger.warning("KAMIS request failed (%s): %s", cat["label"], exc)
            continue

        out.append(
            {
                "snapshot_date": target_date,
                "item_key": f"category_{cat['code']}",
                "payload": {
                    "category_label": cat["label"],
                    "category_code": cat["code"],
                    "country_code": "1101",
                    "product_cls": "retail",
                    "raw": data,
                },
            }
        )
    return out


# -------------------- OPINET (매일 업데이트) --------------------
# avgRecentPrice.do 는 전일부터 7일치를 반환. 하지만 매일 호출해
# date 파라미터로 단일 일자만 가져오도록 한다.
# 무료 키는 일 1500회 제한.

_OPINET_PRODS = ("B027", "B034", "D047", "C004")  # 보통/고급/경유/등유


def fetch_opinet_daily(target_date: date) -> list[dict[str, Any]]:
    api_key = getattr(settings, "OPINET_API_KEY", "") or ""
    if not api_key.strip():
        logger.info("OPINET_API_KEY not set; skip")
        return []

    url = "https://www.opinet.co.kr/api/avgRecentPrice.do"
    payload_per_prod: dict[str, Any] = {}
    for prod in _OPINET_PRODS:
        params = {
            "code": api_key,
            "out": "json",
            "date": target_date.strftime("%Y%m%d"),
            "prodcd": prod,
        }
        try:
            data = _safe_request_json(url, params)
        except Exception as exc:
            logger.warning("OPINET request failed (%s): %s", prod, exc)
            continue
        payload_per_prod[prod] = data

    if not payload_per_prod:
        return []

    return [
        {
            "snapshot_date": target_date,
            "item_key": "national_avg",
            "payload": {
                "products": payload_per_prod,
                "scope": "national",
            },
        }
    ]


# -------------------- 기상청 ASOS 일자료 (10일 단위 요청) --------------------
# 공공데이터포털 — getWthrDataList (ASOS 일자료)
# https://apis.data.go.kr/1360000/AsosDalyInfoService/getWthrDataList
# 한 번에 10일치를 받아서 저장한다. 매일이 아니라 10일에 한 번 호출.


def fetch_kma_asos_block(end_date: date, days: int = 10) -> list[dict[str, Any]]:
    api_key = getattr(settings, "WEATHER_API_KEY", "") or ""
    if not api_key.strip():
        logger.info("WEATHER_API_KEY not set; skip")
        return []

    stn_ids = getattr(settings, "WEATHER_STN_IDS", "108") or "108"
    start_date = end_date - timedelta(days=days - 1)

    url = "https://apis.data.go.kr/1360000/AsosDalyInfoService/getWthrDataList"
    params = {
        "serviceKey": api_key,
        "pageNo": "1",
        "numOfRows": str(max(days * 5, 50)),
        "dataType": "JSON",
        "dataCd": "ASOS",
        "dateCd": "DAY",
        "startDt": start_date.strftime("%Y%m%d"),
        "endDt": end_date.strftime("%Y%m%d"),
        "stnIds": str(stn_ids),
    }
    try:
        data = _safe_request_json(url, params)
    except Exception as exc:
        logger.warning("KMA ASOS request failed: %s", exc)
        return []

    items: list[dict[str, Any]] = []
    try:
        body = (
            data.get("response", {})
            .get("body", {})
            .get("items", {})
            .get("item", [])
        )
    except AttributeError:
        body = []
    if isinstance(body, dict):
        body = [body]

    by_date: dict[str, list[dict[str, Any]]] = {}
    for it in body or []:
        d_raw = str(it.get("tm") or "")
        if not d_raw:
            continue
        by_date.setdefault(d_raw, []).append(it)

    for d_str, daily_items in by_date.items():
        try:
            d = datetime.strptime(d_str.replace("-", ""), "%Y%m%d").date()
        except ValueError:
            continue
        items.append(
            {
                "snapshot_date": d,
                "item_key": f"stn_{stn_ids}",
                "payload": {
                    "stn_ids": str(stn_ids),
                    "items": daily_items,
                    "block_start": start_date.isoformat(),
                    "block_end": end_date.isoformat(),
                },
            }
        )
    return items


def collect_all(target_date: date, include_kma_block: bool = False) -> dict[str, list[dict[str, Any]]]:
    """단일 진입점.

    target_date 기준으로 KAMIS·OPINET 일별 1건을 모은다.
    include_kma_block=True 이면 KMA ASOS 10일치 블록도 함께 모은다.
    """
    result = {
        "kamis": fetch_kamis_daily(target_date),
        "opinet": fetch_opinet_daily(target_date),
        "kma": fetch_kma_asos_block(target_date, days=10) if include_kma_block else [],
    }
    return result
