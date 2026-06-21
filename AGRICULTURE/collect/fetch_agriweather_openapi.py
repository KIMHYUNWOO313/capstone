# -*- coding: utf-8 -*-
"""
공공데이터포털 「농촌진흥청 국립농업과학원_농업기상 기본 관측데이터 조회」 OpenAPI 수집.

데이터셋: https://www.data.go.kr/data/15078057/openapi.do

- 활용신청 후 발급되는 **일반 인증키(serviceKey)** 가 필요합니다.
- 환경 변수: `DATA_GO_KR_SERVICE_KEY` (또는 `AGRI_WEATHER_SERVICE_KEY`)
- 기본 엔드포인트: V2 `getWeatherMonDayList` — 조회 연·월 기준 **일 단위** 기본 관측(전국 관측지점, 페이지네이션).
- 선택: `--obs-code` 로 특정 관측지점만 (포털 기술문서의 파라미터명 `search_Obsrvn_Spot_Code`).

명세·파라미터는 포털에서 받은 최신 기술문서를 우선하세요. 엔드포인트 변경 시 `--endpoint` 로 덮어쓸 수 있습니다.

사용 예:
  set DATA_GO_KR_SERVICE_KEY=발급키
  python AGRICULTURE/collect/fetch_agriweather_openapi.py --year-from 2023 --year-to 2023
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import requests

AG_ROOT = Path(__file__).resolve().parents[1]
RAW = AG_ROOT / "data" / "raw" / "agriweather"
RAW.mkdir(parents=True, exist_ok=True)

# 포털·블로그 기준 경로 (V2 권장). 동작하지 않으면 --legacy 또는 --endpoint 로 조정.
BASE_V2 = (
    "https://apis.data.go.kr/1390802/AgriWeather/WeatherObsrInfo/V2/GnrlWeather"
)
BASE_LEGACY = (
    "https://apis.data.go.kr/1390802/AgriWeather/WeatherObsrInfo/GnrlWeather"
)

OPERATIONS = {
    "mon_day": "getWeatherMonDayList",  # search_Year, search_Month, Page_No, Page_Size
}


def _local_tag(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[-1]
    return tag


def _parse_items(xml_bytes: bytes) -> tuple[str | None, str | None, list[dict[str, str | None]]]:
    root = ET.fromstring(xml_bytes)
    result_code: str | None = None
    result_msg: str | None = None
    for el in root.iter():
        t = _local_tag(el.tag)
        if t == "resultCode" and el.text is not None:
            result_code = el.text.strip()
        if t == "resultMsg" and el.text is not None:
            result_msg = el.text.strip()

    rows: list[dict[str, str | None]] = []
    for el in root.iter():
        if _local_tag(el.tag) != "item":
            continue
        if not len(el):
            continue
        row: dict[str, str | None] = {}
        for child in el:
            row[_local_tag(child.tag)] = child.text
        rows.append(row)
    return result_code, result_msg, rows


def _is_ok_code(code: str | None) -> bool:
    if code is None:
        return False
    return code in {"00", "0", "200"}


def fetch_month(
    session: requests.Session,
    url: str,
    service_key: str,
    year: int,
    month: int,
    *,
    page_size: int,
    obs_code: str | None,
    sleep_sec: float,
) -> list[dict[str, str | None]]:
    month_str = f"{month:02d}"
    all_rows: list[dict[str, str | None]] = []
    page = 1
    while True:
        params: dict[str, str | int] = {
            "serviceKey": service_key,
            "Page_No": page,
            "Page_Size": min(page_size, 100),
            "search_Year": year,
            "search_Month": month_str,
        }
        if obs_code:
            params["search_Obsrvn_Spot_Code"] = obs_code

        r = session.get(url, params=params, timeout=120)
        if r.status_code == 404:
            raise RuntimeError(
                "HTTP 404 (API not found). 공공데이터포털 해당 API 상세에서 "
                "「요청주소/엔드포인트」를 복사해 --endpoint 로 전체 URL을 지정하세요."
            )
        r.raise_for_status()
        code, msg, rows = _parse_items(r.content)

        if not _is_ok_code(code):
            raise RuntimeError(
                f"API 오류 resultCode={code!r} resultMsg={msg!r} "
                f"(첫 500자 응답): {r.text[:500]!r}"
            )

        if not rows:
            break
        all_rows.extend(rows)
        if len(rows) < params["Page_Size"]:
            break
        page += 1
        if sleep_sec > 0:
            time.sleep(sleep_sec)
    return all_rows


def main() -> int:
    p = argparse.ArgumentParser(description="농업기상 기본 관측 OpenAPI 수집 (15078057)")
    p.add_argument("--year-from", type=int, required=True)
    p.add_argument("--year-to", type=int, required=True)
    p.add_argument(
        "--service-key",
        default=os.environ.get("DATA_GO_KR_SERVICE_KEY")
        or os.environ.get("AGRI_WEATHER_SERVICE_KEY")
        or "",
        help="미지정 시 환경 변수 DATA_GO_KR_SERVICE_KEY 사용",
    )
    p.add_argument("--operation", choices=list(OPERATIONS.keys()), default="mon_day")
    p.add_argument("--legacy", action="store_true", help="V2 대신 구 경로(GnrlWeather) 사용")
    p.add_argument(
        "--endpoint",
        default="",
        help="전체 URL(쿼리스트링 제외). 예: .../getWeatherMonDayList",
    )
    p.add_argument("--page-size", type=int, default=100)
    p.add_argument("--sleep", type=float, default=0.12, help="페이지 간 대기(초), TPS 제한 완화")
    p.add_argument("--obs-code", default=None, help="관측지점코드(선택, search_Obsrvn_Spot_Code)")
    p.add_argument("--out-dir", type=Path, default=RAW)
    args = p.parse_args()

    key = (args.service_key or "").strip()
    if not key:
        print(
            "serviceKey 가 없습니다. 공공데이터포털에서 활용신청 후 "
            "DATA_GO_KR_SERVICE_KEY 환경 변수를 설정하세요.",
            file=sys.stderr,
        )
        return 1

    if args.endpoint:
        url = args.endpoint.strip()
    else:
        base = BASE_LEGACY if args.legacy else BASE_V2
        url = f"{base.rstrip('/')}/{OPERATIONS[args.operation]}"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers["User-Agent"] = "capstone-agriculture/1.0 (research)"

    for y in range(args.year_from, args.year_to + 1):
        for m in range(1, 13):
            print(f"Fetching {y}-{m:02d} ...", flush=True)
            try:
                rows = fetch_month(
                    session,
                    url,
                    key,
                    y,
                    m,
                    page_size=args.page_size,
                    obs_code=args.obs_code,
                    sleep_sec=args.sleep,
                )
            except requests.HTTPError as e:
                print(f"HTTP 오류: {e}", file=sys.stderr)
                if e.response is not None:
                    print(e.response.text[:800], file=sys.stderr)
                return 1
            except Exception as e:
                print(f"오류 ({y}-{m:02d}): {e}", file=sys.stderr)
                return 1

            if not rows:
                print(f"  (데이터 없음)", flush=True)
                continue
            df = pd.DataFrame(rows)
            fp = out_dir / f"agriweather_mon_day_{y}_{m:02d}.csv"
            df.to_csv(fp, index=False, encoding="utf-8-sig")
            print(f"  -> {fp} ({len(df)} rows)", flush=True)
            if args.sleep > 0:
                time.sleep(args.sleep)

    print("완료.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
