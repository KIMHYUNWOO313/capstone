"""기상청 단기예보(getVilageFcst) + 중기기온(getMidTa) API 호출 + 일별 집계.

설치:
    pip install requests python-dotenv

환경변수 (.env):
    short_forecast_api   - 단기예보 서비스키
    middle_forecast_api  - 중기예보 서비스키

발급:
    https://www.data.go.kr/data/15084084/openapi.do  (단기예보)
    https://www.data.go.kr/data/15059468/openapi.do  (중기예보)
"""
from __future__ import annotations

import datetime as _dt
import json
import os
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

PKG_DIR = Path(__file__).resolve().parent
load_dotenv(PKG_DIR / ".env")

SHORT_KEY = (os.getenv("short_forecast_api") or "").strip()
MID_KEY = (os.getenv("middle_forecast_api") or "").strip()

SHORT_URL = "http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst"
MID_TA_URL = "http://apis.data.go.kr/1360000/MidFcstInfoService/getMidTa"

TIMEOUT_SEC = 10


# ─── 발표 시각 헬퍼 ────────────────────────────────────────────────────────────

# 단기예보 발표 시각 (02·05·08·11·14·17·20·23시), 매시 10분 후 사용 가능
SHORT_BASE_HOURS = [2, 5, 8, 11, 14, 17, 20, 23]
# 중기예보 발표 시각 (06·18시), 매 회 10분 후 사용 가능
MID_BASE_HOURS = [6, 18]


def latest_short_base(now: _dt.datetime | None = None) -> tuple[str, str]:
    """현재 시각 기준 호출 가능한 가장 최근 단기예보 발표시각.

    Returns:
        (base_date YYYYMMDD, base_time HHMM)
    """
    now = now or _dt.datetime.now()
    cutoff = now - _dt.timedelta(minutes=15)
    for h in reversed(SHORT_BASE_HOURS):
        candidate = cutoff.replace(hour=h, minute=0, second=0, microsecond=0)
        if candidate <= cutoff:
            return candidate.strftime("%Y%m%d"), f"{h:02d}00"
    # 새벽 02시 이전: 전날 23시 사용
    prev = (cutoff - _dt.timedelta(days=1)).replace(hour=23, minute=0)
    return prev.strftime("%Y%m%d"), "2300"


def latest_mid_base(now: _dt.datetime | None = None) -> str:
    """가장 최근 중기예보 발표시각 (YYYYMMDDHHMM)."""
    now = now or _dt.datetime.now()
    cutoff = now - _dt.timedelta(minutes=15)
    for h in reversed(MID_BASE_HOURS):
        candidate = cutoff.replace(hour=h, minute=0, second=0, microsecond=0)
        if candidate <= cutoff:
            return candidate.strftime("%Y%m%d%H%M")
    prev = (cutoff - _dt.timedelta(days=1)).replace(hour=18, minute=0)
    return prev.strftime("%Y%m%d%H%M")


# ─── 단기예보 호출 ────────────────────────────────────────────────────────────

def fetch_short_forecast(
    nx: int,
    ny: int,
    base_date: str | None = None,
    base_time: str | None = None,
    service_key: str | None = None,
    max_pages: int = 5,
) -> pd.DataFrame | None:
    """getVilageFcst 호출. 모든 페이지 합쳐 단일 DataFrame 반환.

    컬럼: baseDate, baseTime, category, fcstDate, fcstTime, fcstValue, nx, ny
    """
    key = service_key or SHORT_KEY
    if not key:
        raise RuntimeError(".env 의 short_forecast_api 가 비어있음")
    if base_date is None or base_time is None:
        base_date, base_time = latest_short_base()

    all_items: list[dict] = []
    for page in range(1, max_pages + 1):
        params = {
            "serviceKey": key,
            "dataType": "JSON",
            "numOfRows": 1000,
            "pageNo": page,
            "base_date": base_date,
            "base_time": base_time,
            "nx": nx,
            "ny": ny,
        }
        try:
            r = requests.get(SHORT_URL, params=params, timeout=TIMEOUT_SEC)
        except requests.RequestException as e:
            print(f"  [short] request error page={page}: {e}")
            return None
        if r.status_code != 200 or r.text.lstrip().startswith("<"):
            print(f"  [short] non-JSON response page={page}: {r.text[:200]}")
            return None
        try:
            payload = r.json()
        except json.JSONDecodeError:
            return None
        body = (payload.get("response", {}) or {}).get("body", {}) or {}
        items_block = body.get("items", {}) or {}
        items = items_block.get("item", []) or []
        if isinstance(items, dict):
            items = [items]
        all_items.extend(items)
        total = int(body.get("totalCount", 0))
        if len(all_items) >= total or not items:
            break

    if not all_items:
        return None
    df = pd.DataFrame(all_items)
    df["fcstValue_raw"] = df["fcstValue"]
    df["fcstValue"] = pd.to_numeric(df["fcstValue"], errors="coerce")
    return df


def aggregate_short_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """1시간 단위 단기예보 → 일 단위 집계.

    컬럼: date, tmn, tmx, temp_range, tmp_mean, pop_mean, reh_mean, wsd_mean
    """
    df = df.copy()
    df["fcstDate"] = pd.to_datetime(df["fcstDate"], format="%Y%m%d")

    pivot = df.pivot_table(
        index=["fcstDate", "fcstTime"],
        columns="category",
        values="fcstValue",
        aggfunc="first",
    ).reset_index()

    rows = []
    for d, sub in pivot.groupby("fcstDate"):
        tmn = sub["TMN"].dropna().iloc[0] if "TMN" in sub and sub["TMN"].notna().any() else None
        tmx = sub["TMX"].dropna().iloc[0] if "TMX" in sub and sub["TMX"].notna().any() else None
        rows.append({
            "date": d.date(),
            "tmn": tmn,
            "tmx": tmx,
            "temp_range": (tmx - tmn) if (tmn is not None and tmx is not None) else None,
            "tmp_mean":  sub["TMP"].mean()  if "TMP"  in sub else None,
            "pop_mean":  sub["POP"].mean()  if "POP"  in sub else None,
            "reh_mean":  sub["REH"].mean()  if "REH"  in sub else None,
            "wsd_mean":  sub["WSD"].mean()  if "WSD"  in sub else None,
        })
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


# ─── 중기기온 호출 ────────────────────────────────────────────────────────────

def fetch_mid_temperature(
    reg_id: str,
    tm_fc: str | None = None,
    service_key: str | None = None,
) -> dict | None:
    """getMidTa 호출. 응답 dict 한 건 반환.

    Returns: {"taMin4": x, "taMax4": y, ..., "taMin10": ..., "taMax10": ...}
    """
    key = service_key or MID_KEY
    if not key:
        raise RuntimeError(".env 의 middle_forecast_api 가 비어있음")
    if tm_fc is None:
        tm_fc = latest_mid_base()

    params = {
        "serviceKey": key,
        "dataType": "JSON",
        "numOfRows": 10,
        "pageNo": 1,
        "regId": reg_id,
        "tmFc": tm_fc,
    }
    try:
        r = requests.get(MID_TA_URL, params=params, timeout=TIMEOUT_SEC)
    except requests.RequestException as e:
        print(f"  [mid] request error: {e}")
        return None
    if r.status_code != 200 or r.text.lstrip().startswith("<"):
        print(f"  [mid] non-JSON: {r.text[:200]}")
        return None
    try:
        payload = r.json()
    except json.JSONDecodeError:
        return None
    items = ((payload.get("response", {}) or {}).get("body", {}) or {}).get("items", {}) or {}
    item = items.get("item")
    if not item:
        return None
    if isinstance(item, list):
        item = item[0]
    out: dict = {"regId": reg_id, "tmFc": tm_fc}
    for d in range(3, 11):  # 06시 발표는 D+4~+10, 18시는 D+5~+10 → 키만 있으면 채움
        for fld in (f"taMin{d}", f"taMax{d}"):
            v = item.get(fld)
            out[fld] = float(v) if v not in (None, "") else None
    return out


def mid_temp_range_series(mid_resp: dict | None, today: _dt.date) -> dict[_dt.date, float]:
    """getMidTa 응답 → {D+4~D+10 date: temp_range}."""
    if not mid_resp:
        return {}
    out = {}
    for d in range(3, 11):
        tmn = mid_resp.get(f"taMin{d}")
        tmx = mid_resp.get(f"taMax{d}")
        if tmn is None or tmx is None:
            continue
        date = today + _dt.timedelta(days=d)
        out[date] = float(tmx) - float(tmn)
    return out


# ─── 단독 실행 (검증용) ───────────────────────────────────────────────────────
if __name__ == "__main__":
    from stn_grid_map import STATION_TO_GRID, STATION_TO_MID_REGID

    station = "jecheon"
    nx, ny = STATION_TO_GRID[station]
    reg = STATION_TO_MID_REGID[station]

    bd, bt = latest_short_base()
    print(f"[short] base_date={bd} base_time={bt}  nx={nx} ny={ny}")
    sdf = fetch_short_forecast(nx, ny, bd, bt)
    if sdf is None:
        print("  단기예보 응답 None")
    else:
        print(f"  rows={len(sdf)}  categories={sorted(sdf['category'].unique())}")
        daily = aggregate_short_to_daily(sdf)
        print("\n[short → 일별 집계]")
        print(daily.to_string(index=False))

    tm = latest_mid_base()
    print(f"\n[mid] tmFc={tm}  regId={reg}")
    mr = fetch_mid_temperature(reg, tm)
    if mr is None:
        print("  중기기온 응답 None")
    else:
        today = _dt.date.today()
        ranges = mid_temp_range_series(mr, today)
        print(f"  D+4~D+10 일교차: {ranges}")
