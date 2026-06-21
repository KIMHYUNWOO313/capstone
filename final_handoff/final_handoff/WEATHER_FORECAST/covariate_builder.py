"""기상청 일기예보 → Chronos2 LoRA known_covariates 프레임 빌드 (운영용).

Chronos2 LoRA (10일 확률예측) 의 5개 known_covariates 중 미래값이 실제로 변동되는
weather_temp_range 만 기상청 API 로 채우고, 나머지 4종은 마지막 관측치 ffill 한다:
  - market_rest          : 마지막 관측치 (운영 시 한국천문연구원 특일정보 API 통합 권장)
  - weather_temp_range   : 단기예보 D+1~D+3 (TMX-TMN) + 중기예보 D+4~D+10 (taMax-taMin)
  - weather_sunshine_dur : 마지막 관측치 (단기/중기 API 미제공)
  - bok_base_rate        : 마지막 관측치 (월 단위 변동, 한국은행 금통위)
  - cpi_growth_rate      : 마지막 관측치 (월 단위 변동, 통계청)

주의:
- 운영 시 full_baseline 은 매일 갱신되어 마지막 timestamp = today-1 이어야 일자 매핑 정확.
- 점예측(TimesFM)에는 xreg 가 백테스트 결과 ZS 대비 +4.8% 악화 → 의도적으로 본 모듈에서 제외.
"""
from __future__ import annotations

import datetime as _dt
from pathlib import Path

import numpy as np
import pandas as pd

from stn_grid_map import STATION_TO_GRID, STATION_TO_MID_REGID
from weather_fetcher import (
    aggregate_short_to_daily,
    fetch_mid_temperature,
    fetch_short_forecast,
    latest_mid_base,
    latest_short_base,
    mid_temp_range_series,
)

KNOWN_COVS = [
    "market_rest",
    "weather_temp_range",
    "weather_sunshine_dur",
    "bok_base_rate",
    "cpi_growth_rate",
]


# ─── 공유 캐시 (광역 stnId당 1회만 호출하기 위함) ─────────────────────────────
_short_cache: dict[tuple[int, int], pd.DataFrame] = {}
_mid_cache: dict[str, dict] = {}


def clear_caches() -> None:
    """일일 갱신 등에서 캐시 비울 때 호출."""
    _short_cache.clear()
    _mid_cache.clear()


def get_short_daily(nx: int, ny: int) -> pd.DataFrame | None:
    """(nx, ny) 단기예보 → 일별 집계 (캐시 사용)."""
    key = (nx, ny)
    if key in _short_cache:
        return _short_cache[key]
    bd, bt = latest_short_base()
    raw = fetch_short_forecast(nx, ny, bd, bt)
    if raw is None:
        _short_cache[key] = None  # type: ignore
        return None
    daily = aggregate_short_to_daily(raw)
    _short_cache[key] = daily
    return daily


def get_mid_temp_range(reg_id: str, today: _dt.date) -> dict[_dt.date, float]:
    """regId 중기기온 → {D+4~D+10 date: temp_range} (캐시 사용)."""
    if reg_id in _mid_cache:
        return _mid_cache[reg_id]
    tm_fc = latest_mid_base()
    mr = fetch_mid_temperature(reg_id, tm_fc)
    ranges = mid_temp_range_series(mr, today)
    _mid_cache[reg_id] = ranges
    return ranges


# ─── 일기예보 → weather_temp_range 시계열 ────────────────────────────────────

def build_temp_range_series(
    station: str,
    horizon_dates: list[_dt.date],
    today: _dt.date | None = None,
    fallback_value: float | None = None,
) -> list[float | None]:
    """horizon_dates 길이만큼 weather_temp_range 리스트 반환.

    매핑 규칙:
      - 첫 3일: 단기예보 일별 (tmx-tmn)
      - 그 외: 중기기온 (taMax-taMin)
      - 둘 다 실패한 일자: fallback_value (보통 historical 평균)
    """
    if station not in STATION_TO_GRID:
        return [fallback_value] * len(horizon_dates)
    today = today or _dt.date.today()
    nx, ny = STATION_TO_GRID[station]
    reg = STATION_TO_MID_REGID[station]

    short_daily = get_short_daily(nx, ny)
    mid_ranges = get_mid_temp_range(reg, today)

    short_ranges: dict[_dt.date, float] = {}
    if short_daily is not None:
        for _, r in short_daily.iterrows():
            if r["temp_range"] is not None and not pd.isna(r["temp_range"]):
                short_ranges[r["date"]] = float(r["temp_range"])

    # 운영 시: 모델 horizon 일자 = today + 1 .. today + N (full_baseline 마지막 = today-1 가정)
    # 평가/오프라인 시: horizon_dates 가 today 와 무관할 수 있어 인덱스 기반 매핑
    forecast_dates_api = [today + _dt.timedelta(days=i + 1) for i in range(len(horizon_dates))]

    out: list[float | None] = []
    for api_d in forecast_dates_api:
        delta = (api_d - today).days  # 1~N
        if delta <= 3 and api_d in short_ranges:
            out.append(short_ranges[api_d])
        elif api_d in mid_ranges:
            out.append(mid_ranges[api_d])
        elif api_d in short_ranges:  # D+4·5 단기 보조
            out.append(short_ranges[api_d])
        else:
            out.append(fallback_value)
    return out


def build_known_covariates_frame(
    full_baseline: pd.DataFrame,
    item_station_map: dict[str, str],
    horizon: int = 10,
    today: _dt.date | None = None,
) -> pd.DataFrame:
    """46품목 × horizon 일 known_covariates DataFrame.

    반환 컬럼: item_id, timestamp, market_rest, weather_temp_range,
              weather_sunshine_dur, bok_base_rate, cpi_growth_rate

    Args:
        full_baseline : full_baseline.parquet (또는 _extended) DataFrame.
                        MultiIndex(item_id, timestamp) 또는 reset 상태 둘 다 허용.
        item_station_map : meta_baseline.json 의 "item_station_map" dict.
        horizon : 예측 horizon (기본 10일, Chronos2).
        today : 일기예보 기준 날짜 (기본 datetime.date.today()).
    """
    if isinstance(full_baseline.index, pd.MultiIndex):
        full_baseline = full_baseline.reset_index()
    full_baseline = full_baseline.copy()
    full_baseline["timestamp"] = pd.to_datetime(full_baseline["timestamp"])

    rows = []
    for item_id, station in item_station_map.items():
        sub = full_baseline[full_baseline["item_id"] == item_id].sort_values("timestamp")
        if sub.empty:
            continue
        last_ts = sub["timestamp"].max()
        # horizon 일자 = 모델이 예측하는 날짜 (마지막 timestamp 다음날부터)
        horizon_dates = [(last_ts + pd.Timedelta(days=i + 1)).date() for i in range(horizon)]

        # historical fallback (최근 30일 평균 일교차)
        recent_tr = sub["weather_temp_range"].tail(30).mean()
        fb = float(recent_tr) if pd.notna(recent_tr) else 10.0

        tr_series = build_temp_range_series(station, horizon_dates, today, fallback_value=fb)

        # 나머지 known 4종은 ffill (마지막 관측치)
        last_row = sub.iloc[-1]
        sun = float(last_row.get("weather_sunshine_dur", 0.0) or 0.0)
        rest = int(last_row.get("market_rest", 0) or 0)
        bok = float(last_row.get("bok_base_rate", 0.0) or 0.0)
        cpi = float(last_row.get("cpi_growth_rate", 0.0) or 0.0)

        for h_date, tr in zip(horizon_dates, tr_series):
            rows.append({
                "item_id": item_id,
                "timestamp": pd.Timestamp(h_date),
                "market_rest": rest,
                "weather_temp_range": float(tr) if tr is not None else fb,
                "weather_sunshine_dur": sun,
                "bok_base_rate": bok,
                "cpi_growth_rate": cpi,
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import json
    FH = Path(__file__).resolve().parent.parent
    BASELINE = FH / "01_DATA/data/full_baseline_extended_20260516.parquet"
    META = FH / "01_DATA/meta_baseline.json"
    df = pd.read_parquet(BASELINE)
    meta = json.loads(META.read_text(encoding="utf-8"))
    ism = meta["item_station_map"]

    known = build_known_covariates_frame(df, ism, horizon=10)
    print(f"[known_covariates] rows={len(known)} items={known['item_id'].nunique()}")
    sample = known[known["item_id"] == "apple_fuji_box10kg_high"]
    print("\n[apple_fuji_box10kg_high 10일 known 프레임]")
    print(sample.to_string(index=False))
    print(f"\nNaN check: {known.isna().sum().to_dict()}")
