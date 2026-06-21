# -*- coding: utf-8 -*-
"""농업 시계열 CSV 로드 및 일자별 실제가 조회."""
from __future__ import annotations

import os
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from django.conf import settings
from django.utils import timezone


_df_cache: pd.DataFrame | None = None
_final_df_cache: pd.DataFrame | None = None

_GRADE_LABELS = {
    "top": "특",
    "high": "상",
    "medium": "중",
    "low": "하",
    "mid": "중",
    "premium": "특",
}

_CROP_PREFIX_LABELS: tuple[tuple[str, str], ...] = (
    ("apple_fuji_box10kg_", "사과 부사"),
    ("cabbage_net8kg_", "양배추"),
    ("carrot_box20kg_", "당근"),
    ("crown_daisy_box4kg_", "쑥갓"),
    ("cucumber_bdadagi_ea100_", "백다다기오이"),
    ("garlic_chive_bundle500g_", "부추"),
    ("honewort_kg4_", "미나리"),
    ("napa_cabbage_net10kg_", "배추"),
    ("onion_kg1_", "양파"),
    ("perilla_leaf_bunch100_", "깻잎"),
    ("potato_sumi_box20kg_", "감자 수미"),
    ("spinach_box4kg_", "시금치"),
    ("sweetpotato_box10kg_", "고구마"),
    ("sweet_potato_", "고구마"),
    ("green_onion_", "대파"),
    ("green_chili_", "풋고추"),
    ("perilla_leaf_", "깻잎"),
    ("chwinamul_", "취나물"),
    ("cabbage_8_net_", "양배추"),
    ("cabbage_10_net_", "배추"),
    ("cabbage_", "양배추"),
    ("radish_", "무"),
    ("garlic_", "마늘"),
    ("onion_", "양파"),
    ("apple_", "사과"),
    ("pear_", "배"),
    ("carrot_", "당근"),
    ("cucumber_", "백다다기오이"),
    ("potato_", "감자"),
    ("spinach_", "시금치"),
    ("mandarin_", "감귤"),
    ("lettuce_", "상추"),
)

_PREFIX_UNIT_LABELS = {
    "apple_fuji_box10kg_": "10kg box",
    "cabbage_net8kg_": "8kg net",
    "cabbage_8_net_": "8kg net",
    "cabbage_10_net_": "10kg net",
    "carrot_box20kg_": "20kg box",
    "crown_daisy_box4kg_": "4kg box",
    "cucumber_bdadagi_ea100_": "100개",
    "garlic_chive_bundle500g_": "500g bundle",
    "honewort_kg4_": "4kg",
    "napa_cabbage_net10kg_": "10kg net",
    "onion_kg1_": "1kg",
    "perilla_leaf_bunch100_": "100속",
    "potato_sumi_box20kg_": "20kg box",
    "spinach_box4kg_": "4kg box",
    "sweetpotato_box10kg_": "10kg box",
}


def _final_baseline_path() -> Path | None:
    configured = os.environ.get("AGRI_FINAL_BASELINE_PARQUET", "").strip()
    candidates = [
        Path(configured) if configured else None,
        Path(settings.BASE_DIR) / "final_handoff" / "final_handoff" / "PROBABILISTIC_FORECAST" / "data" / "full_baseline.parquet",
        Path(settings.BASE_DIR).parent / "capstone_forecast" / "final_handoff" / "PROBABILISTIC_FORECAST" / "data" / "full_baseline.parquet",
        Path("/home/ubuntu/capstone_forecast/final_handoff/PROBABILISTIC_FORECAST/data/full_baseline.parquet"),
    ]
    for path in candidates:
        if path and path.is_file():
            return path
    return None


def _final_dataframe() -> pd.DataFrame | None:
    global _final_df_cache
    if _final_df_cache is not None:
        return _final_df_cache
    path = _final_baseline_path()
    if path is None:
        return None
    df = pd.read_parquet(path)
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    if "timestamp" in df.columns:
        df = df.rename(columns={"timestamp": "date"})
    df["date"] = pd.to_datetime(df["date"])
    df["price"] = pd.to_numeric(df["target"], errors="coerce")
    df = df.sort_values(["item_id", "date"]).reset_index(drop=True)
    _final_df_cache = df
    return df


def is_allowed_item_id(item_id: str) -> bool:
    if not item_id or not isinstance(item_id, str):
        return False
    try:
        df = _final_dataframe()
        if df is None:
            df = _full_dataframe()
    except Exception:
        return True
    return item_id in set(map(str, df["item_id"].unique().tolist()))


def crop_ko_name(item_id: str) -> str:
    """품목 한글명(대분류)."""
    for prefix, label in _CROP_PREFIX_LABELS:
        if item_id.startswith(prefix):
            return label
    return "기타"


def item_option_meta(item_id: str) -> dict[str, str]:
    """item_id를 화면 선택용 작물/단위명/등급명으로 분리한다."""
    crop_label = crop_ko_name(item_id)
    parts = item_id.split("_")
    grade_code = parts[-1] if parts and parts[-1] in _GRADE_LABELS else ""
    grade_label = _GRADE_LABELS.get(grade_code, grade_code or "기본")

    crop_prefix = ""
    for prefix, _label in _CROP_PREFIX_LABELS:
        if item_id.startswith(prefix):
            crop_prefix = prefix
            break

    unit_raw = item_id
    if crop_prefix:
        if crop_prefix in _PREFIX_UNIT_LABELS:
            unit_raw = _PREFIX_UNIT_LABELS[crop_prefix]
        else:
            unit_raw = item_id[len(crop_prefix):]
            crop_code = crop_prefix.strip("_")
            if unit_raw.startswith(crop_code + "_"):
                unit_raw = unit_raw[len(crop_code) + 1 :]
            if grade_code and unit_raw.endswith("_" + grade_code):
                unit_raw = unit_raw[: -(len(grade_code) + 1)]
    unit_label = unit_raw.replace("_", " · ") if unit_raw else "기본"

    return {
        "id": item_id,
        "label": f"{crop_label} · {unit_label} · {grade_label}",
        "crop": crop_label,
        "unit": unit_label,
        "grade": grade_label,
        "grade_code": grade_code,
    }


def format_item_option_label(item_id: str) -> str:
    """드롭다운·범례용: 한글 품목 + 세부 코드."""
    meta = item_option_meta(item_id)
    return f"{meta['crop']} · {meta['unit']} · {meta['grade']}"


def _full_dataframe() -> pd.DataFrame:
    global _df_cache
    if _df_cache is not None:
        return _df_cache
    train = Path(settings.AGRI_TRAIN_CSV)
    test = Path(settings.AGRI_TEST_CSV)
    if not train.is_file():
        raise FileNotFoundError(f"AGRI train CSV not found: {train}")
    parts = [pd.read_csv(train, parse_dates=["date"])]
    if test.is_file():
        parts.append(pd.read_csv(test, parse_dates=["date"]))
    df = pd.concat(parts, ignore_index=True)
    df = df.sort_values(["item_id", "date"]).reset_index(drop=True)
    _df_cache = df
    return df


def list_item_ids() -> list[str]:
    df = _final_dataframe()
    if df is None:
        df = _full_dataframe()
    ids = [str(x) for x in df["item_id"].unique().tolist()]
    return sorted(ids)


def get_item_frame(item_id: str) -> pd.DataFrame:
    if not is_allowed_item_id(item_id):
        raise ValueError(
            f"허용되지 않은 품목입니다: {item_id!r}"
        )
    df = _final_dataframe()
    if df is None:
        df = _full_dataframe()
    sub = df[df["item_id"] == item_id].sort_values("date").reset_index(drop=True)
    if sub.empty:
        raise ValueError(f"Unknown item_id: {item_id}")
    return sub


def price_krw_from_row(row: pd.Series) -> float:
    if "price" in row.index and pd.notna(row["price"]):
        return float(row["price"])
    return float(np.sinh(row["target"]))


def get_actual_price_for_date(item_id: str, d: date) -> Optional[float]:
    """CSV에서 해당 일자 실제가(원). 없으면 None."""
    sub = get_item_frame(item_id)
    hit = sub[sub["date"].dt.date == d]
    if hit.empty:
        return None
    return price_krw_from_row(hit.iloc[0])


def get_actual_prices_range(
    item_id: str, start: date, end: date
) -> dict[str, float]:
    """start~end (포함) 일자별 실제가."""
    sub = get_item_frame(item_id)
    mask = (sub["date"].dt.date >= start) & (sub["date"].dt.date <= end)
    out: dict[str, float] = {}
    for _, row in sub[mask].iterrows():
        d = row["date"].date().isoformat()
        out[d] = price_krw_from_row(row)
    return out


def get_last_data_date(item_id: str) -> date:
    sub = get_item_frame(item_id)
    return sub["date"].iloc[-1].date()


def get_data_updated_at() -> datetime | None:
    """Return the newest modified time among configured agri CSV files."""
    paths = [
        Path(settings.AGRI_TRAIN_CSV),
        Path(settings.AGRI_TEST_CSV),
    ]
    mtimes = [p.stat().st_mtime for p in paths if p.is_file()]
    if not mtimes:
        return None
    return datetime.fromtimestamp(max(mtimes), tz=timezone.get_current_timezone())


def fetch_actual_from_external_api(item_id: str, d: date) -> Optional[float]:
    """
    선택적 외부 API. AGRIDATA_API_URL 이 비어 있으면 항상 None.
    운영 시 requests로 구현해 반환하면 CSV보다 우선해 reconcile에 사용 가능.
    """
    url = getattr(settings, "AGRIDATA_API_URL", "") or ""
    if not url.strip():
        return None
    # 확장 지점: 실제 API 스펙에 맞게 구현
    return None


# GPT 설명용: parquet 컬럼명 → 사용자에게 읽히는 한글 설명
_COVARIATE_ORDER: tuple[tuple[str, str], ...] = (
    ("weather_temp_range", "기온 변화(일교차)"),
    ("weather_rain_sum", "강수(비) 양"),
    ("weather_humidity_avg", "습도"),
    ("weather_sunshine_dur", "일조 시간"),
    ("weather_wind_avg", "풍속"),
    ("weather_pressure_avg", "기압"),
    ("amount", "거래량"),
    ("oil_tax_free_diesel", "경유(유가)"),
    ("bok_base_rate", "기준금리"),
    ("cpi_growth_rate", "물가 상승률"),
    ("news_sentiment_index", "뉴스 심리 지수"),
    ("market_rest", "시장 휴장 여부"),
)


def covariate_and_news_context_for_explain(
    item_id: str, recent_days: int = 14
) -> dict[str, Any]:
    """
    최근 구간 기상·거시 특성 요약 + 선택적 뉴스 텍스트(파일).
    뉴스는 AGRI_NEWS_CONTEXT_PATH 가 가리키는 UTF-8 파일이 있을 때만 포함.
    """
    sub = get_item_frame(item_id)
    n = len(sub)
    if n < 2:
        return {"안내": "데이터가 부족합니다."}

    span = min(recent_days, n)
    tail = sub.tail(span)
    cols_labels = [(c, lab) for c, lab in _COVARIATE_ORDER if c in sub.columns]

    지표_요약: dict[str, Any] = {}
    for c, label in cols_labels:
        s = tail[c].astype(float)
        지표_요약[label] = {
            "이_기간_평균": round(float(s.mean()), 4),
            "이_기간_가장_낮을_때와_높을_때": [
                round(float(s.min()), 4),
                round(float(s.max()), 4),
            ],
        }

    이전_같은_기간_대비: dict[str, float] | None = None
    if n > span:
        prev = sub.iloc[-2 * span : -span]
        if len(prev) >= 1:
            이전_같은_기간_대비 = {}
            for c, label in cols_labels:
                이전_같은_기간_대비[label] = round(
                    float(tail[c].mean()) - float(prev[c].mean()), 4
                )

    news_txt: str | None = None
    path = getattr(settings, "AGRI_NEWS_CONTEXT_PATH", "") or ""
    if path and os.path.isfile(path):
        try:
            raw = Path(path).read_text(encoding="utf-8", errors="replace").strip()
            news_txt = raw[:8000] if raw else None
        except OSError:
            news_txt = None

    d0 = tail["date"].iloc[0]
    d1 = tail["date"].iloc[-1]
    d0s = d0.date().isoformat() if hasattr(d0, "date") else str(d0)
    d1s = d1.date().isoformat() if hasattr(d1, "date") else str(d1)

    return {
        "요약_기간_일수": int(span),
        "요약_기간_날짜": f"{d0s} ~ {d1s}",
        "기상과_경제_지표_요약": 지표_요약,
        "직전_같은_길이_기간과_비교한_평균_변화": 이전_같은_기간_대비,
        "수치_읽는_법": (
            "아래 숫자는 모델이 학습할 때 쓴 스케일(정규화) 값이라, "
            "실제 섭씨 몇 도·몇 mm와 숫자가 똑같이 대응하지는 않을 수 있습니다. "
            "‘최근이 이전보다 큰지/작은지’ 같은 상대 비교에만 쓰면 됩니다."
        ),
        "참고_뉴스_글": news_txt,
        "뉴스_안내": (
            None
            if news_txt
            else "뉴스 글 파일이 연결되어 있지 않습니다. 뉴스 언급 없이 기상·경제 지표만 근거로 쓰세요."
        ),
    }


def _fmt_context_num(v: Any) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    v = float(v)
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    if abs(v) >= 10:
        return f"{v:,.1f}"
    return f"{v:.2f}"


def _context_trend_sign(values: np.ndarray) -> str:
    if len(values) < 5 or np.isnan(values).all():
        return "데이터부족"
    x = np.arange(len(values), dtype=float)
    y = np.where(np.isnan(values), np.nanmean(values), values)
    slope = np.polyfit(x, y, 1)[0]
    if slope > 1e-6:
        return f"상승(기울기 {slope:+.2f})"
    if slope < -1e-6:
        return f"하락(기울기 {slope:+.2f})"
    return "보합"


def summarize_baseline_context_for_explain(item_id: str) -> str:
    """XAI용 365일 baseline 요약(모델 context)."""
    sub = get_item_frame(item_id).sort_values("date").copy()
    if sub.empty:
        return f"(품목 {item_id} baseline 없음)"

    last_ts = sub["date"].max()
    win = {
        7: sub[sub["date"] > last_ts - pd.Timedelta(days=7)],
        30: sub[sub["date"] > last_ts - pd.Timedelta(days=30)],
        90: sub[sub["date"] > last_ts - pd.Timedelta(days=90)],
        365: sub[sub["date"] > last_ts - pd.Timedelta(days=365)],
    }

    def _stats(frame: pd.DataFrame, col: str) -> dict[str, float | None]:
        s = pd.to_numeric(frame[col], errors="coerce").dropna()
        if s.empty:
            return {"mean": None, "std": None, "min": None, "max": None, "last": None}
        return {
            "mean": float(s.mean()),
            "std": float(s.std()),
            "min": float(s.min()),
            "max": float(s.max()),
            "last": float(s.iloc[-1]),
        }

    target_stats = {k: _stats(v, "target") for k, v in win.items()}
    target_last = target_stats[7]["last"]
    trend_30 = _context_trend_sign(win[30]["target"].to_numpy(dtype=float))

    lines = [
        f"- 데이터 마지막 일자: {last_ts.date()}",
        f"- 최근 가격(target) 최근값: ₩{_fmt_context_num(target_last)}",
        f"- 최근 7일 평균/표준편차: ₩{_fmt_context_num(target_stats[7]['mean'])} / "
        f"{_fmt_context_num(target_stats[7]['std'])}",
        f"- 최근 30일 평균/표준편차: ₩{_fmt_context_num(target_stats[30]['mean'])} / "
        f"{_fmt_context_num(target_stats[30]['std'])}",
        f"- 최근 90일 평균: ₩{_fmt_context_num(target_stats[90]['mean'])}",
        f"- 최근 365일 평균/최저/최고: ₩{_fmt_context_num(target_stats[365]['mean'])} / "
        f"₩{_fmt_context_num(target_stats[365]['min'])} / ₩{_fmt_context_num(target_stats[365]['max'])}",
        f"- 최근 30일 가격 추세: {trend_30}",
    ]

    cov_lines: list[str] = []
    for col, label in _COVARIATE_ORDER:
        if col not in sub.columns:
            continue
        s7 = _stats(win[7], col)
        s365 = _stats(win[365], col)
        if s7["mean"] is None and s365["mean"] is None:
            continue
        cov_lines.append(
            f"  - {label}: 최근 7일 평균 {_fmt_context_num(s7['mean'])} / "
            f"365일 평균 {_fmt_context_num(s365['mean'])}"
        )
    if cov_lines:
        lines.append("- 기상·경제·수급 공변량 (최근 7일 vs 365일 평균):")
        lines.extend(cov_lines)
    return "\n".join(lines)


def format_weather_economic_block_for_explain(cov: dict[str, Any]) -> str:
    """기상특보 API가 없을 때 모델 입력 기상·경제 지표를 프롬프트용 텍스트로."""
    indicators = cov.get("기상과_경제_지표_요약") or {}
    changes = cov.get("직전_같은_기간과_비교한_평균_변화") or {}
    period = cov.get("요약_기간_날짜") or ""
    if not indicators:
        return "(최근 기상·경제 공변량을 불러오지 못했습니다.)"

    lines = [
        f"### 모델 입력 기상·경제 지표 ({period})",
        "아래 수치는 모델 학습 데이터에 포함된 기상·거래·유가·물가 지표입니다.",
    ]
    for label, stats in indicators.items():
        avg = stats.get("이_기간_평균")
        lo, hi = stats.get("이_기간_가장_낮을_때와_높을_때") or [None, None]
        delta = changes.get(label) if isinstance(changes, dict) else None
        delta_txt = f", 직전 동기간 대비 {delta:+.4f}" if delta is not None else ""
        lines.append(
            f"- {label}: 최근 평균 {avg}, 범위 {lo}~{hi}{delta_txt}"
        )
    reading = cov.get("수치_읽는_법")
    if reading:
        lines.append(f"- 참고: {reading}")
    return "\n".join(lines)
