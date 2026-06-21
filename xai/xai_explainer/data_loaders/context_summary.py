"""365일 raw context를 품목별 요약 통계 텍스트로 압축."""
from datetime import timedelta

import numpy as np
import pandas as pd

from config import FULL_BASELINE_PATH

NUMERIC_COVARS = [
    "amount",
    "weather_rain_sum",
    "weather_temp_range",
    "weather_sunshine_dur",
    "weather_wind_avg",
    "weather_humidity_avg",
    "weather_pressure_avg",
    "oil_tax_free_diesel",
    "bok_base_rate",
    "cpi_growth_rate",
    "news_sentiment_index",
]


def load_full_baseline() -> pd.DataFrame:
    df = pd.read_parquet(FULL_BASELINE_PATH)
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values(["item_id", "timestamp"]).reset_index(drop=True)


def _trend_sign(values: np.ndarray) -> str:
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


def _safe_stats(s: pd.Series) -> dict:
    s = s.dropna()
    if s.empty:
        return {"mean": None, "std": None, "min": None, "max": None, "last": None}
    return {
        "mean": float(s.mean()),
        "std":  float(s.std()),
        "min":  float(s.min()),
        "max":  float(s.max()),
        "last": float(s.iloc[-1]),
    }


def _fmt_num(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    if abs(v) >= 10:
        return f"{v:,.1f}"
    return f"{v:.2f}"


def summarize_item(df: pd.DataFrame, item_id: str) -> str:
    """단일 품목 365일 요약을 짧은 마크다운 텍스트로 반환."""
    sub = df[df["item_id"] == item_id].copy()
    if sub.empty:
        return f"(no data for {item_id})"

    sub = sub.sort_values("timestamp")
    last_ts = sub["timestamp"].max()
    win = {
        7:   sub[sub["timestamp"] > last_ts - timedelta(days=7)],
        30:  sub[sub["timestamp"] > last_ts - timedelta(days=30)],
        90:  sub[sub["timestamp"] > last_ts - timedelta(days=90)],
        365: sub[sub["timestamp"] > last_ts - timedelta(days=365)],
    }

    target_stats = {k: _safe_stats(v["target"]) for k, v in win.items()}
    target_last = target_stats[7]["last"]

    # 작년 동기간(±15일) 평균 대비
    yoy_window = sub[
        (sub["timestamp"] >= last_ts - timedelta(days=365 + 15))
        & (sub["timestamp"] <= last_ts - timedelta(days=365 - 15))
    ]
    yoy_mean = float(yoy_window["target"].mean()) if not yoy_window.empty else None
    if yoy_mean and target_last:
        yoy_delta_pct = (target_last - yoy_mean) / yoy_mean * 100
    else:
        yoy_delta_pct = None

    trend_30 = _trend_sign(win[30]["target"].to_numpy(dtype=float))

    lines = [
        f"- 데이터 마지막 일자: {last_ts.date()}",
        f"- target 최근값: ₩{_fmt_num(target_last)}",
        f"- target 7일 평균/표준편차: ₩{_fmt_num(target_stats[7]['mean'])} / "
        f"{_fmt_num(target_stats[7]['std'])}",
        f"- target 30일 평균/표준편차: ₩{_fmt_num(target_stats[30]['mean'])} / "
        f"{_fmt_num(target_stats[30]['std'])}",
        f"- target 90일 평균: ₩{_fmt_num(target_stats[90]['mean'])}",
        f"- target 365일 평균/최저/최고: ₩{_fmt_num(target_stats[365]['mean'])} / "
        f"₩{_fmt_num(target_stats[365]['min'])} / ₩{_fmt_num(target_stats[365]['max'])}",
        f"- 최근 30일 추세: {trend_30}",
    ]
    if yoy_delta_pct is not None:
        lines.append(
            f"- 작년 동기(±15일) 평균 ₩{_fmt_num(yoy_mean)} 대비: {yoy_delta_pct:+.1f}%"
        )

    cov_lines = []
    for c in NUMERIC_COVARS:
        if c not in sub.columns:
            continue
        s7 = _safe_stats(win[7][c])
        s365 = _safe_stats(win[365][c])
        if s7["mean"] is None and s365["mean"] is None:
            continue
        cov_lines.append(
            f"  - {c}: 최근 7일 평균 {_fmt_num(s7['mean'])} / "
            f"365일 평균 {_fmt_num(s365['mean'])}"
        )
    if cov_lines:
        lines.append("- 공변량 (최근 7일 vs 365일 평균):")
        lines.extend(cov_lines)

    return "\n".join(lines)


if __name__ == "__main__":
    df = load_full_baseline()
    print(summarize_item(df, "apple_fuji_box10kg_high"))
