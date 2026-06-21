# -*- coding: utf-8 -*-
"""Chronos-2 10일 예측.

기본값은 AWS에 배포된 final_handoff Chronos2 LoRA API를 사용한다.
필요하면 AGRI_USE_FINAL_HANDOFF_API=false 로 기존 로컬 Chronos 추론으로 되돌릴 수 있다.
"""
from __future__ import annotations

from datetime import date, timedelta
from urllib import error, request
import json

import numpy as np
from django.conf import settings

from .agri_data import get_item_frame, get_last_data_date
from .agri_store import load_actuals_from_firestore


_GRADE_MAP = {
    "high": "high",
    "medium": "mid",
    "low": "low",
    "top": "premium",
    "mid": "mid",
    "premium": "premium",
}


def _to_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _expand_prob_quantiles(row: dict) -> dict[str, float | None]:
    """
    API의 주요 분위수(p10,p20,p50,p80,p90)를 기반으로 p1~p99를 선형 보간/외삽한다.
    """
    anchors = {
        10: _to_float(row.get("p10")),
        20: _to_float(row.get("p20")),
        50: _to_float(row.get("p50")),
        80: _to_float(row.get("p80")),
        90: _to_float(row.get("p90")),
    }

    quantiles: dict[str, float | None] = {}
    for q in range(1, 100):
        key = f"p{q}"
        if q in anchors and anchors[q] is not None:
            quantiles[key] = round(float(anchors[q]), 2)
            continue

        low_q = max((k for k, v in anchors.items() if k <= q and v is not None), default=None)
        high_q = min((k for k, v in anchors.items() if k >= q and v is not None), default=None)

        if low_q is not None and high_q is not None and low_q != high_q:
            low_v = anchors[low_q]
            high_v = anchors[high_q]
            if low_v is not None and high_v is not None:
                ratio = (q - low_q) / (high_q - low_q)
                quantiles[key] = round(low_v + (high_v - low_v) * ratio, 2)
                continue

        if q < 10 and anchors[10] is not None and anchors[20] is not None:
            slope = (anchors[20] - anchors[10]) / 10.0
            quantiles[key] = round(anchors[10] + slope * (q - 10), 2)
            continue
        if q > 90 and anchors[80] is not None and anchors[90] is not None:
            slope = (anchors[90] - anchors[80]) / 10.0
            quantiles[key] = round(anchors[90] + slope * (q - 90), 2)
            continue

        quantiles[key] = None

    return quantiles


def _grade_suffix(item_id: str) -> str | None:
    suffix = item_id.rsplit("_", 1)[-1]
    return _GRADE_MAP.get(suffix)


def _latest_actual_date_for_item(item_id: str) -> date:
    latest = get_last_data_date(item_id)
    if getattr(settings, "USE_FIRESTORE", False):
        fs = load_actuals_from_firestore(item_id)
        if fs:
            latest_fs = max(date.fromisoformat(d) for d in fs.keys())
            if latest_fs > latest:
                latest = latest_fs
    return latest


def _final_handoff_item_id(item_id: str) -> str:
    """기존 Django item_id를 final_handoff 모델 item_id로 변환."""
    grade = _grade_suffix(item_id)
    if not grade:
        raise ValueError(f"지원하지 않는 등급 코드입니다: {item_id}")

    final_prefixes = (
        "apple_fuji_box10kg_",
        "cabbage_net8kg_",
        "carrot_box20kg_",
        "crown_daisy_box4kg_",
        "cucumber_bdadagi_ea100_",
        "garlic_chive_bundle500g_",
        "honewort_kg4_",
        "napa_cabbage_net10kg_",
        "onion_kg1_",
        "perilla_leaf_bunch100_",
        "potato_sumi_box20kg_",
        "spinach_box4kg_",
        "sweetpotato_box10kg_",
    )
    if item_id.startswith(final_prefixes):
        return item_id

    prefix_map = (
        ("apple_apple_10kg_", "apple_fuji_box10kg"),
        ("cabbage_8_net_", "cabbage_net8kg"),
        ("cabbage_10_net_", "napa_cabbage_net10kg"),
        ("carrot_carrot_20kg_", "carrot_box20kg"),
        ("cucumber_100ea_", "cucumber_bdadagi_ea100"),
        ("onion_1_", "onion_kg1"),
        ("perilla_leaf_100_", "perilla_leaf_bunch100"),
        ("potato_potato_20kg_", "potato_sumi_box20kg"),
        ("spinach_spinach_4kg_", "spinach_box4kg"),
        ("sweet_potato_sweet_potato_1kg_", "sweetpotato_box10kg"),
    )
    for django_prefix, final_prefix in prefix_map:
        if item_id.startswith(django_prefix):
            return f"{final_prefix}_{grade}"

    raise ValueError(
        "final_handoff 모델에 매핑된 품목이 아닙니다. "
        "현재 AWS 구간 예측(10일)/점예측(3일)은 full_baseline_extended_20260516.parquet의 "
        "46개 원본 품목에 연결되어 있습니다."
    )


def supports_final_handoff_item(item_id: str) -> bool:
    try:
        _final_handoff_item_id(item_id)
    except ValueError:
        return False
    return True


def _run_final_handoff_forecast(
    item_id: str,
    model: str,
    pred_len: int,
) -> tuple[date, list[dict]]:
    model_item_id = _final_handoff_item_id(item_id)
    api_url = getattr(settings, "AGRI_FINAL_HANDOFF_API_URL", "").rstrip("/")
    if not api_url:
        raise ValueError("AGRI_FINAL_HANDOFF_API_URL 설정이 비어 있습니다.")

    if model not in {"point", "probabilistic"}:
        raise ValueError(f"지원하지 않는 예측 모델입니다: {model}")

    payload = json.dumps(
        {"model": model, "item_id": model_item_id}
    ).encode("utf-8")
    req = request.Request(
        f"{api_url}/api/predict",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=getattr(settings, "AGRI_FINAL_HANDOFF_TIMEOUT", 300)) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"final_handoff API error {e.code}: {detail}") from e
    except error.URLError as e:
        raise RuntimeError(f"final_handoff API 연결 실패: {e}") from e

    rows = data.get("outputs", {}).get(model, {}).get("rows", [])
    if not rows:
        raise RuntimeError("final_handoff API가 예측 결과를 반환하지 않았습니다.")

    rows = rows[:pred_len]
    origin = _latest_actual_date_for_item(item_id)
    points = []
    for idx, row in enumerate(rows):
        forecast_date = (origin + timedelta(days=idx + 1)).isoformat()
        if model == "point":
            pred = row["y_pred"]
            source_model = "final_handoff/TimesFM_2.5_ZeroShot"
            extra = {"y_pred": row.get("y_pred")}
        else:
            # README 기준 중앙값(0.5)을 점예측 대용으로 사용한다.
            pred = row.get("p50")
            if pred is None:
                pred = row.get("mean")
            source_model = "final_handoff/Chronos2LoRA_baseline"
            quantiles = _expand_prob_quantiles(row)
            extra = {
                "mean": row.get("mean"),
                "p10": row.get("p10"),
                "p20": row.get("p20"),
                "p50": row.get("p50"),
                "p80": row.get("p80"),
                "p90": row.get("p90"),
                **quantiles,
            }
        points.append(
            {
                "date": forecast_date,
                "pred_krw": float(pred),
                "reconciled": False,
                "source_model": source_model,
                "source_item_id": model_item_id,
                "source_timestamp": row.get("timestamp"),
                **extra,
            }
        )

    return origin, points


def _dummy_forecast(last_target: float, pred_len: int, last_date: date) -> tuple[list[date], np.ndarray]:
    noise = np.linspace(0, 0.02, pred_len)
    pred_arcsinh = last_target * (1.0 + noise)
    dates = [last_date + timedelta(days=i + 1) for i in range(pred_len)]
    return dates, pred_arcsinh


def run_chronos_forecast(
    item_id: str,
    model: str = "probabilistic",
) -> tuple[date, list[dict]]:
    """
    마지막 관측일 기준 익일부터 FORECAST_DAYS일 예측.
    반환: (origin_date = 마지막 관측일, points[{date, pred_krw, reconciled: False}])
    """
    ctx_len = settings.AGRI_CONTEXT_DAYS
    pred_len = settings.AGRI_FORECAST_DAYS

    if getattr(settings, "AGRI_USE_FINAL_HANDOFF_API", True):
        return _run_final_handoff_forecast(item_id, model, pred_len)

    sub = get_item_frame(item_id)
    n = len(sub)
    if n < ctx_len + 1:
        raise ValueError(
            f"{item_id}: 데이터 부족 (행 {n}, 필요 최소 {ctx_len + 1})"
        )

    target = sub["target"].values.astype(float)
    last_row = sub.iloc[-1]
    last_date = last_row["date"].date()
    context = target[-ctx_len:]

    if getattr(settings, "CHRONOS_SKIP_INFERENCE", False):
        dates, pred_a = _dummy_forecast(float(context[-1]), pred_len, last_date)
    else:
        import torch
        from chronos.chronos2.pipeline import Chronos2Pipeline as TSFMPipeline

        pipeline = TSFMPipeline.from_pretrained(
            "amazon/chronos-2",
            device_map="auto",
            torch_dtype=torch.float16,
        )
        pipeline.model.eval()
        pred_input = {"target": context}
        result = pipeline.predict([pred_input], prediction_length=pred_len)
        samples = list(result)[0].numpy()[0]
        pred_a = np.median(samples, axis=0)
        dates = [last_date + timedelta(days=i + 1) for i in range(pred_len)]

    points = []
    for d, a in zip(dates, pred_a):
        pred_krw = float(np.sinh(a))
        points.append(
            {
                "date": d.isoformat(),
                "pred_krw": pred_krw,
                "reconciled": False,
            }
        )

    return last_date, points
