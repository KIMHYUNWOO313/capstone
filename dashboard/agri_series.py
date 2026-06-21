# -*- coding: utf-8 -*-
"""차트용 시계열: 과거 실제 + (미정산) 예측."""
from __future__ import annotations

from datetime import date, timedelta
import calendar
import math

from django.conf import settings
from .agri_data import (
    crop_ko_name,
    format_item_option_label,
    get_data_updated_at,
    get_item_frame,
    get_last_data_date,
    price_krw_from_row,
)
from .agri_store import get_django_actual_map, get_latest_predict_batch, load_actuals_from_firestore


def build_chart_series(
    item_id: str,
    past_days: int | None = None,
    model_name: str = "probabilistic",
    include_prediction: bool = True,
    include_firestore_actuals: bool = True,
) -> dict:
    """
    points: [{date, value, kind: 'actual' | 'predict'}]
    """
    sub = get_item_frame(item_id)
    last_csv = get_last_data_date(item_id)
    extra = {**get_django_actual_map(item_id)}
    if include_firestore_actuals and getattr(settings, "USE_FIRESTORE", False):
        extra.update(load_actuals_from_firestore(item_id))
    extra_dates = [date.fromisoformat(d_str) for d_str in extra.keys()]
    last_actual = max([last_csv, *extra_dates]) if extra_dates else last_csv

    if past_days is None:
        y = last_actual.year
        m = last_actual.month - 3
        while m <= 0:
            y -= 1
            m += 12
        last_day = calendar.monthrange(y, m)[1]
        d = min(last_actual.day, last_day)
        start_hist = date(y, m, d)
    else:
        start_hist = last_actual - timedelta(days=past_days)

    by_date: dict[str, dict] = {}

    mask = (sub["date"].dt.date >= start_hist) & (sub["date"].dt.date <= last_actual)
    for _, row in sub.loc[mask].iterrows():
        d = row["date"].date().isoformat()
        by_date[d] = {
            "date": d,
            "value": round(price_krw_from_row(row), 2),
            "kind": "actual",
        }

    for d_str, price in extra.items():
        d = date.fromisoformat(d_str)
        if d < start_hist or d > last_actual:
            continue
        by_date[d_str] = {
            "date": d_str,
            "value": round(float(price), 2),
            "kind": "actual",
        }

    metrics_batch = get_latest_predict_batch(item_id, model_name=model_name)
    batch = metrics_batch if include_prediction else None
    if batch and batch.get("points"):
        for p in batch["points"]:
            d_str = p["date"]
            d = date.fromisoformat(d_str)
            if d <= last_actual:
                continue
            if p.get("reconciled") and p.get("actual_krw") is not None:
                by_date[d_str] = {
                    "date": d_str,
                    "value": round(float(p["actual_krw"]), 2),
                    "kind": "actual",
                }
            else:
                p10 = p.get("p10")
                p90 = p.get("p90")
                by_date[d_str] = {
                    "date": d_str,
                    "value": round(float(p["pred_krw"]), 2),
                    "kind": "predict",
                    "p10": round(float(p10), 2) if p10 is not None else None,
                    "p90": round(float(p90), 2) if p90 is not None else None,
                }

    points = sorted(by_date.values(), key=lambda x: x["date"])
    data_updated_at = get_data_updated_at()
    actual_values = [float(p["value"]) for p in points if p.get("kind") == "actual"]
    predict_values = []
    confidence_errors = []
    if metrics_batch and metrics_batch.get("points"):
        for p in metrics_batch["points"]:
            pred = p.get("pred_krw")
            if pred is not None:
                predict_values.append(float(pred))
            actual = p.get("actual_krw")
            if p.get("reconciled") and actual is not None and pred is not None:
                actual_f = float(actual)
                pred_f = float(pred)
                if actual_f:
                    confidence_errors.append(abs(pred_f - actual_f) / actual_f)

    recent_actual = actual_values[-10:]
    volatility = None
    if len(recent_actual) >= 2:
        mean = sum(recent_actual) / len(recent_actual)
        if mean:
            variance = sum((v - mean) ** 2 for v in recent_actual) / len(recent_actual)
            volatility = math.sqrt(variance) / mean * 100

    confidence = None
    if confidence_errors:
        mape = sum(confidence_errors) / len(confidence_errors)
        confidence = max(0.0, min(100.0, (1 - mape) * 100))

    metrics = {
        "current_actual_krw": round(actual_values[-1], 2) if actual_values else None,
        "prediction_average_krw": round(sum(predict_values) / len(predict_values), 2)
        if predict_values
        else None,
        "volatility_pct": round(volatility, 2) if volatility is not None else None,
        "confidence_pct": round(confidence, 2) if confidence is not None else None,
        "confidence_sample_size": len(confidence_errors),
    }
    return {
        "item_id": item_id,
        "crop_ko": crop_ko_name(item_id),
        "item_label": format_item_option_label(item_id),
        "model_name": model_name,
        "include_prediction": include_prediction,
        "last_csv_date": last_actual.isoformat(),
        "data_updated_at": data_updated_at.isoformat() if data_updated_at else None,
        "metrics": metrics,
        "points": points,
    }
