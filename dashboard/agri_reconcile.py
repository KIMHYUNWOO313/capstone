# -*- coding: utf-8 -*-
"""
예측 배치 중 과거 일자는 CSV/외부 API로 실제가를 채우고 `actual`에 반영한 뒤
해당 점을 reconciled=True로 표시. 차트에서는 reconciled된 날은 실제가 색으로 표시.
"""
from __future__ import annotations

from datetime import date

from django.conf import settings
from django.utils import timezone

from .agri_data import (
    fetch_actual_from_external_api,
    get_actual_price_for_date,
)
from .agri_store import get_latest_predict_batch, save_actual, update_predict_points


def reconcile_item(item_id: str, model_name: str = "probabilistic") -> dict:
    batch = get_latest_predict_batch(item_id, model_name=model_name)
    if not batch or not batch.get("points"):
        return {"ok": True, "updated": 0, "message": "no predict batch"}

    today = timezone.localdate()
    points = list(batch["points"])
    origin = batch["origin_date"]
    if hasattr(origin, "date") and not isinstance(origin, date):
        origin = origin.date()

    updated = 0
    for p in points:
        d_str = p["date"]
        d = date.fromisoformat(d_str)
        if p.get("reconciled"):
            continue
        if d > today:
            continue

        price = fetch_actual_from_external_api(item_id, d)
        src = "api"
        if price is None:
            price = get_actual_price_for_date(item_id, d)
            src = "csv"
        if price is None:
            continue

        save_actual(item_id, d, price, source=src)
        p["reconciled"] = True
        p["actual_krw"] = price
        updated += 1

    if updated:
        update_predict_points(item_id, origin, points, model_name=model_name)

    return {"ok": True, "updated": updated, "item_id": item_id, "model": model_name}
