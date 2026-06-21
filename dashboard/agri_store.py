# -*- coding: utf-8 -*-
"""Firebase Firestore(`actual`, model-specific predict collections) + Django 폴백."""
from __future__ import annotations

from datetime import date, datetime, timezone
from threading import Lock
from time import monotonic
from typing import Any, Optional

from django.conf import settings
from django.db import transaction

from .models import AgriActual, AgriPredictBatch


_fs_client = None
_fs_init_lock = Lock()
DEFAULT_MODEL_NAME = "probabilistic"
_actual_firestore_cache: dict[str, tuple[float, dict[str, float]]] = {}
_ACTUAL_FIRESTORE_TTL_SECONDS = 600


def _firestore():
    global _fs_client
    if _fs_client is not None:
        return _fs_client
    if not getattr(settings, "USE_FIRESTORE", False):
        return None
    import firebase_admin
    from firebase_admin import credentials, firestore

    with _fs_init_lock:
        try:
            firebase_admin.get_app()
        except ValueError:
            cred = credentials.Certificate(settings.FIREBASE_CREDENTIALS_PATH)
            try:
                firebase_admin.initialize_app(cred)
            except ValueError as exc:
                if "already exists" not in str(exc):
                    raise
    _fs_client = firestore.client()
    return _fs_client


def _doc_actual_id(item_id: str, d: date) -> str:
    return f"{item_id}__{d.isoformat()}"


def _model_name(model_name: str | None) -> str:
    return (model_name or DEFAULT_MODEL_NAME).strip() or DEFAULT_MODEL_NAME


def _doc_predict_id(item_id: str, origin: date, model_name: str | None = None) -> str:
    return f"{item_id}__{origin.isoformat()}"


def _predict_collection_name(model_name: str | None) -> str:
    model_name = _model_name(model_name)
    if model_name == "point":
        return "predict_timesfm"
    if model_name == "probabilistic":
        return "predict_chronos2"
    return f"predict_{model_name}"


def _model_display_name(model_name: str | None) -> str:
    model_name = _model_name(model_name)
    if model_name == "point":
        return "점예측(3일)"
    if model_name == "probabilistic":
        return "구간 예측(10일)"
    return model_name


def save_actual(
    item_id: str,
    d: date,
    price_krw: float,
    source: str = "reconcile",
) -> None:
    _actual_firestore_cache.pop(item_id, None)
    AgriActual.objects.update_or_create(
        item_id=item_id,
        date=d,
        defaults={"price_krw": price_krw, "source": source},
    )
    db = _firestore()
    if db is None:
        return
    doc_ref = db.collection("actual").document(_doc_actual_id(item_id, d))
    doc_ref.set(
        {
            "item_id": item_id,
            "date": d.isoformat(),
            "price_krw": float(price_krw),
            "source": source,
            "updated_at": datetime.now(timezone.utc),
        }
    )


def save_predict_batch(
    item_id: str,
    origin_date: date,
    horizon: int,
    points: list[dict[str, Any]],
    model_name: str = DEFAULT_MODEL_NAME,
) -> None:
    model_name = _model_name(model_name)
    AgriPredictBatch.objects.create(
        item_id=item_id,
        model_name=model_name,
        origin_date=origin_date,
        horizon=horizon,
        points=points,
    )
    db = _firestore()
    if db is None:
        return
    collection_name = _predict_collection_name(model_name)
    doc_ref = db.collection(collection_name).document(
        _doc_predict_id(item_id, origin_date, model_name)
    )
    doc_ref.set(
        {
            "item_id": item_id,
            "model_name": model_name,
            "model": _model_display_name(model_name),
            "collection": collection_name,
            "origin_date": origin_date.isoformat(),
            "horizon": horizon,
            "points": points,
            "created_at": datetime.now(timezone.utc),
        }
    )


def update_predict_points(
    item_id: str,
    origin_date: date,
    points: list[dict],
    model_name: str = DEFAULT_MODEL_NAME,
) -> None:
    model_name = _model_name(model_name)
    batch = (
        AgriPredictBatch.objects.filter(
            item_id=item_id,
            model_name=model_name,
            origin_date=origin_date,
        )
        .order_by("-created_at")
        .first()
    )
    if batch:
        batch.points = points
        batch.save(update_fields=["points"])
    db = _firestore()
    if db is None:
        return
    doc_ref = db.collection(_predict_collection_name(model_name)).document(
        _doc_predict_id(item_id, origin_date, model_name)
    )
    snap = doc_ref.get()
    if snap.exists:
        doc_ref.update({"points": points})


def get_django_actual_map(item_id: str) -> dict[str, float]:
    out = {}
    for row in AgriActual.objects.filter(item_id=item_id):
        out[row.date.isoformat()] = row.price_krw
    return out


def get_latest_predict_batch(
    item_id: str,
    model_name: str = DEFAULT_MODEL_NAME,
) -> Optional[dict]:
    """읽기는 Django를 기준으로 함(저장 시 Firestore에도 동기화)."""
    model_name = _model_name(model_name)
    b = (
        AgriPredictBatch.objects.filter(item_id=item_id, model_name=model_name)
        .order_by("-created_at")
        .first()
    )
    if b:
        return {
            "model_name": b.model_name,
            "origin_date": b.origin_date,
            "horizon": b.horizon,
            "points": list(b.points or []),
            "created_at": b.created_at,
        }
    return None


def load_actuals_from_firestore(item_id: str) -> dict[str, float]:
    cached = _actual_firestore_cache.get(item_id)
    now = monotonic()
    if cached and now - cached[0] < _ACTUAL_FIRESTORE_TTL_SECONDS:
        return dict(cached[1])
    db = _firestore()
    if db is None:
        return {}
    out = {}
    for snap in db.collection("actual").where("item_id", "==", item_id).stream():
        data = snap.to_dict()
        out[data["date"]] = float(data["price_krw"])
    _actual_firestore_cache[item_id] = (now, dict(out))
    return out
