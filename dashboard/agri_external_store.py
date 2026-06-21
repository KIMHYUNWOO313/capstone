# -*- coding: utf-8 -*-
"""외부 API 스냅샷 저장 — Django ApiSnapshot + Firestore `API` 컬렉션."""
from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any, Iterable

from .agri_store import _firestore  # 동일 firebase_admin app 재사용
from .models import ApiSnapshot


def _doc_id(source: str, d: date, item_key: str) -> str:
    safe = item_key.replace("/", "_").replace(":", "_")
    return f"{source}__{d.isoformat()}__{safe}"


def save_api_snapshot(
    source: str,
    snapshot_date: date,
    item_key: str,
    payload: dict[str, Any],
) -> None:
    ApiSnapshot.objects.update_or_create(
        source=source,
        snapshot_date=snapshot_date,
        item_key=item_key,
        defaults={"payload": payload},
    )
    db = _firestore()
    if db is None:
        return
    doc_ref = db.collection("API").document(_doc_id(source, snapshot_date, item_key))
    doc_ref.set(
        {
            "source": source,
            "snapshot_date": snapshot_date.isoformat(),
            "item_key": item_key,
            "payload": payload,
            "fetched_at": datetime.now(timezone.utc),
        }
    )


def save_many(records: Iterable[tuple[str, list[dict[str, Any]]]]) -> dict[str, int]:
    """records: (source, list of {snapshot_date,item_key,payload})."""
    counts: dict[str, int] = {}
    for source, items in records:
        n = 0
        for it in items or []:
            d = it.get("snapshot_date")
            if isinstance(d, str):
                try:
                    d = date.fromisoformat(d)
                except ValueError:
                    continue
            if not isinstance(d, date):
                continue
            save_api_snapshot(
                source=source,
                snapshot_date=d,
                item_key=str(it.get("item_key") or "default"),
                payload=dict(it.get("payload") or {}),
            )
            n += 1
        counts[source] = n
    return counts
