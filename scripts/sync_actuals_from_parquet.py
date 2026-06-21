from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import dotenv_values
from firebase_admin import _apps, credentials, firestore, initialize_app


def _load_firestore():
    env = dotenv_values(".env")
    cred_path = env.get("FIREBASE_CREDENTIALS_PATH", "")
    if not cred_path:
        raise RuntimeError("FIREBASE_CREDENTIALS_PATH is not set in .env")
    cred_file = Path(cred_path)
    if not cred_file.is_file():
        raise FileNotFoundError(f"Firebase credential file not found: {cred_file}")

    if not _apps:
        initialize_app(credentials.Certificate(str(cred_file)))
    return firestore.client()


def _load_rows(parquet_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    if not isinstance(df.index, pd.MultiIndex) or list(df.index.names) != ["item_id", "timestamp"]:
        raise ValueError("Parquet index must be MultiIndex ['item_id', 'timestamp']")
    if "target" not in df.columns:
        raise ValueError("Parquet must contain 'target' column")

    rows = df.reset_index()[["item_id", "timestamp", "target"]].copy()
    rows = rows.rename(columns={"timestamp": "date", "target": "price_krw"})
    rows["date"] = rows["date"].astype(str)
    rows["price_krw"] = pd.to_numeric(rows["price_krw"], errors="coerce")
    rows = rows.dropna(subset=["price_krw"])
    rows["price_krw"] = rows["price_krw"].astype(float)
    return rows


def _alias_item_id(item_id: str) -> str | None:
    grade_map = {"high": "high", "low": "low", "mid": "medium", "premium": "top"}

    def map_grade(suffix: str) -> str | None:
        return grade_map.get(suffix)

    if item_id.startswith("apple_fuji_box10kg_"):
        g = map_grade(item_id.rsplit("_", 1)[-1])
        return f"apple_apple_10kg_{g}" if g else None
    if item_id.startswith("cabbage_net8kg_"):
        g = map_grade(item_id.rsplit("_", 1)[-1])
        return f"cabbage_8_net_{g}" if g else None
    if item_id.startswith("napa_cabbage_net10kg_"):
        g = map_grade(item_id.rsplit("_", 1)[-1])
        return f"cabbage_10_net_{g}" if g else None
    if item_id.startswith("onion_kg1_"):
        g = map_grade(item_id.rsplit("_", 1)[-1])
        return f"onion_1_{g}" if g else None
    if item_id.startswith("carrot_box20kg_"):
        g = map_grade(item_id.rsplit("_", 1)[-1])
        return f"carrot_carrot_20kg_{g}" if g else None
    if item_id.startswith("cucumber_bdadagi_ea100_"):
        g = map_grade(item_id.rsplit("_", 1)[-1])
        return f"cucumber_100ea_{g}" if g else None
    if item_id.startswith("perilla_leaf_bunch100_"):
        g = map_grade(item_id.rsplit("_", 1)[-1])
        return f"perilla_leaf_100_{g}" if g else None
    if item_id.startswith("potato_sumi_box20kg_"):
        g = map_grade(item_id.rsplit("_", 1)[-1])
        return f"potato_potato_20kg_{g}" if g else None
    if item_id.startswith("spinach_box4kg_"):
        g = map_grade(item_id.rsplit("_", 1)[-1])
        return f"spinach_spinach_4kg_{g}" if g else None
    if item_id.startswith("sweetpotato_box10kg_"):
        g = map_grade(item_id.rsplit("_", 1)[-1])
        return f"sweet_potato_sweet_potato_1kg_{g}" if g else None
    return None


def _build_alias_rows(rows: pd.DataFrame) -> pd.DataFrame:
    aliased = rows.copy()
    aliased["item_id"] = aliased["item_id"].map(_alias_item_id)
    aliased = aliased.dropna(subset=["item_id"])
    return aliased


def _upsert_actuals(rows: pd.DataFrame, collection: str, source: str) -> tuple[int, str]:
    db = _load_firestore()
    col = db.collection(collection)

    total = 0
    max_date = ""
    now = datetime.now(timezone.utc)
    batch = db.batch()
    batch_size = 0

    for item_id, d, price in rows.itertuples(index=False, name=None):
        doc_id = f"{item_id}__{d}"
        ref = col.document(doc_id)
        batch.set(
            ref,
            {
                "item_id": item_id,
                "date": d,
                "price_krw": float(price),
                "source": source,
                "updated_at": now,
            },
            merge=True,
        )
        batch_size += 1
        total += 1
        if d > max_date:
            max_date = d

        if batch_size >= 450:
            batch.commit()
            batch = db.batch()
            batch_size = 0

    if batch_size:
        batch.commit()

    return total, max_date


def _delete_collection(collection: str, batch_limit: int = 450) -> int:
    db = _load_firestore()
    col = db.collection(collection)
    deleted = 0
    while True:
        docs = list(col.limit(batch_limit).stream())
        if not docs:
            break
        batch = db.batch()
        for doc in docs:
            batch.delete(doc.reference)
            deleted += 1
        batch.commit()
    return deleted


def main():
    parser = argparse.ArgumentParser(description="Sync parquet actual prices to Firebase Firestore")
    parser.add_argument(
        "--parquet",
        default="full_baseline_extended_20260516.parquet",
        help="Path to parquet file",
    )
    parser.add_argument("--collection", default="actual", help="Firestore collection name")
    parser.add_argument("--source", default="full_baseline_extended_20260516", help="source field")
    parser.add_argument(
        "--delete-existing",
        action="store_true",
        help="Delete every document in the target collection before upload",
    )
    parser.add_argument(
        "--delete-only",
        action="store_true",
        help="Only delete documents in the target collection and exit",
    )
    parser.add_argument(
        "--write-django-alias",
        action="store_true",
        help="Also write alias item_ids used by Django UI",
    )
    args = parser.parse_args()

    parquet_path = Path(args.parquet)
    if not parquet_path.is_file():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    if args.delete_existing:
        deleted = _delete_collection(args.collection)
        print(f"deleted_existing_rows={deleted}")
        if args.delete_only:
            return

    rows = _load_rows(parquet_path)
    total, max_date = _upsert_actuals(rows, args.collection, args.source)
    print(f"synced_rows={total}")
    print(f"max_date={max_date}")

    if args.write_django_alias:
        alias_rows = _build_alias_rows(rows)
        alias_total, alias_max_date = _upsert_actuals(
            alias_rows, args.collection, f"{args.source}_alias"
        )
        print(f"alias_synced_rows={alias_total}")
        print(f"alias_max_date={alias_max_date}")


if __name__ == "__main__":
    main()
