# -*- coding: utf-8 -*-
"""
data/농넷 데이터/*.xlsx 를 결측·보간·재계산 없이 그대로 읽어,
품종 × 거래단위 × 등급 조합마다 CSV로 분리한다. DATE 오름차순만 적용.

예: 당근.xlsx → 당근_당근(전체)_10kg_상품.csv, … (조합 수만큼)

출력: AGRICULTURE/data/processed/nongnet_by_series/
      split_nongnet_raw_manifest.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

AG_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = AG_ROOT.parent
DEFAULT_SRC = REPO_ROOT / "data" / "농넷 데이터"
OUT_DIR = AG_ROOT / "data" / "processed" / "nongnet_by_series"
MANIFEST = AG_ROOT / "data" / "processed" / "split_nongnet_raw_manifest.json"
_WIN_INVALID = re.compile(r'[<>:"/\\|?*]')


def _safe_part(s: object) -> str:
    t = str(s) if s is not None and not (isinstance(s, float) and pd.isna(s)) else "NA"
    t = t.strip()
    t = _WIN_INVALID.sub("_", t)
    t = re.sub(r"\s+", "_", t)
    return t or "NA"


def _pick_date_col(df: pd.DataFrame) -> str:
    for c in df.columns:
        if str(c).strip().upper() == "DATE" or "날짜" in str(c):
            return c
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            return c
    raise ValueError("DATE/날짜 열이 없습니다.")


def split_workbook(path: Path, out_dir: Path) -> list[dict]:
    df = pd.read_excel(path, engine="openpyxl", dtype=object)
    date_col = _pick_date_col(df)

    for col in ("품종", "거래단위", "등급"):
        if col not in df.columns:
            raise ValueError(f"{path.name}: '{col}' 열이 필요합니다.")

    df = df.copy()
    sort_key = "__ts__"
    df[sort_key] = pd.to_datetime(df[date_col], errors="coerce")

    stem = path.stem
    meta: list[dict] = []

    for key, g in df.groupby(["품종", "거래단위", "등급"], dropna=False, sort=False):
        pumjong, unit, grade = key
        sub = g.sort_values(sort_key, na_position="last", kind="mergesort")
        sub = sub.drop(columns=[sort_key])
        fname = f"{stem}_{_safe_part(pumjong)}_{_safe_part(unit)}_{_safe_part(grade)}.csv"
        fp = out_dir / fname
        sub.to_csv(fp, index=False, encoding="utf-8-sig")
        meta.append(
            {
                "source": path.name,
                "output": str(fp.relative_to(AG_ROOT)),
                "품종": pumjong if pd.notna(pumjong) else None,
                "거래단위": unit if pd.notna(unit) else None,
                "등급": grade if pd.notna(grade) else None,
                "rows": len(sub),
            }
        )
    return meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    src: Path = args.src_dir
    out_dir: Path = args.out_dir
    if not src.is_dir():
        print(f"소스 폴더 없음: {src}")
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    all_meta: list[dict] = []

    for path in sorted(src.glob("*.xlsx")):
        print(f"Splitting {path.name} ...", flush=True)
        try:
            all_meta.extend(split_workbook(path, out_dir))
        except ValueError as e:
            print(f"  skip: {e}", flush=True)

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(
        json.dumps(
            {"src_dir": str(src), "out_dir": str(out_dir), "series": all_meta},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {len(all_meta)} series under {out_dir}", flush=True)
    print(f"Manifest: {MANIFEST}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
