# -*- coding: utf-8 -*-
"""
nongnet_by_series/*.csv 에서 평균가격 연속 결측만 채운다.

조건: 결측 구간 **바로 위·바로 아래**에 유효한 평균가격이 있을 때만.
채움값: int(round((위값 + 아래값) / 2)) — 구간 전체에 동일 적용.

전일·전년 등 다른 열은 변경하지 않음.

입력 기본: AGRICULTURE/data/processed/nongnet_by_series/
출력 기본: AGRICULTURE/data/processed/nongnet_by_series_filled/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

AG_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN = AG_ROOT / "data" / "processed" / "nongnet_by_series"
DEFAULT_OUT = AG_ROOT / "data" / "processed" / "nongnet_by_series_filled"
MANIFEST = AG_ROOT / "data" / "processed" / "fill_nongnet_price_manifest.json"


def _pick_date_col(df: pd.DataFrame) -> str:
    for c in df.columns:
        if str(c).strip().upper() == "DATE" or "날짜" in str(c):
            return c
    raise ValueError("DATE 열 없음")


def _price_series(df: pd.DataFrame) -> tuple[pd.Series, str]:
    for name in ("평균가격", "평균가격(원)", "평균"):
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce"), name
    raise ValueError("평균가격 열 없음")


def fill_interior_mean_gaps(values: np.ndarray) -> tuple[np.ndarray, int]:
    """
    values: float array, NaN = missing
    Returns (filled_array, n_filled_cells)
    """
    n = len(values)
    out = values.astype(float, copy=True)
    filled = 0
    i = 0
    while i < n:
        if not np.isnan(out[i]):
            i += 1
            continue
        j = i
        while j < n and np.isnan(out[j]):
            j += 1
        # [i, j) is NaN run
        if i > 0 and j < n and not np.isnan(out[i - 1]) and not np.isnan(out[j]):
            before = float(out[i - 1])
            after = float(out[j])
            fill_val = int(round((before + after) / 2))
            out[i:j] = fill_val
            filled += j - i
        i = j
    return out, filled


def process_file(path: Path, out_dir: Path) -> dict:
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=object)
    date_col = _pick_date_col(df)

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.sort_values(date_col, kind="mergesort").reset_index(drop=True)

    _, price_col = _price_series(df)
    prices_num = pd.to_numeric(df[price_col], errors="coerce")
    arr = prices_num.to_numpy(dtype=float)
    filled_arr, n_fill = fill_interior_mean_gaps(arr)

    col_out: list = []
    for v in filled_arr:
        if np.isnan(v):
            col_out.append(pd.NA)
        else:
            col_out.append(int(round(float(v))))
    df[price_col] = pd.Series(col_out, dtype="Int64")

    out_path = out_dir / path.name
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return {"file": path.name, "rows": len(df), "cells_filled": n_fill, "output": str(out_path.relative_to(AG_ROOT))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=Path, default=DEFAULT_IN)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    indir, outdir = args.in_dir, args.out_dir
    if not indir.is_dir():
        print(f"입력 폴더 없음: {indir}")
        return 1
    outdir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    for path in sorted(indir.glob("*.csv")):
        print(f"Processing {path.name} ...", flush=True)
        try:
            results.append(process_file(path, outdir))
        except ValueError as e:
            print(f"  skip: {e}", flush=True)

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(
        json.dumps(
            {"in_dir": str(indir), "out_dir": str(outdir), "files": results},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {len(results)} files to {outdir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
