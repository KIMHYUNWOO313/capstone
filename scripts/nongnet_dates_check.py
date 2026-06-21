# -*- coding: utf-8 -*-
"""한 품목에 대해 표에 보이는 (단위/등급/일자) 분포를 점검."""
from __future__ import annotations

import os
import sys
import time
from collections import Counter
from pathlib import Path

import django

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from dashboard.agri_nongnet import (  # noqa: E402
    _build_item_url,
    _read_all_rows,
    _row_to_obj,
    _scroll_table_to_load_more,
    _wait_for_table,
)


def main() -> int:
    item = sys.argv[1] if len(sys.argv) > 1 else "양파"
    unit = sys.argv[2] if len(sys.argv) > 2 else "1키로"
    grade = sys.argv[3] if len(sys.argv) > 3 else "상"
    rounds = int(sys.argv[4]) if len(sys.argv) > 4 else 30

    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        b = p.chromium.launch(headless=True)
        ctx = b.new_context(viewport={"width": 1600, "height": 1100}, locale="ko-KR")
        page = ctx.new_page()
        page.goto(_build_item_url(item), wait_until="domcontentloaded", timeout=60000)
        _wait_for_table(page, timeout_sec=40)
        time.sleep(2)
        _scroll_table_to_load_more(page, rounds=rounds)
        rows = _read_all_rows(page)
        print(f"total rows in DOM: {len(rows)}")

        objs = [r for r in (_row_to_obj(c) for c in rows) if r is not None]
        cnt_unit = Counter(o["unit"] for o in objs)
        cnt_grade = Counter(o["grade"] for o in objs)
        cnt_date = Counter(o["date"].isoformat() for o in objs)
        print(f"\n[unit] {len(cnt_unit)}: {cnt_unit.most_common()}")
        print(f"\n[grade] {len(cnt_grade)}: {cnt_grade.most_common()}")
        print(f"\n[date] top 20:")
        for d, n in sorted(cnt_date.items(), reverse=True)[:20]:
            print(f"  {d}: {n}")

        match = [o for o in objs if o["unit"] == unit and o["grade"] == grade]
        match.sort(key=lambda x: x["date"], reverse=True)
        print(f"\n[match unit={unit} grade={grade}] {len(match)} rows")
        for o in match[:14]:
            print(f"  {o['date']} {o['avg_price']} prev={o['prev_day']} prev_y={o['prev_year']}")

        b.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
