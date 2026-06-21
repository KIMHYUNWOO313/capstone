# -*- coding: utf-8 -*-
"""대상일자_선택 ID 하나를 select 파라미터에 박고 어떤 날짜가 표에 뜨는지 본다.

사용:
    python scripts/nongnet_dateid_probe.py 46137
    python scripts/nongnet_dateid_probe.py 46140
"""
from __future__ import annotations

import os
import sys
import time
from collections import Counter
from pathlib import Path
from urllib.parse import quote

import django

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from dashboard.agri_nongnet import (  # noqa: E402
    _read_all_rows,
    _row_to_obj,
    _scroll_table_to_load_more,
    _wait_for_table,
)

BASE = (
    "https://www.nongnet.or.kr/qlik/sso/single/?"
    "appid=21f27d83-cf68-4f03-afe4-aed6907fbe78&"
    "sheet=c262cbfc-2e3c-414b-91a0-a0c9351dfa35&"
    "theme=theme_at_24&opt=ctxmenu,currsel"
)


def main() -> int:
    item = "양파"
    ids = sys.argv[1:] or ["46137"]
    sel_item = f"&select=$::%ED%92%88%EB%AA%A9%EB%AA%85_%EC%84%A0%ED%83%9D,{quote(item)}"
    sel_dates = "&select=$::%EB%8C%80%EC%83%81%EC%9D%BC%EC%9E%90_%EC%84%A0%ED%83%9D," + ",".join(ids)
    url = BASE + sel_item + sel_dates
    print("URL:", url)

    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        b = p.chromium.launch(headless=True)
        ctx = b.new_context(viewport={"width": 1600, "height": 1100}, locale="ko-KR")
        page = ctx.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=60000)
        _wait_for_table(page, timeout_sec=40)
        time.sleep(3)

        all_rows = []
        seen_keys = set()
        # 스크롤을 여러 차례 시도하면서 매번 행을 누적한다.
        for round_idx in range(60):
            _scroll_table_to_load_more(page, rounds=2)
            rows = _read_all_rows(page)
            for r in rows:
                key = tuple(r)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                all_rows.append(r)
            if round_idx % 10 == 0:
                print(f"  round={round_idx} unique rows={len(all_rows)}")

        objs = [r for r in (_row_to_obj(c) for c in all_rows) if r is not None]
        cnt = Counter(o["date"].isoformat() for o in objs)
        print("dates seen:", sorted(cnt.items(), reverse=True))
        b.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
