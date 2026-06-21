# -*- coding: utf-8 -*-
"""시작일을 N일 전으로 세팅하고 '확인'을 누른 뒤 표에 며칠치가 보이는지 점검."""
from __future__ import annotations

import os
import sys
import time
from collections import Counter
from datetime import date, timedelta
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


def _click_variable_input(page, panel_label_contains: str, value_text: str) -> bool:
    """'시작일'/'종료일' 인근 Variable input 위젯에서 value_text 옵션을 클릭한다.
    각 위젯은 'qv-object-qlik-variable-input' 클래스를 가지며, 클릭하면 옵션 리스트가 펼쳐진다.
    """

    js = """
    ([panelText, value]) => {
      const groups = document.querySelectorAll('article.qv-object-grouped_container, article.qv-object-qlik-variable-input');
      // 1) 라벨이 panelText (예: '시작일') 인 컨테이너의 후속 variable-input 들을 찾는다
      const inputs = document.querySelectorAll('article.qv-object-qlik-variable-input');
      const candidates = [];
      inputs.forEach(inp => {
        // input 자체 + 인접 형제 텍스트로 라벨 추정
        const txt = (inp.innerText || '');
        candidates.push({ inp, txt });
      });
      // 우선 panelText 기준으로 같은 grouped_container 안에 있는 inputs를 추려낸다
      const matches = [];
      const conts = document.querySelectorAll('article.qv-object-grouped_container');
      conts.forEach(c => {
        const t = c.innerText || '';
        if (!t.includes(panelText)) return;
        c.querySelectorAll('article.qv-object-qlik-variable-input').forEach(inp => matches.push(inp));
      });
      const list = matches.length ? matches : inputs;

      for (const inp of list) {
        const items = inp.querySelectorAll('div, li, button');
        for (const it of items) {
          const t = (it.innerText || '').trim();
          if (t === value) {
            it.scrollIntoView({block:'center'});
            const r = it.getBoundingClientRect();
            if (r.width>0 && r.height>0) { it.click(); return true; }
          }
        }
      }
      return false;
    }
    """
    try:
        return bool(page.evaluate(js, [panel_label_contains, value_text]))
    except Exception:
        return False


def _click_confirm(page) -> bool:
    js = """
    () => {
      const arts = document.querySelectorAll('article.qv-object-action-button');
      for (const a of arts) {
        const t = (a.innerText || '').trim();
        if (t.includes('확인')) {
          const btn = a.querySelector('button, [role=button], div');
          (btn || a).click();
          return true;
        }
      }
      return false;
    }
    """
    try:
        return bool(page.evaluate(js))
    except Exception:
        return False


def main() -> int:
    item = sys.argv[1] if len(sys.argv) > 1 else "양파"
    unit = sys.argv[2] if len(sys.argv) > 2 else "1키로"
    grade = sys.argv[3] if len(sys.argv) > 3 else "상"
    days = int(sys.argv[4]) if len(sys.argv) > 4 else 7

    end = date.today()
    start = end - timedelta(days=days)

    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        b = p.chromium.launch(headless=True)
        ctx = b.new_context(viewport={"width": 1600, "height": 1100}, locale="ko-KR")
        page = ctx.new_page()
        page.goto(_build_item_url(item), wait_until="domcontentloaded", timeout=60000)
        _wait_for_table(page, timeout_sec=40)
        time.sleep(2)

        s_year = f"{start.year}년"
        s_month = f"{start.month:02d}월"
        s_day = f"{start.day:02d}일"
        e_year = f"{end.year}년"
        e_month = f"{end.month:02d}월"
        e_day = f"{end.day:02d}일"
        print("setting start:", s_year, s_month, s_day, "end:", e_year, e_month, e_day)

        ok_y = _click_variable_input(page, "시작일", s_year); time.sleep(0.4)
        ok_m = _click_variable_input(page, "시작일", s_month); time.sleep(0.4)
        ok_d = _click_variable_input(page, "시작일", s_day); time.sleep(0.4)
        ok_ey = _click_variable_input(page, "종료일", e_year); time.sleep(0.4)
        ok_em = _click_variable_input(page, "종료일", e_month); time.sleep(0.4)
        ok_ed = _click_variable_input(page, "종료일", e_day); time.sleep(0.4)
        print("clicks ok:", ok_y, ok_m, ok_d, ok_ey, ok_em, ok_ed)

        ok_confirm = _click_confirm(page)
        print("confirm clicked:", ok_confirm)
        time.sleep(3)
        _scroll_table_to_load_more(page, rounds=40)

        rows = _read_all_rows(page)
        print(f"total rows in DOM: {len(rows)}")
        objs = [r for r in (_row_to_obj(c) for c in rows) if r is not None]
        cnt_date = Counter(o["date"].isoformat() for o in objs)
        for d, n in sorted(cnt_date.items(), reverse=True)[:14]:
            print(f"  {d}: {n}")

        match = [o for o in objs if o["unit"] == unit and o["grade"] == grade]
        match.sort(key=lambda x: x["date"], reverse=True)
        print(f"\n[match unit={unit} grade={grade}] {len(match)} rows")
        for o in match[:14]:
            print(f"  {o['date']} {o['avg_price']} prev={o['prev_day']}")
        b.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
