# -*- coding: utf-8 -*-
"""농넷(nongnet.or.kr) 가락시장 경락가격 — Playwright 기반 일별 크롤러.

대상 페이지(Qlik 임베디드):
    https://www.nongnet.or.kr/qlik/sso/single/?appid=...&sheet=...

화면 구성(분석 결과):
  - 품목 :        SimpleFieldSelect (단일 선택). URL `select=$::품목명_선택,<품목명>`
  - 단위명 :      filterpane(필터 창)            클릭으로 1개 선택
  - 등급명 :      filterpane(필터 창)            클릭으로 1개 선택
  - 결과 표 :     Qlik 테이블 — 컬럼은
                  [DATE, 품목명, 단위, 등급명, 평균가격, 전일, 전년]

매일 00:00 호출 시 가장 최신 DATE 한 줄(전일자 가락경매)만 발췌해 저장한다.

저장 형식 (ApiSnapshot/`API` 컬렉션):
  source = 'nongnet'
  snapshot_date = DATE 컬럼(예: 2026-04-25)
  item_key = f"{품목}__{단위}__{등급}"  (예: "감자 수미__20키로상자__상")
  payload = {
      "item": "감자 수미",
      "unit": "20키로상자",
      "grade": "상",
      "avg_price": 28500,
      "prev_day": 28000,
      "prev_year": 0,
      "raw_row": [...]
  }
"""
from __future__ import annotations

import logging
import re
import time
from datetime import date, datetime, timedelta
from typing import Any
from urllib.parse import quote

logger = logging.getLogger(__name__)


# 사용자 요구 — (품목명, 단위명, 등급명='상')
TARGETS: tuple[tuple[str, str, str], ...] = (
    ("감자 수미", "20키로상자", "상"),
    ("고구마", "10키로상자", "상"),
    ("당근", "20키로상자", "상"),
    ("대파(일반)", "1키로단", "상"),
    ("배추", "10키로망대", "상"),
    ("백다다기오이", "100개", "상"),
    ("사과 부사", "10키로상자", "상"),
    ("시금치", "4키로상자", "상"),
    ("양파", "1키로", "상"),
)


_BASE_URL = (
    "https://www.nongnet.or.kr/qlik/sso/single/?"
    "appid=21f27d83-cf68-4f03-afe4-aed6907fbe78&"
    "sheet=c262cbfc-2e3c-414b-91a0-a0c9351dfa35&"
    "theme=theme_at_24&opt=ctxmenu,currsel"
)

# 농넷 Qlik 데이터셋의 '대상일자_선택' 필드는 1996-01-01부터 1씩 증가하는
# 일련번호로 인덱싱된다. 실측으로 다음 1쌍을 얻었다:
#   date_id 46137  ↔  2026-04-25
# 이로부터 임의 일자 d에 대해
#   date_id(d) = 46137 + (d - 2026-04-25).days
# 가 성립한다. 사용자가 보낸 URL의 46124..46137 (14일) 도 위 식과 일치.
_DATE_ID_ANCHOR_DATE = date(2026, 4, 25)
_DATE_ID_ANCHOR_VALUE = 46137


def _date_id(d: date) -> int:
    return _DATE_ID_ANCHOR_VALUE + (d - _DATE_ID_ANCHOR_DATE).days


def _build_item_url(item_name: str, target_dates: list[date] | None = None) -> str:
    """품목과(선택) 일자 ID 들을 select 파라미터로 미리 박아 URL을 만든다."""

    url = f"{_BASE_URL}&select=$::%ED%92%88%EB%AA%A9%EB%AA%85_%EC%84%A0%ED%83%9D,{quote(item_name)}"
    if target_dates:
        ids = ",".join(str(_date_id(d)) for d in target_dates)
        url += f"&select=$::%EB%8C%80%EC%83%81%EC%9D%BC%EC%9E%90_%EC%84%A0%ED%83%9D,{ids}"
    return url


_NUM_RE = re.compile(r"^-?[\d,]+$")


def _parse_number(s: str) -> float | None:
    s = (s or "").strip().replace(",", "")
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _parse_date(s: str) -> date | None:
    s = (s or "").strip()
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            pass
    return None


def _wait_for_table(page, timeout_sec: int = 30) -> bool:
    """경락가격 테이블이 렌더링될 때까지 대기."""

    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            count = page.evaluate(
                "() => document.querySelectorAll('td.qv-st-data-cell').length"
            )
        except Exception:
            count = 0
        if count and count > 5:
            return True
        time.sleep(0.5)
    return False


def _click_in_filterpane(page, panel_label: str, value_text: str) -> bool:
    """`필터 창` 패널 중 라벨이 panel_label 인 패널에서 value_text 셀을 클릭."""

    # 모든 filterpane 패널을 훑어 라벨이 일치하는 것을 찾고, 그 안의 셀 중 하나를 클릭한다.
    js = """
    ([panelLabel, valueText]) => {
      const panels = document.querySelectorAll('article.qv-object-filterpane');
      for (const p of panels) {
        const txt = (p.innerText || '').trim();
        if (!txt.includes(panelLabel)) continue;
        // 셀 후보: list/checkbox/listbox row, 일반 div with text
        const cells = p.querySelectorAll('div');
        for (const c of cells) {
          const t = (c.innerText || '').trim();
          if (t === valueText) {
            c.scrollIntoView({block: 'center'});
            const rect = c.getBoundingClientRect();
            if (rect.width > 0 && rect.height > 0) {
              c.click();
              return true;
            }
          }
        }
      }
      return false;
    }
    """
    try:
        return bool(page.evaluate(js, [panel_label, value_text]))
    except Exception as exc:
        logger.warning("filterpane click failed (%s=%s): %s", panel_label, value_text, exc)
        return False


def _read_all_rows(page) -> list[list[str]]:
    """가락시장 경락가격 테이블의 모든 데이터 행을 읽는다.

    컬럼 순서: DATE, 품목명, 단위, 등급명, 평균가격, 전일, 전년
    """

    js = """
    () => {
      const out = [];
      const tables = document.querySelectorAll('table');
      for (const tbl of tables) {
        const text = (tbl.innerText || '');
        if (!text.includes('DATE') || !text.includes('평균가격')) continue;
        const rows = tbl.querySelectorAll('tr');
        for (const tr of rows) {
          const tds = tr.querySelectorAll('td.qv-st-data-cell');
          if (tds.length < 7) continue;
          const cells = Array.from(tds).slice(0, 7).map(td => (td.innerText || '').trim());
          if (/\\d{4}-\\d{2}-\\d{2}/.test(cells[0])) {
            out.push(cells);
          }
        }
      }
      return out;
    }
    """
    try:
        rows = page.evaluate(js)
    except Exception as exc:
        logger.warning("read rows failed: %s", exc)
        return []
    if not isinstance(rows, list):
        return []
    return [r for r in rows if isinstance(r, list) and len(r) >= 7]


def _scroll_table_to_load_more(page, rounds: int = 8) -> None:
    """Qlik 테이블은 가상 스크롤이라 화면에 보이는 행만 DOM에 렌더링된다.

    실 사용에서 가장 잘 먹는 전략:
      1) 테이블 컨테이너의 높이를 매우 크게 늘려 가상 스크롤이 모든 행을 그리게 한다
      2) 그 후 테이블 내부 스크롤 컨테이너를 위·아래로 흔들어 누락 방지
    """

    js_resize = """
    () => {
      const arts = document.querySelectorAll('article.qv-object-table');
      let n = 0;
      for (const a of arts) {
        const txt = (a.innerText || '');
        if (!txt.includes('DATE') || !txt.includes('평균가격')) continue;
        // 부모 컨테이너 + 자기 자신 + 안쪽 .qv-st-data-cell 컨테이너 모두 키운다
        let p = a;
        for (let i=0; i<6 && p; i++) {
          if (p.style) {
            p.style.height = '6000px';
            p.style.maxHeight = '6000px';
            p.style.minHeight = '6000px';
          }
          p = p.parentElement;
        }
        a.style.height = '6000px';
        a.style.maxHeight = '6000px';
        a.querySelectorAll('div').forEach(d => {
          if (d.style && (d.classList.contains('qv-object-content') || d.classList.contains('qv-st') )) {
            d.style.height = '6000px';
            d.style.maxHeight = '6000px';
          }
        });
        n++;
      }
      return n;
    }
    """
    js_scroll = """
    ([dy]) => {
      const tables = document.querySelectorAll('article.qv-object-table');
      for (const t of tables) {
        const txt = (t.innerText || '');
        if (!txt.includes('DATE') || !txt.includes('평균가격')) continue;
        const cands = t.querySelectorAll('.qv-st-scroll, .qv-scrollbar-vertical, .qv-scrollarea-vertical, .qv-st-data-scroll, .qv-object-content, .qv-st');
        for (const sc of cands) {
          if (sc.scrollHeight > sc.clientHeight + 5) {
            sc.scrollTop = (sc.scrollTop || 0) + dy;
          }
        }
        t.scrollTop = (t.scrollTop || 0) + dy;
      }
      window.scrollBy(0, dy);
      return true;
    }
    """
    try:
        n = page.evaluate(js_resize)
        logger.info("table containers resized: %s", n)
    except Exception:
        pass
    time.sleep(0.5)

    for i in range(rounds):
        try:
            page.evaluate(js_scroll, [600])
        except Exception:
            pass
        time.sleep(0.3)

    # 위로 다시 돌아오기
    try:
        page.evaluate(js_scroll, [-100000])
    except Exception:
        pass
    time.sleep(0.3)


def _row_to_obj(cells: list[str]) -> dict[str, Any] | None:
    if len(cells) < 7:
        return None
    d = _parse_date(cells[0])
    if d is None:
        return None
    return {
        "date": d,
        "item": cells[1].strip(),
        "unit": cells[2].strip(),
        "grade": cells[3].strip(),
        "avg_price": _parse_number(cells[4]),
        "prev_day": _parse_number(cells[5]),
        "prev_year": _parse_number(cells[6]),
        "raw_row": cells,
    }


def _pick_target_row(
    rows: list[list[str]], unit: str, grade: str
) -> dict[str, Any] | None:
    """단위·등급이 일치하는 가장 최신 행을 고른다."""

    best: dict[str, Any] | None = None
    for cells in rows:
        obj = _row_to_obj(cells)
        if obj is None:
            continue
        if obj["unit"] != unit:
            continue
        if obj["grade"] != grade:
            continue
        if best is None or obj["date"] > best["date"]:
            best = obj
    return best


def _scrape_one_day(
    page,
    item: str,
    unit: str,
    grade: str,
    target_date: date,
) -> dict[str, Any] | None:
    """단일 (품목, 단위, 등급, 일자) 조합 1행을 긁는다."""

    url = _build_item_url(item, target_dates=[target_date])
    logger.info("nongnet open item=%s date=%s", item, target_date.isoformat())
    page.goto(url, wait_until="domcontentloaded", timeout=60000)
    if not _wait_for_table(page, timeout_sec=40):
        logger.warning("nongnet table did not render for %s @ %s", item, target_date)
        return None
    time.sleep(1.5)

    _scroll_table_to_load_more(page, rounds=4)
    rows = _read_all_rows(page)

    # 1) 정확히 (unit, grade, target_date)
    for cells in rows:
        obj = _row_to_obj(cells)
        if obj is None:
            continue
        if obj["date"] != target_date:
            continue
        if obj["unit"] != unit or obj["grade"] != grade:
            continue
        return obj
    # 2) (unit, grade) 일치하는 가장 가까운 일자(보통 동일 날짜)
    candidates = [
        o for o in (_row_to_obj(c) for c in rows)
        if o is not None and o["unit"] == unit and o["grade"] == grade
    ]
    if candidates:
        candidates.sort(key=lambda x: abs((x["date"] - target_date).days))
        return candidates[0]
    return None


def _scrape_item_window(
    playwright,
    item: str,
    unit: str,
    grade: str,
    target_dates: list[date],
) -> list[dict[str, Any]]:
    """1개 (품목, 단위, 등급)에 대해 일자 윈도우 전체를 순회한다.

    하나의 브라우저 인스턴스를 재사용해 페이지만 새로 띄운다.
    """

    out: list[dict[str, Any]] = []
    browser = playwright.chromium.launch(headless=True)
    try:
        context = browser.new_context(
            viewport={"width": 1600, "height": 1100},
            locale="ko-KR",
        )
        page = context.new_page()
        for d in target_dates:
            try:
                row = _scrape_one_day(page, item, unit, grade, d)
            except Exception as exc:
                logger.warning(
                    "nongnet scrape error %s/%s/%s @%s: %s", item, unit, grade, d, exc
                )
                row = None
            if row is None:
                logger.info("nongnet no match %s/%s/%s @%s", item, unit, grade, d)
                continue
            out.append(row)
    finally:
        try:
            browser.close()
        except Exception:
            pass
    return out


def fetch_nongnet_for_dates(
    target_dates: list[date],
    targets: tuple[tuple[str, str, str], ...] | None = None,
) -> list[dict[str, Any]]:
    """주어진 일자 리스트와 (품목·단위·등급) 조합 전부를 수집한다.

    반환: [{"snapshot_date": date, "item_key": str, "payload": dict}, ...]
    Playwright 미설치 또는 실패 시 빈 리스트.
    """

    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:
        logger.warning("playwright not available; skip nongnet: %s", exc)
        return []

    targets = targets or TARGETS
    out: list[dict[str, Any]] = []
    if not target_dates:
        return out

    with sync_playwright() as p:
        for item, unit, grade in targets:
            rows = _scrape_item_window(p, item, unit, grade, list(target_dates))
            for row in rows:
                d = row["date"]
                out.append(
                    {
                        "snapshot_date": d,
                        "item_key": f"{item}__{unit}__{grade}",
                        "payload": {
                            "item": item,
                            "unit": unit,
                            "grade": grade,
                            "matched_item": row.get("item"),
                            "matched_unit": row.get("unit"),
                            "matched_grade": row.get("grade"),
                            "avg_price": row.get("avg_price"),
                            "prev_day": row.get("prev_day"),
                            "prev_year": row.get("prev_year"),
                            "raw_row": row.get("raw_row"),
                        },
                    }
                )
    return out


def fetch_nongnet_daily(
    targets: tuple[tuple[str, str, str], ...] | None = None,
    target_date: date | None = None,
) -> list[dict[str, Any]]:
    """매일 00:00 호출용 — 단일 일자(default = 어제)에 대해 모든 품목을 수집."""

    if target_date is None:
        target_date = date.today() - timedelta(days=1)
    return fetch_nongnet_for_dates([target_date], targets=targets)


def fetch_nongnet_window(
    days: int = 7,
    end_date: date | None = None,
    targets: tuple[tuple[str, str, str], ...] | None = None,
) -> list[dict[str, Any]]:
    """end_date(default=어제) 기준 days일치(end_date 포함)를 수집한다."""

    if end_date is None:
        end_date = date.today() - timedelta(days=1)
    dates = [end_date - timedelta(days=i) for i in range(days - 1, -1, -1)]
    return fetch_nongnet_for_dates(dates, targets=targets)
