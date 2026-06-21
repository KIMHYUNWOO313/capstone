# -*- coding: utf-8 -*-
"""농넷(nongnet.or.kr) Qlik 임베디드 페이지 구조 진단 스크립트.

페이지를 헤드리스 Chromium으로 열고:
  1) 네트워크 요청을 캡처해 Qlik 데이터 호출 패턴 확인
  2) DOM에서 표(table)·셀 형태로 보이는 요소 후보 수집
  3) 페이지 스크린샷 + 전체 HTML 덤프
출력은 scripts/_nongnet_dump/ 폴더에 저장.

수동 실행:
    python scripts/nongnet_inspect.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

URL = (
    "https://www.nongnet.or.kr/qlik/sso/single/?"
    "appid=21f27d83-cf68-4f03-afe4-aed6907fbe78&"
    "sheet=c262cbfc-2e3c-414b-91a0-a0c9351dfa35&"
    "theme=theme_at_24&opt=ctxmenu,currsel"
)

OUT_DIR = Path(__file__).resolve().parent / "_nongnet_dump"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> int:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        print(f"[err] playwright import failed: {exc}", file=sys.stderr)
        return 2

    requests_log: list[dict] = []
    websocket_log: list[dict] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": 1600, "height": 1100},
            locale="ko-KR",
        )
        page = context.new_page()

        def on_request(req):
            try:
                requests_log.append(
                    {
                        "method": req.method,
                        "url": req.url,
                        "resource_type": req.resource_type,
                    }
                )
            except Exception:
                pass

        def on_websocket(ws):
            websocket_log.append({"url": ws.url, "events": []})

            def on_frame_received(payload):
                try:
                    websocket_log[-1]["events"].append(
                        {"dir": "in", "preview": str(payload)[:400]}
                    )
                except Exception:
                    pass

            def on_frame_sent(payload):
                try:
                    websocket_log[-1]["events"].append(
                        {"dir": "out", "preview": str(payload)[:400]}
                    )
                except Exception:
                    pass

            ws.on("framereceived", on_frame_received)
            ws.on("framesent", on_frame_sent)

        page.on("request", on_request)
        page.on("websocket", on_websocket)

        print(f"[info] navigating: {URL}")
        page.goto(URL, wait_until="domcontentloaded", timeout=60000)
        # Qlik이 비동기로 렌더링하므로 충분히 대기
        for _ in range(20):
            time.sleep(1)
            ready = page.evaluate(
                "() => document.querySelectorAll('.qv-object').length"
            )
            print(f"  qv-object count = {ready}")
            if ready and ready > 0:
                break
        time.sleep(3)

        # 표 후보 수집
        tables_info = page.evaluate(
            """
            () => {
              const out = [];
              const sel = '.qv-object, .qv-st-data-cell, [data-qid], table, .qv-object-content';
              document.querySelectorAll(sel).forEach((el, i) => {
                if (i > 200) return;
                const cls = el.className || '';
                const txt = (el.innerText || '').slice(0, 200);
                out.push({
                  tag: el.tagName,
                  cls: typeof cls === 'string' ? cls : '',
                  qid: el.getAttribute('data-qid') || '',
                  text_head: txt,
                });
              });
              return out;
            }
            """
        )

        # 스크린샷 / HTML 덤프
        page.screenshot(path=str(OUT_DIR / "page.png"), full_page=True)
        html = page.content()
        (OUT_DIR / "page.html").write_text(html, encoding="utf-8")

        (OUT_DIR / "requests.json").write_text(
            json.dumps(requests_log, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (OUT_DIR / "websockets.json").write_text(
            json.dumps(websocket_log, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (OUT_DIR / "dom_candidates.json").write_text(
            json.dumps(tables_info, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        print(f"[ok] dumped to: {OUT_DIR}")
        browser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
