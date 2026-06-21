"""XAI 파이프라인 진입점.

사용 예:
    python run_xai.py
    python run_xai.py --items apple_fuji_box10kg_high,onion_kg1_high
    python run_xai.py --limit 3 --dry-run
    python run_xai.py --no-web-search
"""
import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

from config import OUTPUT_DIR, require_keys
from data_loaders.context_summary import load_full_baseline, summarize_item
from data_loaders.forecasts import load_forecasts, slice_item
from data_loaders.reports import load_all_reports, relevant_excerpts_for_crop
from data_loaders.weather_warn import fetch_warnings_by_item, format_warnings
from explain import explain_item
from prompt_builder import build_system_prompt, build_user_prompt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--items", type=str, default=None,
                   help="콤마 구분 item_id 목록. 미지정시 전체.")
    p.add_argument("--limit", type=int, default=None,
                   help="처리할 품목 수 상한 (테스트용).")
    p.add_argument("--dry-run", action="store_true",
                   help="프롬프트만 조립 후 outputs/dry_run/에 저장, GPT 호출 안 함.")
    p.add_argument("--no-web-search", action="store_true",
                   help="web_search 도구 비활성화.")
    p.add_argument("--skip-warnings", action="store_true",
                   help="기상특보 API 호출 건너뛰기.")
    return p.parse_args()


def _crop_from_item(item_id: str) -> str:
    parts = item_id.split("_")
    head = parts[0]
    if head in {"cucumber", "potato", "perilla", "garlic", "napa", "crown", "sweetpotato"}:
        return "_".join(parts[:2])
    return head


def main():
    args = parse_args()

    if not args.dry_run:
        require_keys()

    print("[1/6] forecast 로드...")
    forecasts = load_forecasts()
    all_items = sorted(forecasts["chronos2"]["item_id"].unique())
    if args.items:
        wanted = [s.strip() for s in args.items.split(",") if s.strip()]
        items = [i for i in wanted if i in all_items]
        missing = sorted(set(wanted) - set(items))
        if missing:
            print(f"  경고: 미존재 item_id 무시: {missing}")
    else:
        items = all_items
    if args.limit:
        items = items[: args.limit]
    print(f"  대상 품목 수: {len(items)}")

    print("[2/6] 365일 context 로드...")
    full = load_full_baseline()

    if args.skip_warnings:
        print("[3/6] (skipped) 기상특보")
        warnings_by_item = {}
    else:
        print("[3/6] 기상특보 API 호출...")
        try:
            warnings_by_item = fetch_warnings_by_item()
        except Exception as e:
            print(f"  특보 호출 실패: {e} — 빈 결과로 진행")
            warnings_by_item = {}

    print("[4/6] 농업 월보 PDF 로드...")
    reports = load_all_reports()
    print(f"  로드된 PDF: {list(reports.keys())}")

    print("[5/6] system prompt 빌드...")
    sys_prompt = build_system_prompt()

    OUTPUT_DIR.mkdir(exist_ok=True)
    today = _dt.date.today().strftime("%Y%m%d")
    if args.dry_run:
        out_path = OUTPUT_DIR / "dry_run" / f"prompts_{today}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_path = OUTPUT_DIR / f"explanations_{today}.json"

    print(f"[6/6] {'프롬프트 저장' if args.dry_run else '품목별 GPT 호출'}...")
    results: dict[str, dict] = {}
    for i, item_id in enumerate(items, start=1):
        crop = _crop_from_item(item_id)
        item_fc = slice_item(forecasts, item_id)
        ctx = summarize_item(full, item_id)
        warns = warnings_by_item.get(item_id, [])
        warns_block = format_warnings(warns)
        excerpt = relevant_excerpts_for_crop(reports, crop)

        user_prompt = build_user_prompt(
            item_id=item_id,
            item_forecasts=item_fc,
            context_summary=ctx,
            warnings_block=warns_block,
            report_excerpt=excerpt,
        )

        if args.dry_run:
            results[item_id] = {
                "system_prompt": sys_prompt,
                "user_prompt": user_prompt,
            }
            print(f"  ({i}/{len(items)}) {item_id}: dry-run 저장")
            continue

        try:
            res = explain_item(
                system_prompt=sys_prompt,
                user_prompt=user_prompt,
                enable_web_search=not args.no_web_search,
            )
            results[item_id] = res
            ws = res.get("_meta", {}).get("web_search_calls", 0)
            print(f"  ({i}/{len(items)}) {item_id}: OK (web_search={ws})")
        except Exception as e:
            results[item_id] = {"_error": str(e)}
            print(f"  ({i}/{len(items)}) {item_id}: ERROR — {e}")

    out_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()
