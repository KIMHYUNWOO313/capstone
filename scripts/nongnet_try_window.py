# -*- coding: utf-8 -*-
"""농넷 일주일 윈도우 수집 디버그.

  python scripts/nongnet_try_window.py            # 9개 품목 × 7일치
  python scripts/nongnet_try_window.py 양파 1키로 상 7
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import django

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from dashboard.agri_nongnet import TARGETS, fetch_nongnet_window  # noqa: E402


def main() -> int:
    if len(sys.argv) >= 5:
        item, unit, grade, days = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
        targets = ((item, unit, grade),)
    else:
        targets = TARGETS
        days = int(sys.argv[1]) if len(sys.argv) > 1 else 7

    out = fetch_nongnet_window(days=days, targets=targets)
    print(f"got {len(out)} records")
    print(json.dumps(out, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
