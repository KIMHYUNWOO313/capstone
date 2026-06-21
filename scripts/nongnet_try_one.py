# -*- coding: utf-8 -*-
"""농넷 스크레이퍼를 1품목만 시도하는 디버그 스크립트.

  python scripts/nongnet_try_one.py "양파" "1키로" "상"
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

# 콘솔 인코딩(Windows cp949) 우회
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from dashboard.agri_nongnet import fetch_nongnet_daily, TARGETS  # noqa: E402


def main() -> int:
    if len(sys.argv) >= 4:
        item, unit, grade = sys.argv[1], sys.argv[2], sys.argv[3]
        targets = ((item, unit, grade),)
    else:
        targets = TARGETS

    out = fetch_nongnet_daily(targets=targets)
    print(f"got {len(out)} records")
    print(json.dumps(out, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
