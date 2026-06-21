# -*- coding: utf-8 -*-
"""재정경제부 일일경제지표 게시판 크롤링.

사용 예:
  최근 1페이지(약 10건)만:           python manage.py fetch_mofe_daily --pages 1
  2026년만:                           python manage.py fetch_mofe_daily --pages 30
  2015년까지 전체 백필:               python manage.py fetch_mofe_daily --pages 651 --stop-year 2015
  일부 페이지 범위만(예: 100~120):     python manage.py fetch_mofe_daily --start-page 100 --pages 21
  이미 있는 일자도 다시 받기:          python manage.py fetch_mofe_daily --pages 5 --overwrite

출력:
  data/mofe_daily/hwp/<YYYYMMDD>.hwp
  data/mofe_daily/html/<YYYYMMDD>/index.xhtml
  data/mofe_daily/csv/<YYYYMMDD>.csv
  data/mofe_daily/index.csv  (메타 누적)
"""
from __future__ import annotations

import logging

from django.core.management.base import BaseCommand

from dashboard.mofe_daily import CrawlOptions, crawl, rebuild_wide_csvs

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "재정경제부(MOFE) 일일경제지표 게시판 → HWP → CSV 크롤러"

    def add_arguments(self, parser):
        parser.add_argument(
            "--start-page",
            type=int,
            default=1,
            help="시작 페이지 번호(1부터)",
        )
        parser.add_argument(
            "--pages",
            type=int,
            default=None,
            help="가져올 페이지 수(미지정 시 stop-year 까지)",
        )
        parser.add_argument(
            "--end-page",
            type=int,
            default=651,
            help="종료 페이지 번호 (--pages 미지정 시 사용; 기본 651 = 전체 추정)",
        )
        parser.add_argument(
            "--stop-year",
            type=int,
            default=2015,
            help="이 연도 미만의 게시물에 도달하면 중단 (기본 2015)",
        )
        parser.add_argument(
            "--overwrite",
            action="store_true",
            help="이미 다운로드/CSV 파일이 있어도 다시 받기",
        )
        parser.add_argument(
            "--sleep",
            type=float,
            default=0.4,
            help="요청 사이 sleep(초)",
        )
        parser.add_argument(
            "--rebuild-wide",
            action="store_true",
            help=(
                "이미 변환된 xhtml 들을 재파싱해 보고서 형태 wide CSV "
                "(csv/wide/<YYYYMMDD>.csv) 와 섹션별 시계열 wide CSV "
                "(wide/<섹션>.csv) 만 다시 만든다. 다운로드/HWP 변환은 하지 않음."
            ),
        )

    def handle(self, *args, **options):
        if options.get("rebuild_wide"):
            self.stdout.write("rebuild wide csvs from existing xhtml ...")
            result = rebuild_wide_csvs(overwrite=True)
            self.stdout.write(self.style.SUCCESS(f"done: {result}"))
            return

        end_page = options["end_page"]
        if options.get("pages"):
            end_page = options["start_page"] + options["pages"] - 1

        opts = CrawlOptions(
            start_page=options["start_page"],
            end_page=end_page,
            stop_year=options["stop_year"],
            overwrite=options["overwrite"],
            sleep_per_request=float(options["sleep"]),
            sleep_per_page=float(options["sleep"]) * 1.5,
        )
        self.stdout.write(
            f"crawl pages {opts.start_page}..{opts.end_page}, stop_year={opts.stop_year}, "
            f"overwrite={opts.overwrite}"
        )
        result = crawl(opts)
        self.stdout.write(self.style.SUCCESS(f"done: {result}"))
