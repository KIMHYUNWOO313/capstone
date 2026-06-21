# -*- coding: utf-8 -*-
"""외부 API 자동 수집 (KAMIS·OPINET·KMA·NongNet) → Firebase `API` + Django ApiSnapshot.

사용 예:
  매일 1회:                 python manage.py fetch_external_data --daily
  10일에 한 번(기상):        python manage.py fetch_external_data --kma-block
  농넷만 수동 실행:          python manage.py fetch_external_data --nongnet
  특정 일자만:               python manage.py fetch_external_data --date 2025-04-01 --daily
  매일+10일+농넷 동시 실행:  python manage.py fetch_external_data --daily --kma-block --nongnet

스케줄러(권장 — apps.py 통해 자동 실행):
  매일 00:00 (Asia/Seoul)   KAMIS·OPINET·NongNet, 10일마다 KMA 블록
"""
from __future__ import annotations

from datetime import date, datetime, timedelta

from django.core.management.base import BaseCommand, CommandError

from dashboard.agri_external import (
    fetch_kamis_daily,
    fetch_kma_asos_block,
    fetch_opinet_daily,
)
from dashboard.agri_external_store import save_many
from dashboard.agri_nongnet import fetch_nongnet_daily, fetch_nongnet_window


class Command(BaseCommand):
    help = "KAMIS/OPINET/KMA/NongNet 외부 API 자동 수집기"

    def add_arguments(self, parser):
        parser.add_argument(
            "--date",
            type=str,
            default="",
            help="기준 일자(YYYY-MM-DD). 비우면 어제(전일)",
        )
        parser.add_argument(
            "--daily",
            action="store_true",
            help="KAMIS·OPINET 일별 수집(매일 호출)",
        )
        parser.add_argument(
            "--kma-block",
            action="store_true",
            help="기상청 ASOS 일자료 10일치 블록 수집(10일 주기 호출)",
        )
        parser.add_argument(
            "--nongnet",
            action="store_true",
            help="농넷(가락시장 경락가격) Playwright 크롤링(매일 호출, --date 또는 어제 1일)",
        )
        parser.add_argument(
            "--nongnet-days",
            type=int,
            default=0,
            help="농넷 윈도우 수집 — 종료일(default=어제) 포함 N일치를 한 번에 수집",
        )

    def handle(self, *args, **options):
        if options["date"]:
            try:
                base = datetime.strptime(options["date"], "%Y-%m-%d").date()
            except ValueError as exc:
                raise CommandError(f"--date 형식 오류: {exc}") from exc
        else:
            base = date.today() - timedelta(days=1)

        nongnet_days = int(options.get("nongnet_days") or 0)
        if not (
            options["daily"]
            or options["kma_block"]
            or options["nongnet"]
            or nongnet_days > 0
        ):
            raise CommandError(
                "--daily 또는 --kma-block 또는 --nongnet 또는 --nongnet-days 중 하나 이상 지정"
            )

        records: list[tuple[str, list]] = []
        if options["daily"]:
            records.append(("kamis", fetch_kamis_daily(base)))
            records.append(("opinet", fetch_opinet_daily(base)))
        if options["kma_block"]:
            records.append(("kma", fetch_kma_asos_block(base, days=10)))
        if nongnet_days > 0:
            records.append(
                ("nongnet", fetch_nongnet_window(days=nongnet_days, end_date=base))
            )
        elif options["nongnet"]:
            records.append(("nongnet", fetch_nongnet_daily(target_date=base)))

        counts = save_many(records)
        for src, n in counts.items():
            self.stdout.write(f"{src}: 저장 {n}건 (기준일 {base.isoformat()})")
        self.stdout.write(self.style.SUCCESS("완료"))
