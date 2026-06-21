# -*- coding: utf-8 -*-
from __future__ import annotations

from django.core.management.base import BaseCommand
from django.utils import timezone

from dashboard.agri_chronos import run_chronos_forecast, supports_final_handoff_item
from dashboard.agri_data import list_item_ids
from dashboard.agri_store import get_latest_predict_batch, save_predict_batch


class Command(BaseCommand):
    help = "AWS final_handoff 모델 예측을 실행해 Django DB와 Firebase predict 컬렉션에 저장합니다."

    def add_arguments(self, parser):
        parser.add_argument(
            "--model",
            choices=["point", "probabilistic", "both"],
            default="both",
            help="실행할 모델. 기본값은 both.",
        )
        parser.add_argument(
            "--item-id",
            action="append",
            dest="item_ids",
            help="특정 item_id만 실행. 여러 번 지정 가능.",
        )
        parser.add_argument(
            "--skip-existing",
            action="store_true",
            help="오늘 이미 저장된 같은 모델 예측이 있으면 건너뜁니다.",
        )

    def handle(self, *args, **options):
        models = (
            ["point", "probabilistic"]
            if options["model"] == "both"
            else [options["model"]]
        )
        item_ids = options["item_ids"] or list_item_ids()
        item_ids = [item_id for item_id in item_ids if supports_final_handoff_item(item_id)]
        today = timezone.localdate()

        if not item_ids:
            self.stdout.write(self.style.WARNING("실행 가능한 final_handoff 매핑 품목이 없습니다."))
            return

        total_saved = 0
        total_skipped = 0
        total_failed = 0

        for item_id in item_ids:
            for model_name in models:
                if options["skip_existing"]:
                    batch = get_latest_predict_batch(item_id, model_name=model_name)
                    created_at = batch.get("created_at") if batch else None
                    if created_at and timezone.localtime(created_at).date() == today:
                        total_skipped += 1
                        self.stdout.write(f"SKIP {item_id} {model_name}: already saved today")
                        continue

                try:
                    origin, points = run_chronos_forecast(item_id, model=model_name)
                    save_predict_batch(
                        item_id=item_id,
                        origin_date=origin,
                        horizon=len(points),
                        points=points,
                        model_name=model_name,
                    )
                except Exception as exc:
                    total_failed += 1
                    self.stderr.write(self.style.ERROR(f"FAIL {item_id} {model_name}: {exc}"))
                    continue

                total_saved += 1
                self.stdout.write(
                    self.style.SUCCESS(
                        f"SAVED {item_id} {model_name}: origin={origin} horizon={len(points)}"
                    )
                )

        self.stdout.write(
            f"done saved={total_saved} skipped={total_skipped} failed={total_failed}"
        )
