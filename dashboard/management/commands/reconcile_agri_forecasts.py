# -*- coding: utf-8 -*-
from django.core.management.base import BaseCommand

from dashboard.agri_data import list_item_ids
from dashboard.agri_reconcile import reconcile_item


class Command(BaseCommand):
    help = "모든 품목 예측 배치를 CSV/API 실제가로 정산(10일 경과분). cron에 등록하세요."

    def add_arguments(self, parser):
        parser.add_argument(
            "--model",
            choices=["point", "probabilistic", "both"],
            default="both",
        )

    def handle(self, *args, **options):
        models = (
            ["point", "probabilistic"]
            if options["model"] == "both"
            else [options["model"]]
        )
        for item_id in list_item_ids():
            for model_name in models:
                r = reconcile_item(item_id, model_name=model_name)
                self.stdout.write(f"{item_id} {model_name}: {r}")
