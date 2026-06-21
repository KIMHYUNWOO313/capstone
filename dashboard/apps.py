import os
import sys
import threading

from django.apps import AppConfig


class DashboardConfig(AppConfig):
    name = 'dashboard'

    def ready(self):
        # 마이그레이션·매니지 명령에서는 스케줄러를 띄우지 않는다.
        argv = sys.argv if hasattr(sys, "argv") else []
        if any(
            cmd in argv
            for cmd in (
                "makemigrations",
                "migrate",
                "collectstatic",
                "shell",
                "createsuperuser",
                "test",
            )
        ):
            return
        # runserver 자동 reload 자식 프로세스에서만 시작
        if "runserver" in argv and os.environ.get("RUN_MAIN") != "true":
            return
        try:
            from . import scheduler

            scheduler.start()
        except Exception:
            pass
        self._warm_prediction_cache()

    def _warm_prediction_cache(self):
        def _warm():
            try:
                from .agri_data import list_item_ids
                from .agri_series import build_chart_series

                ids = list_item_ids()
                item_id = next((i for i in ids if i.startswith("cabbage_")), ids[0] if ids else None)
                if item_id:
                    build_chart_series(
                        item_id,
                        model_name="probabilistic",
                        include_prediction=True,
                        include_firestore_actuals=False,
                    )
            except Exception:
                pass

        threading.Thread(target=_warm, daemon=True).start()
