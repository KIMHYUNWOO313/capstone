# -*- coding: utf-8 -*-
"""백그라운드 스케줄러 — 외부 API/뉴스 자동 수집.

매일 KAMIS+OPINET을 수집하고, 마지막 수집일이 10일 이상 지났으면
기상청 ASOS 10일치 블록도 함께 수집한다.

런서버/배포 프로세스 시작 시 한 번만 동작한다.
"""
from __future__ import annotations

import logging
import os
import threading
from datetime import date, timedelta

from django.conf import settings

logger = logging.getLogger(__name__)

_scheduler = None
_started_lock = threading.Lock()
_started = False


def _kma_block_due(today: date) -> bool:
    """기상청 10일 블록이 새로 필요한지 판단(가장 최근 KMA 스냅샷 날짜로부터 10일 이상)."""
    try:
        from .models import ApiSnapshot

        last = (
            ApiSnapshot.objects.filter(source="kma")
            .order_by("-snapshot_date")
            .values_list("snapshot_date", flat=True)
            .first()
        )
    except Exception as exc:
        logger.warning("KMA due check failed: %s", exc)
        return True
    if last is None:
        return True
    return (today - last) >= timedelta(days=10)


def run_daily_collection() -> dict:
    """전일 기준으로 KAMIS+OPINET+NongNet (+필요 시 KMA 10일) 수집해 저장."""
    from .agri_external import (
        fetch_kamis_daily,
        fetch_kma_asos_block,
        fetch_opinet_daily,
    )
    from .agri_external_store import save_many
    from .agri_nongnet import fetch_nongnet_daily

    today = date.today()
    base = today - timedelta(days=1)
    records = [
        ("kamis", fetch_kamis_daily(base)),
        ("opinet", fetch_opinet_daily(base)),
        ("nongnet", fetch_nongnet_daily()),
    ]
    if _kma_block_due(today):
        records.append(("kma", fetch_kma_asos_block(base, days=10)))
    counts = save_many(records)
    logger.info("daily collection done base=%s counts=%s", base.isoformat(), counts)
    return {"base_date": base.isoformat(), "counts": counts}


def run_news_collection() -> dict:
    """OpenAI Web Search로 농산물·기상 뉴스를 수집해 저장."""
    if not (getattr(settings, "OPENAI_API_KEY", "") or "").strip():
        logger.info("skip agri news collection: OPENAI_API_KEY is empty")
        return {"skipped": "OPENAI_API_KEY empty"}
    from .agri_news import fetch_agri_news_with_web_search

    snapshot = fetch_agri_news_with_web_search()
    count = len(snapshot.articles or [])
    logger.info("agri news collection done fetched_at=%s articles=%s", snapshot.fetched_at, count)
    return {"fetched_at": snapshot.fetched_at.isoformat(), "articles": count}


def start():
    """프로세스 당 1회만 스케줄러를 시작."""
    global _scheduler, _started
    if not getattr(settings, "AGRI_AUTO_FETCH_ENABLED", True):
        return
    # runserver 자동리로더가 만든 자식 프로세스에서만 실행되도록 가드
    # (RUN_MAIN=true 가 자식, 부모는 None)
    if os.environ.get("RUN_MAIN") not in (None, "true"):
        return

    with _started_lock:
        if _started:
            return
        try:
            from apscheduler.schedulers.background import BackgroundScheduler
            from apscheduler.triggers.cron import CronTrigger
        except Exception as exc:
            logger.warning("APScheduler not available, skip auto fetch: %s", exc)
            return

        sched = BackgroundScheduler(timezone="Asia/Seoul")
        sched.add_job(
            run_daily_collection,
            trigger=CronTrigger(hour=0, minute=0),
            id="agri_daily_external_fetch",
            replace_existing=True,
            max_instances=1,
            coalesce=True,
            misfire_grace_time=3600,
        )
        sched.add_job(
            run_news_collection,
            trigger=CronTrigger(hour="0,3,6,9,12,15,18,21", minute=0),
            id="agri_openai_web_search_news",
            replace_existing=True,
            max_instances=1,
            coalesce=True,
            misfire_grace_time=1800,
        )
        sched.start()
        _scheduler = sched
        _started = True
        logger.info("agri scheduler started (external 00:00, news every 3h Asia/Seoul)")
