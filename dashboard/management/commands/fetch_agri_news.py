# -*- coding: utf-8 -*-
from django.core.management.base import BaseCommand, CommandError

from dashboard.agri_news import fetch_agri_news_with_web_search


class Command(BaseCommand):
    help = "OpenAI Web Search로 농산물·기상 뉴스를 수집해 저장"

    def handle(self, *args, **options):
        try:
            snapshot = fetch_agri_news_with_web_search()
        except Exception as exc:
            raise CommandError(str(exc)) from exc
        self.stdout.write(
            self.style.SUCCESS(
                f"뉴스 수집 완료: {snapshot.fetched_at.isoformat()} / {len(snapshot.articles or [])}건"
            )
        )
