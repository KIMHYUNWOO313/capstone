from django.db import models


class PatchNote(models.Model):
    title = models.CharField(max_length=200)
    content = models.TextField()
    version = models.CharField(max_length=50, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']


class AWSInstance(models.Model):
    INSTANCE_TYPES = [
        ('t2.micro', 't2.micro - 1 vCPU, 1GB RAM'),
        ('t2.small', 't2.small - 1 vCPU, 2GB RAM'),
        ('t2.medium', 't2.medium - 2 vCPU, 4GB RAM'),
        ('t2.large', 't2.large - 2 vCPU, 8GB RAM'),
        ('t2.xlarge', 't2.xlarge - 4 vCPU, 16GB RAM'),
        ('t2.2xlarge', 't2.2xlarge - 8 vCPU, 32GB RAM'),
        ('t3.micro', 't3.micro - 2 vCPU, 1GB RAM'),
        ('t3.small', 't3.small - 2 vCPU, 2GB RAM'),
        ('t3.medium', 't3.medium - 2 vCPU, 4GB RAM'),
        ('t3.large', 't3.large - 2 vCPU, 8GB RAM'),
        ('t3.xlarge', 't3.xlarge - 4 vCPU, 16GB RAM'),
        ('t3.2xlarge', 't3.2xlarge - 8 vCPU, 32GB RAM'),
    ]
    instance_type = models.CharField(max_length=50, choices=INSTANCE_TYPES)
    count = models.PositiveIntegerField(default=1)
    description = models.CharField(max_length=200, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)


class AgriActual(models.Model):
    """Firebase `actual` 컬렉션과 동일 개념 — Firestore 미설정 시 로컬 저장."""

    item_id = models.CharField(max_length=256, db_index=True)
    date = models.DateField(db_index=True)
    price_krw = models.FloatField()
    source = models.CharField(max_length=64, default="csv")

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["item_id", "date"],
                name="uniq_dashboard_agriactual_item_date",
            )
        ]


class ApiSnapshot(models.Model):
    """외부 API에서 받아온 원본 스냅샷 — Firebase `API` 컬렉션과 1:1.

    source: 'kamis' | 'opinet' | 'kma'
    snapshot_date: 데이터의 기준 날짜(일자), 같은 날 여러 항목은 item_key로 구분
    item_key: 같은 날 안에서 항목을 구분하는 키(품목·코드·관측소 등 자유)
    """

    SOURCE_CHOICES = [
        ("kamis", "KAMIS 농산물 가격"),
        ("opinet", "오피넷 유가"),
        ("kma", "기상청 일자료"),
        ("nongnet", "농넷 가락시장 경락가격"),
    ]

    source = models.CharField(max_length=16, choices=SOURCE_CHOICES, db_index=True)
    snapshot_date = models.DateField(db_index=True)
    item_key = models.CharField(max_length=128, default="default")
    payload = models.JSONField(default=dict)
    fetched_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["source", "snapshot_date", "item_key"],
                name="uniq_dashboard_apisnapshot_src_date_item",
            )
        ]
        indexes = [
            models.Index(fields=["source", "-snapshot_date"]),
        ]


class AgriPredictBatch(models.Model):
    """Firebase `predict` 컬렉션과 동일 개념 — 모델별 예측 배치."""

    item_id = models.CharField(max_length=256, db_index=True)
    model_name = models.CharField(max_length=32, default="probabilistic", db_index=True)
    origin_date = models.DateField()
    horizon = models.PositiveSmallIntegerField(default=10)
    points = models.JSONField(default=list)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["item_id", "-created_at"]),
            models.Index(fields=["item_id", "model_name", "-created_at"]),
        ]


class AgriNewsSnapshot(models.Model):
    """OpenAI Web Search로 수집한 농산물·기상 뉴스 요약 스냅샷."""

    query = models.TextField()
    summary = models.TextField(blank=True)
    articles = models.JSONField(default=list)
    raw_response = models.JSONField(default=dict)
    fetched_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        ordering = ["-fetched_at"]
        indexes = [
            models.Index(fields=["-fetched_at"]),
        ]
