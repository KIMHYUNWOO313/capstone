from django.shortcuts import render, redirect
import json
from urllib.parse import quote

from django.http import HttpResponse, JsonResponse
from django.views.decorators.http import require_http_methods
from .models import AWSInstance
from .report_pdf import build_report_pdf


def home(request):
    return render(request, 'home.html')


def products_page(request):
    products = [
        {
            "name": "감자 수미",
            "category": "구근류",
            "summary": "수미 품종 감자의 도매 가격 흐름과 계절 수급 변화를 확인합니다.",
            "factors": "저장 물량 · 산지 출하량 · 기상",
            "icon": "fa-seedling",
        },
        {
            "name": "고구마",
            "category": "구근류",
            "summary": "저장성과 산지 출하 주기에 따라 달라지는 고구마 가격 변화를 분석합니다.",
            "factors": "저장 상태 · 출하 시기 · 소비 수요",
            "icon": "fa-leaf",
        },
        {
            "name": "당근",
            "category": "채소류",
            "summary": "월동·봄 작형 전환과 산지 이동에 따른 당근 가격 흐름을 제공합니다.",
            "factors": "작형 전환 · 제주 산지 · 강수량",
            "icon": "fa-carrot",
        },
        {
            "name": "대파",
            "category": "양념채소",
            "summary": "기상 변화와 출하량에 민감한 대파의 단기 가격 변동을 추적합니다.",
            "factors": "한파 · 출하량 · 도매 수요",
            "icon": "fa-wheat-awn",
        },
        {
            "name": "배추",
            "category": "엽채류",
            "summary": "김장 수요와 작황 영향을 크게 받는 배추 가격 예측 정보를 제공합니다.",
            "factors": "김장철 · 작황 · 산지 이동",
            "icon": "fa-seedling",
        },
        {
            "name": "백다다기오이",
            "category": "과채류",
            "summary": "시설 재배 환경과 일조량에 따른 오이 가격 흐름을 확인합니다.",
            "factors": "시설 재배 · 일조량 · 기온",
            "icon": "fa-spa",
        },
        {
            "name": "사과 부사",
            "category": "과일류",
            "summary": "저장 사과 출하와 명절 수요에 따른 부사 가격 움직임을 분석합니다.",
            "factors": "저장 물량 · 명절 수요 · 품질",
            "icon": "fa-apple-whole",
        },
        {
            "name": "시금치",
            "category": "엽채류",
            "summary": "기온과 강수에 민감한 시금치 가격의 빠른 변화를 살펴봅니다.",
            "factors": "기온 · 강수 · 산지 출하",
            "icon": "fa-leaf",
        },
        {
            "name": "양파",
            "category": "양념채소",
            "summary": "저장 양파와 신규 출하 물량에 따른 중기 가격 흐름을 제공합니다.",
            "factors": "저장 물량 · 수입량 · 출하량",
            "icon": "fa-circle-nodes",
        },
    ]
    return render(request, 'products.html', {"products": products})


def news_page(request):
    from .agri_news import latest_news_snapshot, latest_reachable_news_snapshots

    try:
        snapshots = latest_reachable_news_snapshots()
        snapshot = snapshots[0] if snapshots else latest_news_snapshot()
    except Exception:
        snapshot = None
        snapshots = []
    return render(request, "news.html", {"snapshot": snapshot, "snapshots": snapshots})


def dashboard(request):
    instances = AWSInstance.objects.all()
    return render(request, 'dashboard.html', {
        'page': 'overview',
        'instances': instances,
    })


def prediction_page(request):
    instances = AWSInstance.objects.all()
    return render(request, 'dashboard.html', {
        'page': 'prediction',
        'instances': instances,
    })


def report_page(request):
    instances = AWSInstance.objects.all()
    return render(request, 'dashboard.html', {
        'page': 'report',
        'instances': instances,
    })


@require_http_methods(["POST"])
def report_pdf(request):
    try:
        payload = json.loads(request.body.decode("utf-8") or "{}")
        pdf_bytes = build_report_pdf(payload)
    except Exception as exc:
        return JsonResponse({"ok": False, "error": str(exc)}, status=400)

    filename = payload.get("filename") or "agri-price-report.pdf"
    response = HttpResponse(pdf_bytes, content_type="application/pdf")
    response["Content-Disposition"] = (
        'attachment; filename="agri-price-report.pdf"; '
        f"filename*=UTF-8''{quote(str(filename))}"
    )
    return response


def aws_page(request):
    instances = AWSInstance.objects.all()
    return render(request, 'dashboard.html', {
        'page': 'aws',
        'instances': instances,
    })


@require_http_methods(["POST"])
def add_instance(request):
    instance_type = request.POST.get('instance_type', '')
    count = int(request.POST.get('count', 1))
    description = request.POST.get('description', '')
    if instance_type:
        AWSInstance.objects.create(
            instance_type=instance_type,
            count=count,
            description=description
        )
    return redirect('aws_page')


def predict_users(request):
    # AI 모델 연동 전 더미 예측값 반환
    inst_count = AWSInstance.objects.count()
    dummy_prediction = 1000 + (inst_count * 150)
    return JsonResponse({
        'predicted_users': dummy_prediction,
        'message': 'AI 모델 연동 후 정확한 예측이 가능합니다.'
    })


def scaling_recommendation(request):
    # AI 모델 연동 전 더미 권장사항 반환
    CAPACITY_MAP = {
        't2.micro': 1, 't2.small': 2, 't2.medium': 4, 't2.large': 8,
        't2.xlarge': 16, 't2.2xlarge': 32, 't3.micro': 1, 't3.small': 2,
        't3.medium': 4, 't3.large': 8, 't3.xlarge': 16, 't3.2xlarge': 32
    }
    instances = AWSInstance.objects.all()
    total_capacity = sum(
        CAPACITY_MAP.get(inst.instance_type, 1) * inst.count
        for inst in instances
    )
    return JsonResponse({
        'recommendation': 'maintain' if 500 < total_capacity < 2000 else ('scale_up' if total_capacity < 500 else 'scale_down'),
        'total_capacity_gb': total_capacity,
        'message': 'AI 모델 연동 후 사용자 수 기반 정확한 스케일링 권장이 가능합니다.'
    })
