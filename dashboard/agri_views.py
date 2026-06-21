# -*- coding: utf-8 -*-
import json
from time import monotonic

from django.http import JsonResponse
from django.views.decorators.http import require_GET, require_POST

from datetime import date, datetime, timedelta

from .agri_chronos import run_chronos_forecast, supports_final_handoff_item
from .agri_data import format_item_option_label, is_allowed_item_id, item_option_meta, list_item_ids
from .agri_external import (
    fetch_kamis_daily,
    fetch_kma_asos_block,
    fetch_opinet_daily,
)
from .agri_external_store import save_many
from .agri_gpt import explain_forecast, generate_analysis_image
from .agri_reconcile import reconcile_item
from .agri_series import build_chart_series
from .agri_store import get_latest_predict_batch, save_predict_batch

_CHART_CACHE_TTL_SECONDS = 120
_chart_cache: dict[tuple[str, str, bool, int | None], tuple[float, dict]] = {}


def _chart_cache_key(
    item_id: str,
    model: str,
    include_prediction: bool,
    past: int | None,
) -> tuple[str, str, bool, int | None]:
    return (item_id, model, include_prediction, past)


def _get_chart_cache(key: tuple[str, str, bool, int | None]) -> dict | None:
    hit = _chart_cache.get(key)
    if not hit:
        return None
    ts, payload = hit
    if monotonic() - ts > _CHART_CACHE_TTL_SECONDS:
        _chart_cache.pop(key, None)
        return None
    return dict(payload)


def _set_chart_cache(key: tuple[str, str, bool, int | None], payload: dict) -> None:
    _chart_cache[key] = (monotonic(), dict(payload))


def _clear_chart_cache(item_id: str, model: str) -> None:
    for key in list(_chart_cache.keys()):
        if key[0] == item_id and key[1] == model:
            _chart_cache.pop(key, None)


def _to_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _expand_prob_point_quantiles(point: dict) -> dict:
    anchors = {
        10: _to_float(point.get("p10")),
        20: _to_float(point.get("p20")),
        50: _to_float(point.get("p50", point.get("pred_krw"))),
        80: _to_float(point.get("p80")),
        90: _to_float(point.get("p90")),
    }
    out = {}
    for q in range(1, 100):
        k = f"p{q}"
        if point.get(k) is not None:
            out[k] = point.get(k)
            continue
        if q in anchors and anchors[q] is not None:
            out[k] = round(float(anchors[q]), 2)
            continue

        low_q = max((a for a, v in anchors.items() if a <= q and v is not None), default=None)
        high_q = min((a for a, v in anchors.items() if a >= q and v is not None), default=None)
        if low_q is not None and high_q is not None and low_q != high_q:
            lv = anchors[low_q]
            hv = anchors[high_q]
            ratio = (q - low_q) / (high_q - low_q)
            out[k] = round(lv + (hv - lv) * ratio, 2)
            continue
        if q < 10 and anchors[10] is not None and anchors[20] is not None:
            slope = (anchors[20] - anchors[10]) / 10.0
            out[k] = round(anchors[10] + slope * (q - 10), 2)
            continue
        if q > 90 and anchors[80] is not None and anchors[90] is not None:
            slope = (anchors[90] - anchors[80]) / 10.0
            out[k] = round(anchors[90] + slope * (q - 90), 2)
            continue
        out[k] = None
    return out


def _forecast_table(item_id: str, model: str, points: list[dict] | None = None) -> dict:
    if points is None:
        batch = get_latest_predict_batch(item_id, model_name=model)
        points = list((batch or {}).get("points") or [])
    source_item_id = ""
    if points:
        source_item_id = str(points[0].get("source_item_id") or item_id)

    if model == "point":
        rows = [
            {
                "item_id": p.get("source_item_id") or source_item_id or item_id,
                "date": p.get("date"),
                "step": idx + 1,
                "y_pred": p.get("y_pred", p.get("pred_krw")),
            }
            for idx, p in enumerate(points)
        ]
        return {
            "model": "point",
            "title": "점예측(3일)",
            "columns": ["ITEM_ID", "DATE", "STEP", "Y_PRED"],
            "rows": rows,
        }

    selected_quantiles = [1, 99]
    if not points:
        return {
            "model": "probabilistic",
            "title": "예측 결과 (백분위 예측가)",
            "orientation": "wide",
            "columns": ["백분위 (원)"],
            "rows": [],
        }

    day_columns: list[str] = []
    day_dates: list[str] = []
    expanded: list[dict] = []
    for idx, p in enumerate(points, start=1):
        date_str = str(p.get("date") or "")
        try:
            md = date_str[5:].replace("-", ".")
        except Exception:
            md = date_str
        day_columns.append(f"{md} (D+{idx})")
        day_dates.append(date_str)
        expanded.append(_expand_prob_point_quantiles(p))

    rows = []
    for q in selected_quantiles:
        label = "P50 (중앙값)" if q == 50 else f"P{q}"
        row = {"label": label}
        for idx, q_dict in enumerate(expanded):
            try:
                value = float(q_dict.get(f"p{q}"))
            except (TypeError, ValueError):
                value = None
            col_key = day_columns[idx]
            row[col_key] = round(value) if value is not None else None
        rows.append(row)

    return {
        "model": "probabilistic",
        "title": "예측 결과 (백분위 예측가)",
        "orientation": "wide",
        "columns": ["백분위 (원)", *day_columns],
        "dates": day_dates,
        "rows": rows,
    }


@require_GET
def agri_items(request):
    try:
        ids = list_item_ids()
    except Exception as e:
        return JsonResponse({"ok": False, "error": str(e)}, status=500)
    items = [
        {
            **item_option_meta(i),
            "forecast_supported": supports_final_handoff_item(i),
        }
        for i in ids
    ]
    return JsonResponse({"ok": True, "items": items})


@require_GET
def agri_chart(request):
    item_id = request.GET.get("item_id") or ""
    model = request.GET.get("model") or "probabilistic"
    include_prediction = request.GET.get("include_prediction", "").lower() in (
        "1",
        "true",
        "yes",
    )
    past_raw = request.GET.get("past_days")
    past = int(past_raw) if past_raw else None
    if not item_id:
        return JsonResponse({"ok": False, "error": "item_id required"}, status=400)
    if model not in {"point", "probabilistic"}:
        return JsonResponse({"ok": False, "error": "지원하지 않는 모델입니다."}, status=400)
    if not is_allowed_item_id(item_id):
        return JsonResponse({"ok": False, "error": "허용되지 않은 품목입니다."}, status=400)
    cache_key = _chart_cache_key(item_id, model, include_prediction, past)
    if not include_prediction:
        cached = _get_chart_cache(cache_key)
        if cached is not None:
            return JsonResponse({"ok": True, **cached, "served_from_cache": True})
    try:
        data = build_chart_series(
            item_id,
            past_days=past,
            model_name=model,
            include_prediction=include_prediction,
            include_firestore_actuals=True,
        )
        if include_prediction:
            data["forecast_table"] = _forecast_table(item_id, model)
        if not include_prediction:
            _set_chart_cache(cache_key, data)
    except Exception as e:
        return JsonResponse({"ok": False, "error": str(e)}, status=400)
    return JsonResponse({"ok": True, **data})


@require_POST
def agri_run_forecast(request):
    try:
        body = json.loads(request.body.decode() or "{}")
    except json.JSONDecodeError:
        body = {}
    item_id = body.get("item_id") or request.POST.get("item_id")
    model = body.get("model") or request.POST.get("model") or "probabilistic"
    force = bool(body.get("force", False))
    if not item_id:
        return JsonResponse({"ok": False, "error": "item_id required"}, status=400)
    if model not in {"point", "probabilistic"}:
        return JsonResponse({"ok": False, "error": "지원하지 않는 모델입니다."}, status=400)
    if not is_allowed_item_id(item_id):
        return JsonResponse({"ok": False, "error": "허용되지 않은 품목입니다."}, status=400)
    try:
        reconcile_item(item_id, model_name=model)
        batch = get_latest_predict_batch(item_id, model_name=model)
        latest_anchor = build_chart_series(
            item_id,
            model_name=model,
            include_prediction=False,
        ).get("last_csv_date")
        latest_anchor_date = (
            date.fromisoformat(latest_anchor) if latest_anchor else None
        )
        batch_origin = batch.get("origin_date") if batch else None
        used_cached = (
            batch is not None
            and not force
            and (
                latest_anchor_date is None
                or batch_origin is None
                or batch_origin >= latest_anchor_date
            )
        )
        if used_cached:
            origin = batch["origin_date"]
            points = batch["points"]
        else:
            origin, points = run_chronos_forecast(item_id, model=model)
            save_predict_batch(
                item_id,
                origin,
                len(points),
                points,
                model_name=model,
            )
            _clear_chart_cache(item_id, model)
        chart = build_chart_series(item_id, model_name=model)
        table = _forecast_table(item_id, model, points)
    except Exception as e:
        return JsonResponse({"ok": False, "error": str(e)}, status=500)
    return JsonResponse(
        {
            "ok": True,
            "origin_date": origin.isoformat(),
            "horizon": len(points),
            "model": model,
            "used_cached": used_cached,
            "chart": chart,
            "forecast_table": table,
        }
    )


@require_POST
def agri_reconcile(request):
    try:
        body = json.loads(request.body.decode() or "{}")
    except json.JSONDecodeError:
        body = {}
    item_id = body.get("item_id") or request.POST.get("item_id")
    model = body.get("model") or request.POST.get("model") or "probabilistic"
    if not item_id:
        return JsonResponse({"ok": False, "error": "item_id required"}, status=400)
    if model not in {"point", "probabilistic"}:
        return JsonResponse({"ok": False, "error": "지원하지 않는 모델입니다."}, status=400)
    if not is_allowed_item_id(item_id):
        return JsonResponse({"ok": False, "error": "허용되지 않은 품목입니다."}, status=400)
    try:
        result = reconcile_item(item_id, model_name=model)
        _clear_chart_cache(item_id, model)
        chart = build_chart_series(item_id, model_name=model)
        table = _forecast_table(item_id, model)
    except Exception as e:
        return JsonResponse({"ok": False, "error": str(e)}, status=500)
    return JsonResponse({"ok": True, "reconcile": result, "chart": chart, "forecast_table": table})


@require_POST
def agri_explain(request):
    try:
        body = json.loads(request.body.decode() or "{}")
    except json.JSONDecodeError:
        body = {}
    item_id = body.get("item_id") or ""
    model = body.get("model") or "probabilistic"
    if not item_id:
        return JsonResponse({"ok": False, "error": "item_id required"}, status=400)
    if model not in {"point", "probabilistic"}:
        return JsonResponse({"ok": False, "error": "지원하지 않는 모델입니다."}, status=400)
    if not is_allowed_item_id(item_id):
        return JsonResponse({"ok": False, "error": "허용되지 않은 품목입니다."}, status=400)
    try:
        chart = build_chart_series(item_id, model_name=model)
        batch = get_latest_predict_batch(item_id, model_name=model)
        meta = None
        if batch:
            meta = {
                "origin_date": batch["origin_date"].isoformat()
                if hasattr(batch["origin_date"], "isoformat")
                else str(batch["origin_date"]),
                "horizon": batch.get("horizon"),
                "model": model,
            }
        result = explain_forecast(item_id, chart, meta)
        if isinstance(result, dict):
            text = result.get("explanation") or ""
            plain = result.get("explanation_plain") or text
        else:
            text = str(result or "")
            plain = text
    except Exception as e:
        return JsonResponse({"ok": False, "error": str(e)}, status=500)
    return JsonResponse({"ok": True, "explanation": text, "explanation_plain": plain})


@require_POST
def agri_analysis_image(request):
    try:
        body = json.loads(request.body.decode() or "{}")
    except json.JSONDecodeError:
        body = {}
    item_id = body.get("item_id") or ""
    model = body.get("model") or "probabilistic"
    if not item_id:
        return JsonResponse({"ok": False, "error": "item_id required"}, status=400)
    if model not in {"point", "probabilistic"}:
        return JsonResponse({"ok": False, "error": "지원하지 않는 모델입니다."}, status=400)
    if not is_allowed_item_id(item_id):
        return JsonResponse({"ok": False, "error": "허용되지 않은 품목입니다."}, status=400)
    try:
        chart = build_chart_series(item_id, model_name=model)
        metrics = chart.get("metrics") or {}
        meta = item_option_meta(item_id)
        points = chart.get("points") or []
        predict = [p for p in points if p.get("kind") == "predict" and p.get("value") is not None]
        direction = "예측값 부족"
        if len(predict) >= 2:
            first_value = float(predict[0]["value"])
            last_value = float(predict[-1]["value"])
            if last_value > first_value:
                direction = "상승"
            elif last_value < first_value:
                direction = "하락"
            else:
                direction = "횡보"
        batch = get_latest_predict_batch(item_id, model_name=model)
        origin_date = ""
        if batch and batch.get("origin_date"):
            origin = batch["origin_date"]
            origin_date = origin.isoformat() if hasattr(origin, "isoformat") else str(origin)
        prompt_context = json.dumps(
            {
                "label": meta.get("label") or format_item_option_label(item_id),
                "crop": meta.get("crop"),
                "unit": meta.get("unit"),
                "grade": meta.get("grade"),
                "current_actual_krw": metrics.get("current_actual_krw"),
                "prediction_average_krw": metrics.get("prediction_average_krw"),
                "volatility_pct": metrics.get("volatility_pct"),
                "direction": direction,
                "origin_date": origin_date or chart.get("last_csv_date"),
                "analysis_text": body.get("analysis_text") or "",
            },
            ensure_ascii=False,
        )
        image = generate_analysis_image(item_id, prompt_context)
    except Exception as e:
        return JsonResponse({"ok": False, "error": str(e)}, status=500)
    return JsonResponse({"ok": True, "image": image})


def _parse_date_param(s):
    if s:
        try:
            return datetime.strptime(s, "%Y-%m-%d").date()
        except ValueError:
            pass
    return date.today() - timedelta(days=1)


@require_POST
def agri_external_fetch(request):
    """KAMIS/OPINET(매일) + 선택적 KMA(10일) 호출 → Firebase `API` 컬렉션 저장."""
    try:
        body = json.loads(request.body.decode() or "{}")
    except json.JSONDecodeError:
        body = {}
    base = _parse_date_param(body.get("date"))
    do_daily = bool(body.get("daily", True))
    do_kma_block = bool(body.get("kma_block", False))

    if not (do_daily or do_kma_block):
        return JsonResponse(
            {"ok": False, "error": "daily 또는 kma_block 중 하나는 true여야 합니다."},
            status=400,
        )

    records = []
    if do_daily:
        records.append(("kamis", fetch_kamis_daily(base)))
        records.append(("opinet", fetch_opinet_daily(base)))
    if do_kma_block:
        records.append(("kma", fetch_kma_asos_block(base, days=10)))
    counts = save_many(records)
    return JsonResponse(
        {
            "ok": True,
            "base_date": base.isoformat(),
            "saved_counts": counts,
            "firestore_collection": "API",
        }
    )
